"""
Pipeline job: runs steps 0-2 in order, then forks into a graph branch and a vector branch.

    run_job(job_id, repo_url, tenant_id, workspaces_root, ctx=None, mode="full")

Flow (mode="full"; "incremental" raises NotImplementedError)
-------------------------------------------------------------
    0. clone      clone_repo_to_workspace -> RepoHandle. A CloneError fails the job.
    1. parse      parse_repo (tree-sitter). Any other exception fails the job.
    2. scip       run_scip under a process-wide semaphore, then build_symbol_map and
                  write_symbol_map. SCIP failing completely is NOT fatal: the symbol
                  map then uses tree-sitter boundaries and the resolver falls back.
    fork          only after run_scip has fully returned (scip-typescript may create and
                  delete a tsconfig.json inside the repo while it runs).
        graph     graph_branch(records, scip, handle, gaps)   <- needs records + ScipResult
        vector    vector_branch(repo_root, artifacts_dir, tenant_id, gaps)
                  <- sees the world ONLY through artifacts/symbol_map.json
    join          merge_gaps(main, graph, vector) -> artifacts/gaps.json
    finally       cleanup_workspace(keep_artifacts=True), even on exceptions.

Each branch runs through `_safe`: an exception is turned into a `branch_failed` Gap and the
other branch carries on. Each branch has its OWN gap list, so no locks are needed.
Job status: both branches ok -> "ok", one failed -> "partial", both failed -> "failed".

Files in workspaces/{job_id}/artifacts/
---------------------------------------
    repo_handle.json  (clone)       symbol_map.json  (step 2)      vector_stub.json (vector stub)
    gaps.json         (join)        status.json      (every stage transition, atomic)

status.json (poll it from another thread or process; every write is a temp file + rename)
    { "job_id", "tenant_id", "mode", "state": "pending|running|done|failed",
      "result": null | "ok" | "partial" | "failed", "error": "",
      "started_at", "updated_at", "finished_at",
      "stages": { "<clone|parse|scip|symbol_map|graph|vector|join>":
                  {"state": "pending|running|done|failed", "started_at", "finished_at", "detail"} } }
    state is "done" for result ok or partial, "failed" for result failed. Timestamps are UTC ISO-8601.

Concurrency
-----------
Jobs with different job_ids are independent (separate workspace dirs). The same job_id twice
at once is rejected: the second call returns a failed JobResult and touches nothing, so it
cannot overwrite the first job's status or delete its checkout. SCIP indexers are heavy, so
at most PEACH_MAX_CONCURRENT_SCIP (default 2) run at once in this process.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# _validate_job_id / _workspace_paths are clone.py's own path rules (id pattern, no escape
# from the workspaces root); reusing them keeps one definition of "a valid workspace".
from .clone import (CloneError, RepoHandle, _validate_job_id, _workspace_paths,
                    cleanup_workspace, clone_repo_to_workspace)
from .gaps import Gap, emit, merge_gaps, write_gaps
from .obs import get_logger, log
from .parser import parse_repo
from .resolver import resolve_calls
from .scip_check import _LANGUAGE_TO_INDEXER, ScipResult, run_scip
from .symbol_map import (COVERAGE_VALUES, SYMBOL_MAP_FILENAME, build_symbol_map,
                         load_symbol_map, write_symbol_map)

LOG = get_logger("pipeline_job")

STATUS_FILENAME = "status.json"
GAPS_FILENAME = "gaps.json"
VECTOR_STUB_FILENAME = "vector_stub.json"

SCIP_CONCURRENCY_ENV = "PEACH_MAX_CONCURRENT_SCIP"
DEFAULT_MAX_CONCURRENT_SCIP = 2

STAGES = ("clone", "parse", "scip", "symbol_map", "graph", "vector", "join")

_GRAPH_IMPACT = "The call graph (and every graph-based finding) is missing for this job."
_VECTOR_IMPACT = "Vector search over this repo is unavailable for this job."


@dataclass
class JobResult:
    """What `run_job` returns. `status` is "ok" | "partial" | "failed".

    graph_status / vector_status are "ok" | "failed" | "not_run" (the stage was never
    reached). `error` is set when the job failed before the fork. `artifacts_dir` is
    workspaces/{job_id}/artifacts, which outlives the job; the checkout does not.
    `gaps` is the merged gap list, the same content as gaps.json."""
    job_id: str
    status: str = "failed"
    error: str = ""
    commit_sha: str = ""
    graph_status: str = "not_run"
    vector_status: str = "not_run"
    artifacts_dir: Optional[Path] = None
    gaps: List[Gap] = field(default_factory=list)


# --------------------------------------------------------------------------- files ----

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload) -> None:
    """Write `payload` as JSON via a temp file in the same directory + `os.replace`, so a
    reader sees the old file or the new one, never a partial one. Raises OSError/TypeError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _StatusWriter:
    """Owns status.json. Thread-safe: the two branches report from their own threads.
    A failed write is logged and ignored: losing a progress report must not fail a job."""

    def __init__(self, path: Path, job_id: str, tenant_id: str, mode: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._doc = {
            "job_id": job_id, "tenant_id": tenant_id, "mode": mode,
            "state": "pending", "result": None, "error": "",
            "started_at": None, "updated_at": _now(), "finished_at": None,
            "stages": {name: {"state": "pending", "started_at": None,
                              "finished_at": None, "detail": ""} for name in STAGES},
        }
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        self._doc["updated_at"] = _now()
        try:
            _write_json_atomic(self._path, self._doc)
        except (OSError, TypeError) as exc:
            log(LOG, "warning", "could not write status.json", path=str(self._path),
                error=f"{type(exc).__name__}: {exc}")

    def start_job(self) -> None:
        with self._lock:
            self._doc["state"] = "running"
            self._doc["started_at"] = _now()
            self._flush()

    def stage(self, name: str, state: str, detail: str = "") -> None:
        with self._lock:
            entry = self._doc["stages"][name]
            entry["state"] = state
            if state == "running":
                entry["started_at"] = _now()
            else:
                entry["finished_at"] = _now()
            if detail:
                entry["detail"] = detail
            self._flush()

    def finish(self, result: str, error: str = "") -> None:
        with self._lock:
            self._doc["state"] = "failed" if result == "failed" else "done"
            self._doc["result"] = result
            self._doc["error"] = error
            self._doc["finished_at"] = _now()
            self._flush()


# --------------------------------------------------------------------------- SCIP slot ----

_scip_sem: Optional[threading.BoundedSemaphore] = None
_scip_sem_lock = threading.Lock()


def _scip_limit() -> int:
    """PEACH_MAX_CONCURRENT_SCIP as a positive int; unset or invalid -> the default."""
    raw = os.environ.get(SCIP_CONCURRENCY_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_CONCURRENT_SCIP
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        log(LOG, "warning", "ignoring invalid SCIP concurrency limit",
            env=SCIP_CONCURRENCY_ENV, value=raw)
        return DEFAULT_MAX_CONCURRENT_SCIP
    return value


def _scip_semaphore() -> threading.BoundedSemaphore:
    """The process-wide SCIP semaphore, created on first use."""
    global _scip_sem
    with _scip_sem_lock:
        if _scip_sem is None:
            _scip_sem = threading.BoundedSemaphore(_scip_limit())
        return _scip_sem


# --------------------------------------------------------------------------- stages ----

def _run_scip_stage(records, repo_root: Path, ctx, gaps: list) -> ScipResult:
    """Stage 2a: run_scip inside the semaphore. Never raises and always returns a
    ScipResult. `run_scip` already swallows indexer failures; if it raises anyway, the
    languages it should have covered get a `scip_language_failed` gap and an EMPTY
    ScipResult is returned (not None: None would make the resolver run SCIP a second time)."""
    with _scip_semaphore():
        try:
            return run_scip(records, repo_root, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 - SCIP failing must not fail the job
            detail = f"{type(exc).__name__}: {exc}"
            log(LOG, "error", "run_scip raised; continuing without SCIP", error=detail)
            by_language: dict = {}
            for rec in records:
                if rec.language in _LANGUAGE_TO_INDEXER and not rec.is_config and not rec.skipped:
                    by_language[rec.language] = by_language.get(rec.language, 0) + 1
            for language, n in sorted(by_language.items()):
                emit(gaps, Gap(
                    stage="scip", scope="language", kind="scip_language_failed", count=n,
                    detail=f"{language}: SCIP stage raised {detail}",
                    impact="Calls and symbol spans in these files use the tree-sitter fallback."))
            return ScipResult()


def _symbol_map_stage(records, scip: ScipResult, handle: RepoHandle, artifacts_dir: Path,
                      gaps: list, ctx) -> None:
    """Stage 2b: build and write artifacts/symbol_map.json. Raises on failure."""
    symbol_map = build_symbol_map(records, scip, handle.commit_sha, gaps=gaps, ctx=ctx,
                                  repo_root=handle.root)
    write_symbol_map(symbol_map, artifacts_dir)


def run_later_graph_stages(records, edges, scip, handle, gaps, ctx=None) -> None:
    """STUB for the graph stages after call resolution (ast-grep, config, Joern, LLM, sync)."""
    log(LOG, "info", "later graph stages not wired", job_id=handle.job_id,
        files=len(records), edges=len(edges))


def graph_branch(records, scip, handle, gaps, ctx=None):
    """Graph branch: resolve calls with the job's ScipResult (SCIP is NOT run again), then
    hand over to the later graph stages. Returns the CallEdge list. May raise; `_safe`
    turns that into a `branch_failed` gap."""
    edges = resolve_calls(records, ctx=ctx, repo_root=handle.root, gaps=gaps, scip=scip)
    run_later_graph_stages(records, edges, scip, handle, gaps, ctx=ctx)
    return edges


def vector_branch(repo_root, artifacts_dir, tenant_id, gaps) -> dict:
    """Vector branch STUB. It takes no `records` and no ScipResult on purpose: the symbol
    map file is its only input from steps 1-2. Loads artifacts/symbol_map.json, logs file
    and symbol counts per coverage type and writes artifacts/vector_stub.json.
    `repo_root` and `gaps` are unused until chunking exists. Raises if the map is missing
    or invalid (FileNotFoundError / SymbolMapError)."""
    artifacts = Path(artifacts_dir)
    symbol_map = load_symbol_map(artifacts / SYMBOL_MAP_FILENAME)
    files_by_coverage = {c: 0 for c in COVERAGE_VALUES}
    symbols_by_coverage = {c: 0 for c in COVERAGE_VALUES}
    for entry in symbol_map["files"].values():
        files_by_coverage[entry["coverage"]] += 1
        symbols_by_coverage[entry["coverage"]] += len(entry["symbols"])
    summary = {
        "stub": True,
        "note": "Chunking and embedding are not implemented; this file only proves the "
                "vector branch can read symbol_map.json.",
        "tenant_id": tenant_id,
        "commit_sha": symbol_map["commit_sha"],
        "total_files": sum(files_by_coverage.values()),
        "total_symbols": sum(symbols_by_coverage.values()),
        "files_by_coverage": files_by_coverage,
        "symbols_by_coverage": symbols_by_coverage,
    }
    log(LOG, "info", "vector stub: symbol map loaded", tenant_id=tenant_id,
        files=summary["total_files"], symbols=summary["total_symbols"],
        **{f"files_{c}": n for c, n in files_by_coverage.items()},
        **{f"symbols_{c}": n for c, n in symbols_by_coverage.items()})
    _write_json_atomic(artifacts / VECTOR_STUB_FILENAME, summary)
    return summary


def _safe(stage: str, fn: Callable[[], object], gaps: list, impact: str) -> str:
    """Run one branch. Returns "ok", or "failed" after appending a `branch_failed` Gap
    for `stage` to `gaps`. Catches Exception only (Ctrl-C still stops the process)."""
    try:
        fn()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - one branch failing must not stop the other
        detail = f"{type(exc).__name__}: {exc}"
        log(LOG, "error", "branch failed", branch=stage, error=detail)
        emit(gaps, Gap(stage=stage, scope="repo", kind="branch_failed",
                       detail=f"{stage} branch raised {detail}", impact=impact))
        return "failed"


def _combine(graph_status: str, vector_status: str) -> str:
    failed = [s for s in (graph_status, vector_status) if s != "ok"]
    return "ok" if not failed else ("failed" if len(failed) == 2 else "partial")


def _fork(records, scip, handle, artifacts_dir, tenant_id, ctx, status: _StatusWriter,
          graph_gaps: list, vector_gaps: list, run_vector: bool):
    """Run both branches in two threads; returns (graph_status, vector_status)."""
    def tracked(name, fn, gaps, impact):
        status.stage(name, "running")
        outcome = _safe(name, fn, gaps, impact)
        status.stage(name, "done" if outcome == "ok" else "failed",
                     detail="" if outcome == "ok" else gaps[-1].detail)
        return outcome

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="peach-branch") as pool:
        graph_future = pool.submit(
            tracked, "graph",
            lambda: graph_branch(records, scip, handle, graph_gaps, ctx=ctx),
            graph_gaps, _GRAPH_IMPACT)
        vector_future = pool.submit(
            tracked, "vector",
            lambda: vector_branch(handle.root, artifacts_dir, tenant_id, vector_gaps),
            vector_gaps, _VECTOR_IMPACT) if run_vector else None
        graph_status = graph_future.result()
        vector_status = vector_future.result() if vector_future is not None else "failed"
    return graph_status, vector_status


# --------------------------------------------------------------------------- job ----

_active_jobs: set = set()
_active_lock = threading.Lock()


def _claim(job_id: str) -> bool:
    with _active_lock:
        if job_id in _active_jobs:
            return False
        _active_jobs.add(job_id)
        return True


def _release(job_id: str) -> None:
    with _active_lock:
        _active_jobs.discard(job_id)


def run_job(job_id: str, repo_url: str, tenant_id: str, workspaces_root, ctx=None,
            mode: str = "full", base_commit: Optional[str] = None) -> JobResult:
    """Run one analysis job (see the module docstring for the flow and the files).

    `mode` must be "full"; "incremental" raises NotImplementedError and anything else
    ValueError, before anything is created. `base_commit` is accepted for the future
    incremental mode and ignored. An invalid `job_id` raises CloneError (there is no safe
    place to report it). Every other failure is returned as `JobResult(status="failed")`
    with the reason in `error` and in status.json; this function does not raise for them.
    A job_id that is already running, or whose workspace already holds a checkout, is
    rejected as failed WITHOUT touching that workspace."""
    if mode == "incremental":
        raise NotImplementedError(
            "mode='incremental' is not implemented yet; use mode='full' "
            "(base_commit is accepted but ignored until it is.)")
    if mode != "full":
        raise ValueError(f"unknown mode {mode!r}; expected 'full'")
    job_id = _validate_job_id(job_id)
    _, repo_dir, artifacts_dir = _workspace_paths(workspaces_root, job_id)

    if not _claim(job_id):
        return JobResult(job_id=job_id, artifacts_dir=artifacts_dir,
                         error=f"Job {job_id} is already running in this process.")
    try:
        if os.path.lexists(repo_dir):
            return JobResult(job_id=job_id, artifacts_dir=artifacts_dir,
                             error=f"Workspace for job {job_id} already contains a cloned repo.")
        return _run_owned_job(job_id, repo_url, tenant_id, workspaces_root, artifacts_dir, ctx)
    finally:
        _release(job_id)


def _run_owned_job(job_id, repo_url, tenant_id, workspaces_root, artifacts_dir: Path,
                   ctx) -> JobResult:
    """The body of `run_job` once this call owns the job's workspace."""
    result = JobResult(job_id=job_id, artifacts_dir=artifacts_dir)
    main_gaps: list = []
    graph_gaps: list = []
    vector_gaps: list = []
    status: Optional[_StatusWriter] = None
    current = "clone"

    def fail(message: str) -> None:
        result.error = message
        result.gaps = merge_gaps(main_gaps, graph_gaps, vector_gaps)
        log(LOG, "error", "job failed", job_id=job_id, stage=current, error=message)
        if status is not None:
            status.stage(current, "failed", detail=message)
            status.finish("failed", error=message)

    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        status = _StatusWriter(artifacts_dir / STATUS_FILENAME, job_id, tenant_id, "full")
        status.start_job()

        # Stage 0
        status.stage("clone", "running")
        handle = clone_repo_to_workspace(repo_url, job_id, workspaces_root, ctx=ctx)
        result.commit_sha = handle.commit_sha
        status.stage("clone", "done")

        # Stage 1
        current = "parse"
        status.stage("parse", "running")
        records = parse_repo(handle.root, ctx=ctx, gaps=main_gaps)
        status.stage("parse", "done")

        # Stage 2: SCIP (bounded), then the symbol map the vector branch depends on
        current = "scip"
        status.stage("scip", "running")
        scip = _run_scip_stage(records, handle.root, ctx, main_gaps)
        status.stage("scip", "done")

        current = "symbol_map"
        status.stage("symbol_map", "running")
        run_vector = True
        try:
            _symbol_map_stage(records, scip, handle, artifacts_dir, main_gaps, ctx)
            status.stage("symbol_map", "done")
        except Exception as exc:  # noqa: BLE001 - no map means no vector branch, not no job
            run_vector = False
            detail = f"{type(exc).__name__}: {exc}"
            log(LOG, "error", "symbol map failed; vector branch skipped", error=detail)
            emit(main_gaps, Gap(
                stage="vector", scope="repo", kind="branch_failed",
                detail=f"symbol map could not be built or written ({detail}); "
                       f"the vector branch did not run",
                impact=_VECTOR_IMPACT))
            status.stage("symbol_map", "failed", detail=detail)
            status.stage("vector", "failed", detail="skipped: symbol map unavailable")

        # Fork: run_scip has fully returned, so nothing touches the repo any more.
        current = "graph"
        graph_status, vector_status = _fork(records, scip, handle, artifacts_dir, tenant_id,
                                            ctx, status, graph_gaps, vector_gaps, run_vector)
        result.graph_status, result.vector_status = graph_status, vector_status

        # Join
        current = "join"
        status.stage("join", "running")
        result.gaps = merge_gaps(main_gaps, graph_gaps, vector_gaps)
        join_detail = ""
        try:
            write_gaps(result.gaps, artifacts_dir / GAPS_FILENAME)
        except Exception as exc:  # noqa: BLE001 - result.gaps still carries them
            join_detail = f"gaps.json not written: {type(exc).__name__}: {exc}"
            log(LOG, "error", "could not write gaps.json", error=join_detail)
        result.status = _combine(graph_status, vector_status)
        status.stage("join", "done", detail=join_detail)
        status.finish(result.status)
        return result
    except Exception as exc:  # noqa: BLE001 - the job reports failure; it does not raise
        fail(str(exc) if isinstance(exc, CloneError) else f"{type(exc).__name__}: {exc}")
        return result
    except BaseException as exc:  # KeyboardInterrupt etc.: mark the status, then propagate
        fail(f"interrupted: {type(exc).__name__}")
        raise
    finally:
        try:
            cleanup_workspace(workspaces_root, job_id, keep_artifacts=True)
        except Exception as exc:  # noqa: BLE001 - cleanup trouble must not mask the result
            log(LOG, "error", "workspace cleanup failed", job_id=job_id,
                error=f"{type(exc).__name__}: {exc}")