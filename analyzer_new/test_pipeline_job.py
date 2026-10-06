"""Tests for peach.pipeline_job. Fakes only: no network, no git, no real indexers.

Patched in `pipeline_job`: clone_repo_to_workspace, parse_repo, run_scip. Real: build_symbol_map,
write_symbol_map, resolve_calls (wrapped in a spy), cleanup_workspace (wrapped in a spy), gaps.
"""
import hashlib
import inspect
import json
import threading
import time
from pathlib import Path

import pytest

from analyzer_new import pipeline_job as pj
from analyzer_new.clone import CloneError, RepoHandle, cleanup_workspace as real_cleanup
from analyzer_new.gaps import read_gaps
from analyzer_new.parser import CallSite, Definition, FileRecord
from analyzer_new.resolver import resolve_calls as real_resolve_calls
from analyzer_new.scip_check import ScipResult
from analyzer_new.symbol_map import load_symbol_map

JOB = "job-00001"


# ------------------------------------------------------------------ fakes ----

def sha_for(job_id: str) -> str:
    return hashlib.sha1(job_id.encode()).hexdigest()


def fake_records():
    return [
        FileRecord(path="a.py", language="python", parse_mode="treesitter",
                   definitions=[Definition("f", "f", "function", "a.py", 1, 2, 1, 4)],
                   calls=[CallSite(caller="<module>:a.py", callee_expr="f", line=4,
                                   arg_text="", file="a.py", callee_line=4, callee_col=0)]),
        FileRecord(path="b.js", language="javascript", parse_mode="treesitter",
                   definitions=[Definition("g", "g", "function", "b.js", 1, 1, 1, 9)]),
    ]


def scip_ok():
    return ScipResult(
        ref_map={("a.py", 4, 0): (("a.py", 1, 4),)},
        status={"indexed": ["python", "javascript"], "skipped": {},
                "covered_files": {"a.py", "b.js"}, "scip_print_available": True})


def scip_all_failed():
    return ScipResult(status={"indexed": [], "covered_files": set(),
                              "skipped": {"python": "boom", "javascript": "boom"},
                              "scip_print_available": True})


class Recorder:
    """Thread-safe-enough event log (list.append is atomic) plus captured call args."""
    def __init__(self):
        self.events = []
        self.resolve_kwargs = None
        self.cleanup_calls = []
        self.handles = {}

    def index(self, name):
        return self.events.index(name)


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()

    def fake_clone(url, job_id, workspaces_root, ctx=None):
        job_dir = Path(workspaces_root) / job_id
        repo = job_dir / "repo"
        (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        repo.mkdir()
        (repo / "a.py").write_text("def f():\n    return 1\n\nf()\n")
        (repo / "b.js").write_text("function g() {}\n")
        handle = RepoHandle(root=repo.resolve(), commit_sha=sha_for(job_id), url=url,
                            is_shallow=True, default_branch="main", job_id=job_id)
        r.handles[job_id] = handle
        r.events.append("clone")
        return handle

    def fake_parse(root, ctx=None, gaps=None, **kw):
        r.events.append("parse")
        return fake_records()

    def fake_scip(records, repo_root, ctx=None):
        r.events.append("scip_start")
        out = scip_ok()
        r.events.append("scip_end")
        return out

    def spy_resolve(records, ctx=None, repo_root=None, gaps=None, scip=None):
        r.events.append("graph_start")
        r.resolve_kwargs = dict(repo_root=repo_root, gaps=gaps, scip=scip)
        return real_resolve_calls(records, ctx=ctx, repo_root=repo_root, gaps=gaps, scip=scip)

    real_vector = pj.vector_branch

    def spy_vector(repo_root, artifacts_dir, tenant_id, gaps):
        r.events.append("vector_start")
        return real_vector(repo_root, artifacts_dir, tenant_id, gaps)

    def spy_later(*a, **kw):
        r.events.append("later_graph_stages")

    def spy_cleanup(workspaces_root, job_id, keep_artifacts=True):
        r.cleanup_calls.append((job_id, keep_artifacts))
        real_cleanup(workspaces_root, job_id, keep_artifacts=keep_artifacts)

    monkeypatch.setattr(pj, "clone_repo_to_workspace", fake_clone)
    monkeypatch.setattr(pj, "parse_repo", fake_parse)
    monkeypatch.setattr(pj, "run_scip", fake_scip)
    monkeypatch.setattr(pj, "resolve_calls", spy_resolve)
    monkeypatch.setattr(pj, "vector_branch", spy_vector)
    monkeypatch.setattr(pj, "run_later_graph_stages", spy_later)
    monkeypatch.setattr(pj, "cleanup_workspace", spy_cleanup)
    monkeypatch.setattr(pj, "_scip_sem", None)          # fresh semaphore per test
    monkeypatch.delenv(pj.SCIP_CONCURRENCY_ENV, raising=False)
    return r


def run(tmp_path, job_id=JOB, tenant="tenant-1", **kw):
    return pj.run_job(job_id, "https://github.com/o/r", tenant, tmp_path, **kw)


def art(tmp_path, job_id=JOB) -> Path:
    return tmp_path / job_id / "artifacts"


def read_status(tmp_path, job_id=JOB) -> dict:
    return json.loads((art(tmp_path, job_id) / "status.json").read_text(encoding="utf-8"))


def in_thread(fn):
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    return t, box


# ------------------------------------------------------------------ happy path ----

def test_happy_path(tmp_path, rec):
    res = run(tmp_path)
    assert (res.status, res.graph_status, res.vector_status) == ("ok", "ok", "ok")
    assert res.commit_sha == sha_for(JOB) and res.error == ""
    a = art(tmp_path)
    for name in ("status.json", "symbol_map.json", "vector_stub.json", "gaps.json"):
        assert (a / name).is_file(), name
    # checkout removed, artifacts kept
    assert not (tmp_path / JOB / "repo").exists()
    assert rec.cleanup_calls == [(JOB, True)]
    # the resolver got the job's ScipResult and the checkout root; SCIP was not re-run there
    assert isinstance(rec.resolve_kwargs["scip"], ScipResult)
    assert rec.resolve_kwargs["repo_root"] == rec.handles[JOB].root
    assert rec.events.count("scip_start") == 1
    assert "later_graph_stages" in rec.events
    smap = load_symbol_map(a / "symbol_map.json")
    assert smap["commit_sha"] == sha_for(JOB)
    assert smap["files"]["a.py"]["coverage"] == "scip"
    assert smap["files"]["a.py"]["file_sha"] != ""          # repo_root was passed
    assert read_gaps(a / "gaps.json") == res.gaps


def test_status_json_final_shape(tmp_path, rec):
    run(tmp_path)
    st = read_status(tmp_path)
    assert st["state"] == "done" and st["result"] == "ok" and st["error"] == ""
    assert st["tenant_id"] == "tenant-1" and st["mode"] == "full"
    assert list(st["stages"]) == list(pj.STAGES)
    for name, entry in st["stages"].items():
        assert entry["state"] == "done", name
        assert entry["started_at"] and entry["finished_at"], name
    assert st["started_at"] and st["finished_at"]


def test_fork_happens_after_run_scip_returned(tmp_path, rec):
    run(tmp_path)
    assert rec.index("scip_end") < rec.index("graph_start")
    assert rec.index("scip_end") < rec.index("vector_start")


def test_base_commit_is_accepted_and_ignored(tmp_path, rec):
    assert run(tmp_path, base_commit="abc123").status == "ok"


# ------------------------------------------------------------------ mode / ids ----

def test_incremental_mode_not_implemented_and_touches_nothing(tmp_path, rec):
    with pytest.raises(NotImplementedError, match="incremental"):
        run(tmp_path, mode="incremental", base_commit="abc")
    assert list(tmp_path.iterdir()) == []
    assert rec.events == []


def test_unknown_mode_is_value_error(tmp_path, rec):
    with pytest.raises(ValueError):
        run(tmp_path, mode="nonsense")


def test_invalid_job_id_raises_clone_error(tmp_path, rec):
    with pytest.raises(CloneError):
        run(tmp_path, job_id="../etc")
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ clone failure ----

def test_clone_failure(tmp_path, rec, monkeypatch):
    def bad_clone(url, job_id, workspaces_root, ctx=None):
        raise CloneError("git clone failed: repository not found")
    monkeypatch.setattr(pj, "clone_repo_to_workspace", bad_clone)
    res = run(tmp_path)
    assert res.status == "failed"
    assert res.error == "git clone failed: repository not found"
    assert res.graph_status == res.vector_status == "not_run"
    assert "parse" not in rec.events
    st = read_status(tmp_path)
    assert st["state"] == "failed" and st["result"] == "failed"
    assert st["stages"]["clone"]["state"] == "failed"
    assert "repository not found" in st["stages"]["clone"]["detail"]
    assert st["stages"]["parse"]["state"] == "pending"
    assert rec.cleanup_calls == [(JOB, True)]


# ------------------------------------------------------------------ SCIP failure ----

def test_scip_failing_entirely_is_not_fatal(tmp_path, rec, monkeypatch):
    monkeypatch.setattr(pj, "run_scip", lambda records, root, ctx=None: scip_all_failed())
    res = run(tmp_path)
    assert res.status == "ok"
    smap = load_symbol_map(art(tmp_path) / "symbol_map.json")
    assert {f["coverage"] for f in smap["files"].values()} == {"treesitter_only"}
    assert all(s["boundary_source"] == "treesitter"
               for f in smap["files"].values() for s in f["symbols"])
    assert "scip_language_failed" in {g.kind for g in res.gaps}      # resolver reports it


def test_run_scip_raising_gives_empty_result_and_language_gaps(tmp_path, rec, monkeypatch):
    def boom(records, root, ctx=None):
        raise RuntimeError("indexer exploded")
    monkeypatch.setattr(pj, "run_scip", boom)
    res = run(tmp_path)
    assert res.status == "ok"
    # an empty ScipResult, never None (None would make the resolver run SCIP again)
    assert isinstance(rec.resolve_kwargs["scip"], ScipResult)
    assert rec.resolve_kwargs["scip"].status["covered_files"] == set()
    langs = {g.detail.split(":")[0] for g in res.gaps
             if g.stage == "scip" and g.kind == "scip_language_failed" and "RuntimeError" in g.detail}
    assert langs == {"python", "javascript"}
    assert all(g.scope == "language" and g.path == ""
               for g in res.gaps if "RuntimeError" in g.detail)
    smap = load_symbol_map(art(tmp_path) / "symbol_map.json")
    assert {f["coverage"] for f in smap["files"].values()} == {"treesitter_only"}


# ------------------------------------------------------------------ branch isolation ----

def test_graph_failure_vector_output_intact(tmp_path, rec, monkeypatch):
    def boom(records, ctx=None, repo_root=None, gaps=None, scip=None):
        raise ValueError("resolver blew up")
    monkeypatch.setattr(pj, "resolve_calls", boom)
    res = run(tmp_path)
    assert (res.status, res.graph_status, res.vector_status) == ("partial", "failed", "ok")
    stub = json.loads((art(tmp_path) / "vector_stub.json").read_text())
    assert stub["total_files"] == 2 and stub["commit_sha"] == sha_for(JOB)
    gaps = read_gaps(art(tmp_path) / "gaps.json")
    failed = [g for g in gaps if g.kind == "branch_failed"]
    assert len(failed) == 1
    assert failed[0].stage == "graph" and failed[0].scope == "repo"
    assert "ValueError: resolver blew up" in failed[0].detail and failed[0].impact
    st = read_status(tmp_path)
    assert st["state"] == "done" and st["result"] == "partial"
    assert st["stages"]["graph"]["state"] == "failed"
    assert st["stages"]["vector"]["state"] == "done"


def test_vector_failure_graph_output_intact(tmp_path, rec, monkeypatch):
    def boom(repo_root, artifacts_dir, tenant_id, gaps):
        raise OSError("vector store down")
    monkeypatch.setattr(pj, "vector_branch", boom)
    res = run(tmp_path)
    assert (res.status, res.graph_status, res.vector_status) == ("partial", "ok", "failed")
    assert "later_graph_stages" in rec.events
    failed = [g for g in res.gaps if g.kind == "branch_failed"]
    assert [g.stage for g in failed] == ["vector"]
    assert "OSError: vector store down" in failed[0].detail
    assert not (art(tmp_path) / "vector_stub.json").exists()
    assert read_status(tmp_path)["stages"]["graph"]["state"] == "done"


def test_both_branches_failing_is_failed(tmp_path, rec, monkeypatch):
    def g_boom(*a, **kw):
        raise ValueError("g")

    def v_boom(*a, **kw):
        raise ValueError("v")
    monkeypatch.setattr(pj, "resolve_calls", g_boom)
    monkeypatch.setattr(pj, "vector_branch", v_boom)
    res = run(tmp_path)
    assert (res.status, res.graph_status, res.vector_status) == ("failed", "failed", "failed")
    assert sorted(g.stage for g in res.gaps if g.kind == "branch_failed") == ["graph", "vector"]
    st = read_status(tmp_path)
    assert st["state"] == "failed" and st["result"] == "failed"
    assert (art(tmp_path) / "gaps.json").is_file()


def test_branches_use_their_own_gap_lists(tmp_path, rec, monkeypatch):
    seen = {}
    real_vector = pj.vector_branch

    def spy_resolve(records, ctx=None, repo_root=None, gaps=None, scip=None):
        seen["graph"] = gaps
        return []

    def spy_vector(repo_root, artifacts_dir, tenant_id, gaps):
        seen["vector"] = gaps
        return real_vector(repo_root, artifacts_dir, tenant_id, gaps)
    monkeypatch.setattr(pj, "resolve_calls", spy_resolve)
    monkeypatch.setattr(pj, "vector_branch", spy_vector)
    run(tmp_path)
    assert seen["graph"] is not seen["vector"]


def test_symbol_map_write_failure_skips_vector_but_graph_runs(tmp_path, rec, monkeypatch):
    def disk_full(symbol_map, artifacts_dir):
        raise OSError("disk full")
    monkeypatch.setattr(pj, "write_symbol_map", disk_full)
    res = run(tmp_path)
    assert (res.status, res.graph_status, res.vector_status) == ("partial", "ok", "failed")
    assert "vector_start" not in rec.events
    failed = [g for g in res.gaps if g.kind == "branch_failed"]
    assert len(failed) == 1 and failed[0].stage == "vector"
    assert "disk full" in failed[0].detail
    st = read_status(tmp_path)
    assert st["stages"]["symbol_map"]["state"] == "failed"
    assert st["stages"]["vector"]["state"] == "failed"


# ------------------------------------------------------------------ status polling ----

def test_status_json_parseable_while_running(tmp_path, rec, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocking_resolve(records, ctx=None, repo_root=None, gaps=None, scip=None):
        started.set()
        assert release.wait(10)
        return []
    monkeypatch.setattr(pj, "resolve_calls", blocking_resolve)

    errors, stop = [], threading.Event()
    status_path = art(tmp_path) / "status.json"

    def hammer():                       # reads continuously while the job keeps writing
        while not stop.is_set():
            try:
                json.loads(status_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass                    # before the first write
            except Exception as exc:    # noqa: BLE001 - a partial file would land here
                errors.append(repr(exc))

    reader = threading.Thread(target=hammer)
    reader.start()
    t, box = in_thread(lambda: run(tmp_path))
    try:
        assert started.wait(10)
        st = read_status(tmp_path)
        assert st["state"] == "running" and st["result"] is None
        assert st["stages"]["scip"]["state"] == "done"
        assert st["stages"]["graph"]["state"] == "running"
        assert st["finished_at"] is None
    finally:
        release.set()
        t.join(10)
        stop.set()
        reader.join(10)
    assert box["result"].status == "ok"
    assert errors == []
    assert read_status(tmp_path)["state"] == "done"


# ------------------------------------------------------------------ cleanup ----

def test_cleanup_runs_when_a_stage_raises(tmp_path, rec, monkeypatch):
    def bad_parse(root, ctx=None, gaps=None, **kw):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(pj, "parse_repo", bad_parse)
    res = run(tmp_path)
    assert res.status == "failed" and "RuntimeError: kaboom" in res.error
    assert rec.cleanup_calls == [(JOB, True)]
    assert not (tmp_path / JOB / "repo").exists()
    assert art(tmp_path).is_dir()                       # artifacts survive
    st = read_status(tmp_path)
    assert st["state"] == "failed" and st["stages"]["parse"]["state"] == "failed"
    assert "kaboom" in st["error"]


def test_cleanup_runs_and_status_is_marked_on_keyboard_interrupt(tmp_path, rec, monkeypatch):
    def interrupted(root, ctx=None, gaps=None, **kw):
        raise KeyboardInterrupt()
    monkeypatch.setattr(pj, "parse_repo", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    assert rec.cleanup_calls == [(JOB, True)]
    assert not (tmp_path / JOB / "repo").exists()
    assert read_status(tmp_path)["state"] == "failed"
    # the job id is released, so a retry is not rejected as "already running"
    monkeypatch.setattr(pj, "parse_repo", lambda root, ctx=None, gaps=None, **kw: fake_records())
    assert run(tmp_path).status == "ok"


def test_cleanup_failure_does_not_mask_the_result(tmp_path, rec, monkeypatch):
    def broken_cleanup(workspaces_root, job_id, keep_artifacts=True):
        raise CloneError("cleanup exploded")
    monkeypatch.setattr(pj, "cleanup_workspace", broken_cleanup)
    assert run(tmp_path).status == "ok"


# ------------------------------------------------------------------ concurrency ----

def test_two_concurrent_jobs_use_separate_workspaces(tmp_path, rec, monkeypatch):
    barrier = threading.Barrier(2, timeout=10)      # both jobs are inside parse at once

    def parse(root, ctx=None, gaps=None, **kw):
        barrier.wait()
        return fake_records()
    monkeypatch.setattr(pj, "parse_repo", parse)

    t1, b1 = in_thread(lambda: run(tmp_path, job_id="job-aaaaaaaa", tenant="tenant-A"))
    t2, b2 = in_thread(lambda: run(tmp_path, job_id="job-bbbbbbbb", tenant="tenant-B"))
    t1.join(20)
    t2.join(20)
    assert "error" not in b1 and "error" not in b2
    assert b1["result"].status == b2["result"].status == "ok"
    for job_id, tenant in (("job-aaaaaaaa", "tenant-A"), ("job-bbbbbbbb", "tenant-B")):
        stub = json.loads((art(tmp_path, job_id) / "vector_stub.json").read_text())
        assert stub["commit_sha"] == sha_for(job_id) and stub["tenant_id"] == tenant
        st = read_status(tmp_path, job_id)
        assert st["job_id"] == job_id and st["tenant_id"] == tenant and st["result"] == "ok"
        assert load_symbol_map(art(tmp_path, job_id) / "symbol_map.json")["commit_sha"] == sha_for(job_id)
        assert not (tmp_path / job_id / "repo").exists()
    assert rec.handles["job-aaaaaaaa"].root != rec.handles["job-bbbbbbbb"].root
    assert sorted(rec.cleanup_calls) == [("job-aaaaaaaa", True), ("job-bbbbbbbb", True)]


def test_same_job_id_twice_is_rejected_without_touching_the_first(tmp_path, rec, monkeypatch):
    in_parse, release = threading.Event(), threading.Event()

    def parse(root, ctx=None, gaps=None, **kw):
        in_parse.set()
        assert release.wait(10)
        return fake_records()
    monkeypatch.setattr(pj, "parse_repo", parse)

    t, box = in_thread(lambda: run(tmp_path))
    try:
        assert in_parse.wait(10)
        second = run(tmp_path)
        assert second.status == "failed" and "already running" in second.error
        # the first job's workspace and status are untouched
        assert (tmp_path / JOB / "repo").is_dir()
        assert read_status(tmp_path)["state"] == "running"
        assert rec.cleanup_calls == []
    finally:
        release.set()
        t.join(10)
    assert box["result"].status == "ok"
    assert rec.cleanup_calls == [(JOB, True)]


def test_existing_checkout_is_rejected_and_left_alone(tmp_path, rec):
    repo = tmp_path / JOB / "repo"
    repo.mkdir(parents=True)
    (repo / "keep.txt").write_text("precious")
    res = run(tmp_path)
    assert res.status == "failed" and "already contains" in res.error
    assert (repo / "keep.txt").read_text() == "precious"
    assert not (tmp_path / JOB / "artifacts").exists()
    assert rec.cleanup_calls == [] and rec.events == []


# ------------------------------------------------------------------ SCIP semaphore ----

@pytest.mark.parametrize("raw,expected", [
    (None, 2), ("", 2), ("1", 1), ("5", 5), ("0", 2), ("-3", 2), ("many", 2), (" 4 ", 4)])
def test_scip_limit_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(pj.SCIP_CONCURRENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(pj.SCIP_CONCURRENCY_ENV, raw)
    assert pj._scip_limit() == expected


def _concurrent_scip_probe(monkeypatch, jobs: int, hold: float):
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def slow_scip(records, root, ctx=None):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(hold)
        with lock:
            state["now"] -= 1
        return scip_ok()
    monkeypatch.setattr(pj, "run_scip", slow_scip)
    return state


def test_scip_concurrency_limited_to_env_value(tmp_path, rec, monkeypatch):
    monkeypatch.setenv(pj.SCIP_CONCURRENCY_ENV, "1")
    state = _concurrent_scip_probe(monkeypatch, 3, 0.05)
    threads = [in_thread(lambda i=i: run(tmp_path, job_id=f"job-0000000{i}")) for i in range(3)]
    for t, _ in threads:
        t.join(20)
    assert all(b["result"].status == "ok" for _, b in threads)
    assert state["max"] == 1


def test_scip_concurrency_default_is_two(tmp_path, rec, monkeypatch):
    state = _concurrent_scip_probe(monkeypatch, 4, 0.05)
    threads = [in_thread(lambda i=i: run(tmp_path, job_id=f"job-0000000{i}")) for i in range(4)]
    for t, _ in threads:
        t.join(20)
    assert all(b["result"].status == "ok" for _, b in threads)
    assert 1 <= state["max"] <= 2


# ------------------------------------------------------------------ branch units ----

def test_vector_branch_signature_takes_no_records_or_scip():
    params = inspect.signature(pj.vector_branch).parameters
    assert list(params) == ["repo_root", "artifacts_dir", "tenant_id", "gaps"]
    assert "records" not in params and "scip" not in params
    assert not any("Scip" in str(p.annotation) or "FileRecord" in str(p.annotation)
                   for p in params.values())


def test_graph_branch_signature():
    assert list(inspect.signature(pj.graph_branch).parameters) == [
        "records", "scip", "handle", "gaps", "ctx"]


def test_vector_branch_counts_per_coverage(tmp_path, monkeypatch):
    from analyzer_new.symbol_map import build_symbol_map, write_symbol_map
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.js").write_text("function g() {}\n")
    scip = ScipResult(status={"indexed": ["python"], "skipped": {"javascript": "x"},
                              "covered_files": {"a.py"}, "scip_print_available": True})
    smap = build_symbol_map(fake_records(), scip, "c" * 40, gaps=[], repo_root=root)
    artifacts = tmp_path / "artifacts"
    write_symbol_map(smap, artifacts)

    out = pj.vector_branch(root, artifacts, "tenant-9", [])
    assert out["files_by_coverage"] == {"scip": 1, "treesitter_only": 1, "fallback_lines": 0,
                                        "skipped": 0, "unparsed": 0}
    assert out["symbols_by_coverage"]["scip"] == 1
    assert out["symbols_by_coverage"]["treesitter_only"] == 1
    assert out["total_files"] == 2 and out["total_symbols"] == 2
    on_disk = json.loads((artifacts / "vector_stub.json").read_text())
    assert on_disk == out and on_disk["tenant_id"] == "tenant-9"
    assert on_disk["commit_sha"] == "c" * 40


def test_vector_branch_without_symbol_map_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pj.vector_branch(tmp_path, tmp_path / "artifacts", "t", [])


def test_safe_returns_ok_or_appends_branch_failed():
    gaps = []
    assert pj._safe("graph", lambda: None, gaps, "impact") == "ok" and gaps == []

    def boom():
        raise KeyError("x")
    assert pj._safe("vector", boom, gaps, "lost search") == "failed"
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.stage, g.scope, g.kind, g.impact) == ("vector", "repo", "branch_failed", "lost search")
    assert "KeyError" in g.detail


def test_safe_does_not_swallow_keyboard_interrupt():
    def stop():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        pj._safe("graph", stop, [], "")


@pytest.mark.parametrize("g,v,expected", [
    ("ok", "ok", "ok"), ("failed", "ok", "partial"), ("ok", "failed", "partial"),
    ("failed", "failed", "failed")])
def test_combine(g, v, expected):
    assert pj._combine(g, v) == expected