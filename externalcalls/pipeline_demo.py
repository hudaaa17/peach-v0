"""Run the real pipeline stages with reports; never replace pipeline_job.py.

Example (container):
    python -m externalcalls.pipeline_demo --source /repos/example --out /reports/demo

The output directory must be outside a local source folder. Cloned workspaces
are retained for diagnosis. Only SCIP may temporarily write into the checkout.
Imports are lazy so an unavailable stage does not prevent later diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass, replace
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid

STAGES = (
    "clone", "parse_repo", "scip", "resolve_calls", "ast_grep",
    "literal", "constprop", "config", "joern_tasks", "slices",
    "joern_run", "joern_apply", "gaps",
)
JOERN_HOME = Path("/opt/joern/joern-cli")
HOST_STAGES = {6: "literal", 7: "constprop", 8: "config", 12: "joern"}


def _json_safe(value):
    if hasattr(value, "to_json_safe"):
        return _json_safe(value.to_json_safe())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _json(value):
    return json.dumps(_json_safe(value), ensure_ascii=True, default=str)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, indent=2, ensure_ascii=True, default=str)
        stream.write("\n")


def _brief(value):
    text = _json(value)
    if len(text) <= 1800:
        return text
    if isinstance(value, dict):
        sample = dict(list(value.items())[:5])
    elif isinstance(value, (list, tuple)):
        sample = value[:5]
    else:
        return text[:1800] + " ... (full value in stage JSON)"
    return f"{len(value)} entries; sample={_json(sample)} (full value in stage JSON)"


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile
        self.encoding = "utf-8"

    def write(self, text):
        # Some Windows terminals cannot display all source characters.
        encoding = getattr(self.console, "encoding", None) or "utf-8"
        visible = text.encode(encoding, errors="backslashreplace").decode(encoding)
        self.console.write(visible)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return False


@contextmanager
def _logging_to_both(out):
    root_logger = logging.getLogger()
    previous_handlers, previous_level = root_logger.handlers[:], root_logger.level
    with (out / "pipeline.log").open("w", encoding="utf-8", newline="\n") as logfile:
        tee = _Tee(sys.stdout, logfile)
        handler = logging.StreamHandler(tee)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root_logger.handlers = [handler]
        root_logger.setLevel(logging.INFO)
        try:
            with redirect_stdout(tee), redirect_stderr(tee):
                yield
        finally:
            root_logger.handlers = previous_handlers
            root_logger.setLevel(previous_level)
            handler.close()


class _Skipped(Exception):
    pass


class _Demo:
    def __init__(self, args, out, local_root):
        from analyzer_new.run_stages import Ctx

        self.args, self.out, self.local_root = args, out, local_root
        self.environment = "Windows host" if sys.platform == "win32" else "container"
        self.ctx = Ctx()
        self.gaps = self.ctx.gaps
        self.root = self.records = self.scip = self.edges = None
        self.pattern_report = self.join_report = self.literal_report = None
        self.constprop_report = self.config_report = self.task_report = None
        self.tasks = self.task_slices = self.groups = self.results = None
        self.apply_report = None
        self.task_runs, self.task_directories = {}, {}
        self.rows, self.probes = [], []
        self.temporary = None
        self.current = self.data = None

    def _counts(self):
        return {
            "root": int(self.root is not None),
            "records": None if self.records is None else len(self.records),
            "calls": None if self.records is None else sum(len(r.calls) for r in self.records),
            "edges": None if self.edges is None else len(self.edges),
            "tasks": None if self.tasks is None else len(self.tasks),
            "slices": None if self.task_slices is None else len(self.task_slices),
            "groups": None if self.groups is None else len(self.groups),
            "results": None if self.results is None else len(self.results),
            "gaps": len(self.gaps),
        }

    def _edge_snapshot(self):
        edges = self.edges or []
        return {"total": len(edges), "by_status": dict(Counter(e.status for e in edges)),
                "with_host": sum(bool(e.host) for e in edges),
                "hosts": [e.host for e in edges]}

    def _require(self, *requirements):
        for attribute, stage in requirements:
            value = getattr(self, attribute)
            if value is None or (isinstance(value, dict) and
                                 (value.get("aborted") or value.get("problem"))):
                raise _Skipped(f"missing input from stage {stage} ({attribute})")

    def _check_report(self, report, name):
        if (report.get("aborted") or report.get("problem") or any(
                reason.get("kind") == "stage_failed"
                or reason.get("cause") == "stage_failed"
                for reason in report.get("reasons", []) if isinstance(reason, dict))
                or (name == "run_config_check" and report.get("reasons"))):
            raise RuntimeError(f"{name} reported failure: {_json(report)}")

    def _exception(self, context):
        detail = traceback.format_exc()
        self.current["status"] = "FAILED"
        self.current["errors"].append({"context": context, "traceback": detail})
        print(f"{context}:\n{detail}", end="" if detail.endswith("\n") else "\n")

    def _stage(self, number, action):
        from analyzer_new.gaps import gaps_to_dicts, summarize_gaps

        name = STAGES[number - 1]
        print(f"\n=== STAGE {number}/13 {name} ===")
        print(f"environment: {self.environment}")
        before = self._edge_snapshot()
        counters = self.ctx.counters.copy()
        gap_start, note_start = len(self.gaps), len(self.ctx.notes)
        row = {"stage": number, "name": name, "environment": self.environment,
               "status": "OK", "reason": None, "inputs": self._counts(), "errors": []}
        self.current, self.data = row, {"reports": {}, "samples": []}
        started = time.perf_counter()
        try:
            action()
        except _Skipped as error:
            if row["status"] != "FAILED":
                row["status"] = "SKIPPED"
            row["reason"] = str(error)
        except Exception:
            self._exception(f"stage {number} {name}")
        if number == 11:
            for task in self.tasks or []:
                print(self._task_line(task))
        row["seconds"] = time.perf_counter() - started
        after = self._edge_snapshot()
        changed = {key: {"before": counters[key], "after": self.ctx.counters[key],
                         "delta": self.ctx.counters[key] - counters[key]}
                   for key in counters.keys() | self.ctx.counters.keys()
                   if counters[key] != self.ctx.counters[key]}
        added = self.gaps[gap_start:]
        row.update(outputs=self._counts(), edges_before=before, edges_after=after,
                   counters_changed=changed, gaps_added=gaps_to_dicts(added),
                   gaps_added_by_kind=summarize_gaps(added),
                   gap_entries_added_by_kind=dict(Counter(g.kind for g in added)),
                   notes=self.ctx.notes[note_start:], data=self.data)
        print(f"STATUS: {row['status']}" + (f": {row['reason']}" if row["reason"] else ""))
        print(f"seconds: {row['seconds']:.6f}")
        print(f"inputs (counts): {_json(row['inputs'])}")
        print(f"outputs (counts): {_json(row['outputs'])}")
        print(f"edges by status: {_json(before['by_status'])} -> {_json(after['by_status'])}")
        print(f"edges with a host: {before['with_host']} -> {after['with_host']}")
        print(f"gaps added: entries={len(added)}; by kind={_json(row['gaps_added_by_kind'])}")
        print(f"counters changed: {_json(changed)}")
        print(f"samples (up to 5): {_json(self.data['samples'][:5])}")
        for report_name, report in self.data["reports"].items():
            print(f"report {report_name} (real keys):")
            for key, value in report.items():
                print(f"  {key}: {_brief(value)}")
        if not self.data["reports"]:
            print("report: no report dict returned; outputs are saved in stage JSON")
        self.rows.append(row)
        # Freeze each stage while its reports still describe this point in time.
        payload = dict(row, edges=self.edges or [])
        try:
            _write_json(self.out / f"stage{number:02d}_{name}.json", payload)
        except Exception:
            self._exception(f"writing stage {number} JSON")
            print("STATUS: FAILED: could not save the stage JSON")

    def clone(self):
        if self.local_root is not None:
            self.root = self.local_root
            self.data.update(root=self.root, source_kind="local", samples=[str(self.root)])
            print("WARNING: SCIP may touch the local folder temporarily.")
            raise _Skipped("local folder; clone skipped, source is used in place")
        from analyzer_new.clone import clone_repo_to_workspace

        handle = clone_repo_to_workspace(
            self.args.source, f"demo_{uuid.uuid4().hex}", self.out / "workspaces", ctx=self.ctx)
        self.root = handle.root
        self.data.update(root=self.root, repo_handle=json.loads(handle.to_json()),
                         samples=[str(self.root)])
        print(f"checkout retained on disk: {self.root}")

    def parse(self):
        self._require(("root", 1))
        from analyzer_new.parser import parse_repo

        self.records = parse_repo(self.root, ctx=self.ctx, max_files=self.args.max_files,
                                  gaps=self.gaps)
        self.data.update(records=self.records, max_files=self.args.max_files)
        self.data["samples"] = [dict(path=r.path, language=r.language, skipped=r.skipped,
                                     parse_mode=r.parse_mode, calls=len(r.calls))
                                for r in self.records[:5]]

    def index(self):
        self._require(("records", 2), ("root", 1))
        from analyzer_new.scip_check import run_scip

        self.scip = run_scip(self.records, self.root, ctx=self.ctx)
        self.data["scip"] = self.scip
        self.data["reports"]["scip.status"] = self.scip.status
        self.data["samples"] = list(self.scip.to_json_safe()["ref_map"].items())[:5]
        if self.scip.status["skipped"]:
            print(f"SCIP languages not indexed: {_json(self.scip.status['skipped'])}")
            if not self.scip.status["indexed"]:
                raise _Skipped("SCIP indexing unavailable; real ScipResult retained for resolver fallback")

    def resolve(self):
        self._require(("records", 2), ("scip", 3), ("root", 1))
        from analyzer_new.resolver import resolve_calls

        self.edges = resolve_calls(self.records, ctx=self.ctx, repo_root=self.root,
                                   gaps=self.gaps, scip=self.scip)
        self.data["samples"] = self.edges[:5]

    def patterns(self):
        self._require(("edges", 4), ("records", 2), ("root", 1))
        from externalcalls.ast_grep_check import run_patterns
        from externalcalls.ast_grep_join import join_ast_grep_hits

        files_by_language = defaultdict(list)
        for record in self.records:
            if (not record.skipped and not record.is_config
                    and record.language in {"python", "javascript", "typescript"}):
                files_by_language[record.language].append(record.path)
        hits, self.pattern_report = run_patterns(self.root, dict(files_by_language), ctx=self.ctx)
        self.data.update(files_by_language=dict(files_by_language), hits=hits, samples=hits[:5])
        self.data["reports"]["run_patterns"] = self.pattern_report
        # Even a missing ast-grep binary supplies the real degradation report
        # that the real join needs for its own regex fallback.
        self.join_report = join_ast_grep_hits(
            self.edges, self.records, hits, self.pattern_report, ctx=self.ctx)
        self.data["reports"]["join_ast_grep_hits"] = self.join_report
        if self.pattern_report["binary"] is None:
            print("run_patterns: pattern scan unavailable; join's real fallback was attempted")
        self._check_report(self.pattern_report, "run_patterns")
        self._check_report(self.join_report, "join_ast_grep_hits")

    def literal(self):
        self._require(("edges", 4), ("records", 2), ("root", 1), ("join_report", 5))
        from externalcalls.literal_check import run_literal_check

        self.literal_report = run_literal_check(self.edges, self.records, self.root, ctx=self.ctx)
        self.data["reports"]["run_literal_check"] = self.literal_report
        self.data["samples"] = list(self.literal_report["resolved_edges"].items())[:5]
        self._check_report(self.literal_report, "run_literal_check")

    def constprop(self):
        self._require(("edges", 4), ("records", 2), ("root", 1), ("literal_report", 6))
        from externalcalls.constprop_check import run_constprop_check

        self.constprop_report = run_constprop_check(
            self.edges, self.records, self.root, self.literal_report, ctx=self.ctx)
        self.data["reports"]["run_constprop_check"] = self.constprop_report
        self.data["samples"] = list(self.constprop_report["still_pending"].items())[:5]
        self._check_report(self.constprop_report, "run_constprop_check")

    def config(self):
        self._require(("edges", 4), ("records", 2), ("literal_report", 6), ("constprop_report", 7))
        from externalcalls.config_check import run_config_check

        self.config_report = run_config_check(
            self.edges, self.records, self.literal_report, self.constprop_report, ctx=self.ctx)
        self.data["reports"]["run_config_check"] = self.config_report
        self.data["samples"] = list(self.config_report["resolved_edges"].items())[:5]
        self._check_report(self.config_report, "run_config_check")

    def make_tasks(self):
        self._require(("edges", 4), ("records", 2), ("literal_report", 6), ("constprop_report", 7))
        from externalcalls.joern_tasks import build_joern_tasks

        self.task_report = build_joern_tasks(
            self.edges, self.records, self.literal_report, self.constprop_report,
            config_report=self.config_report)
        self.data["reports"]["build_joern_tasks"] = self.task_report
        self._check_report(self.task_report, "build_joern_tasks")
        self.tasks = self.task_report["tasks"]
        self.data["samples"] = self.tasks[:5]

    def slices(self):
        self._require(("tasks", 9), ("records", 2), ("edges", 4), ("root", 1))
        from externalcalls.slice_builder import build_slice, group_tasks

        records_by_path = {record.path: record for record in self.records}
        self.task_slices = {}
        for task in self.tasks:
            try:
                source_slice = build_slice(task, records_by_path, self.edges, self.root)
                self.task_slices[task.task_id] = source_slice
                print(f"task {task.task_id}: slice files={_json(source_slice.files)} "
                      f"total lines={source_slice.total_lines} incomplete={source_slice.incomplete} "
                      f"omitted={_json(source_slice.omitted)}")
            except Exception:
                self._exception(f"build_slice task {task.task_id}")
        self.data["task_slices"] = self.task_slices
        self.groups = group_tasks([(task, self.task_slices[task.task_id]) for task in self.tasks
                                   if task.task_id in self.task_slices])
        self.data.update(groups=self.groups, samples=list(self.task_slices.items())[:5])

    def _task_line(self, task):
        run = self.task_runs.get(task.task_id, {})
        return (f"task_id={task.task_id} {task.file}:{task.line} reason={task.reason} "
                f"url_arg_keyword={task.url_arg_keyword!r} url_arg_index={task.url_arg_index!r} "
                f"status={run.get('status', 'not run (missing input from stage 10)')} "
                f"sources={run.get('source_count', 0)} group_seconds={run.get('seconds', 0):.6f} "
                f"runner_reason={_json(run.get('reason', 'slice missing from stage 10'))}")

    def run_joern(self):
        self._require(("tasks", 9), ("groups", 10), ("root", 1))
        from externalcalls.joern_runner import run_joern_tasks
        from externalcalls.slice_builder import materialize_slice

        self.results = []
        self.data.update(results=self.results, groups=[], task_runs=self.task_runs)
        tasks_by_id = {task.task_id: task for task in self.tasks}
        installed = (JOERN_HOME / "joern").is_file() and (JOERN_HOME / "joern-parse").is_file()
        for group_index, (language, files, task_ids) in enumerate(self.groups):
            group = dict(index=group_index, language=language, files=files, task_ids=task_ids,
                         seconds=0.0)
            self.data["groups"].append(group)
            if group_index >= self.args.max_joern_groups:
                status, reason = "not run (cap)", "--max-joern-groups cap"
            elif not installed:
                status, reason = "not run (Joern missing)", f"Joern install missing: {JOERN_HOME}"
            else:
                status, reason = "no result", "runner returned no result for task"
                started = time.perf_counter()
                try:
                    if self.temporary is None:
                        self.temporary = tempfile.TemporaryDirectory(
                            prefix="pipeline-demo-slices-", dir=str(self.out))
                    destination = Path(self.temporary.name) / f"group-{group_index:03d}"
                    if destination.resolve() == self.root or self.root in destination.resolve().parents:
                        raise ValueError("slice destination is inside the checkout")
                    materialized, error = materialize_slice(files, self.root, destination)
                    group.update(slice_directory=materialized, materialize_error=error)
                    if error:
                        raise RuntimeError(f"materialize_slice: {error}")
                    for task_id in task_ids:
                        self.task_directories[task_id] = materialized
                    group_results = run_joern_tasks(
                        [tasks_by_id[task_id] for task_id in task_ids], materialized,
                        timeout_seconds=self.args.joern_timeout, joern_home=JOERN_HOME,
                        task_slices={task_id: self.task_slices[task_id] for task_id in task_ids})
                    self.results.extend(group_results)
                    group["results"] = group_results
                except Exception:
                    self._exception(f"Joern group {group_index}")
                    status, reason = "not run (group failed)", self.current["errors"][-1]["traceback"]
                group["seconds"] = time.perf_counter() - started
            group.update(status=status, reason=reason)
            by_id = {result["task_id"]: result for result in group.get("results", [])}
            if by_id:
                group["status"] = "returned results"
                group["reason"] = None
            print(f"group {group_index}: {group['status']}; task_ids={_json(task_ids)}")
            for task_id in task_ids:
                result = by_id.get(task_id)
                self.task_runs[task_id] = {
                    "group": group_index, "seconds": group["seconds"],
                    "status": result["status"] if result is not None else status,
                    "source_count": len(result.get("sources", [])) if result is not None else 0,
                    "reason": result.get("reason") if result is not None else reason,
                }
        self.data["samples"] = self.results[:5]
        if not self.groups:
            raise _Skipped("no Joern groups to run")
        if self.args.max_joern_groups == 0:
            raise _Skipped("--max-joern-groups 0; Joern skipped")
        if not installed:
            raise _Skipped(f"Joern install missing: {JOERN_HOME}")

    def probe(self):
        print(f"\n=== KEYWORD PROBE ({self.environment}; diagnostic only) ===")
        print("task_id | original status | original sources | probe status | probe sources")
        tasks_by_id = {task.task_id: task for task in self.tasks or []}
        for original in self.results or []:
            task = tasks_by_id.get(original["task_id"])
            if (task is None or original["status"] != "no_sources_found"
                    or task.url_arg_keyword is None):
                continue
            row = {"task_id": task.task_id, "original_status": original["status"],
                   "original_source_count": len(original.get("sources", [])),
                   "original_result": original, "probe_status": "failed", "probe_source_count": 0}
            started = time.perf_counter()
            try:
                from externalcalls.joern_runner import run_joern_tasks

                probe_task = replace(task, url_arg_keyword=None)
                probe_results = run_joern_tasks(
                    [probe_task], self.task_directories[task.task_id],
                    timeout_seconds=self.args.joern_timeout, joern_home=JOERN_HOME,
                    task_slices={task.task_id: self.task_slices[task.task_id]})
                result = next(item for item in probe_results if item["task_id"] == task.task_id)
                row.update(probe_status=result["status"], probe_source_count=len(result.get("sources", [])),
                           probe_result=result, slice_directory=self.task_directories[task.task_id])
            except Exception:
                row["traceback"] = traceback.format_exc()
                print(row["traceback"])
            row["seconds"] = time.perf_counter() - started
            self.probes.append(row)
            print(f"{task.task_id} | {row['original_status']} | {row['original_source_count']} | "
                  f"{row['probe_status']} | {row['probe_source_count']}")
        if not self.probes:
            print("No eligible keyword tasks; probe was not run.")
        _write_json(self.out / "probe.json", self.probes)

    def _joern_reason(self, index, edge, candidates):
        tasks = [task for task in self.tasks or [] if task.edge_index == index]
        if not tasks:
            return str((self.task_report or {}).get("skipped", {}).get(index, "no Joern task"))
        task = tasks[0]
        run = self.task_runs.get(task.task_id)
        if run is None:
            return "missing input from stage 10 or stage 11"
        reason = f"{run['status']}: {run.get('reason') or ''}"
        if run["status"] == "complete" and not edge.host:
            hosts = sorted({item["host"] for item in candidates if item.get("host")})
            reason += f"; verified candidate hosts={_json(hosts)}"
            if not candidates:
                reason += "; no verified candidates retained"
            reason += "; J4 did not set a host (see stage12 gaps and report)"
        return reason

    def apply(self):
        self._require(("edges", 4), ("tasks", 9), ("root", 1))
        from externalcalls.joern_apply import apply_joern_results

        self.data["edge_decisions"] = decisions = []
        if self.results:
            result_ids = {result["task_id"] for result in self.results}
            # J4 invents a failed result for a supplied task with no result.
            # Supply only tasks that actually returned a result; keep the full
            # edge list in its original order and use the one shared gap list.
            applied_tasks = [task for task in self.tasks if task.task_id in result_ids]
            self.apply_report = apply_joern_results(
                self.edges, applied_tasks, self.results, self.root,
                task_slices=self.task_slices, gaps=self.gaps)
            self.data.update(applied_task_ids=[task.task_id for task in applied_tasks],
                             not_applied_task_ids=[task.task_id for task in self.tasks
                                                   if task.task_id not in result_ids])
            self.data["reports"]["apply_joern_results"] = self.apply_report
            print(f"J4 counters: {_json(self.apply_report['counters'])}")
        else:
            print("J4 counters: unavailable (no runner results to apply)")
        for index, edge in enumerate(self.edges):
            candidates = (self.apply_report or {}).get("joern_candidates", {}).get(index, [])
            reason = None if edge.host else self._joern_reason(index, edge, candidates)
            decision = dict(edge_index=index, file=edge.file, line=edge.line,
                            candidates=candidates, host=edge.host, reason=reason)
            decisions.append(decision)
            print(f"edge {index} {edge.file}:{edge.line}: candidates={_json(candidates)} "
                  + (f"host={edge.host}" if edge.host else f"no host: {reason}"))
        self.data["samples"] = decisions[:5]
        if not self.results:
            raise _Skipped("missing input from stage 11 (no runner results to apply)")

    def finish_gaps(self):
        from analyzer_new.gaps import gaps_to_dicts, summarize_gaps
        from externalcalls.external_gaps import build_external_call_gaps

        external = build_external_call_gaps(
            self.pattern_report, self.join_report, self.literal_report,
            self.constprop_report, self.config_report)
        self.gaps.extend(external)
        self.data.update(external_gaps=gaps_to_dicts(external), gaps=gaps_to_dicts(self.gaps),
                         samples=gaps_to_dicts(external[:5]))
        self.data["reports"]["summarize_gaps"] = summarize_gaps(self.gaps)
        _write_json(self.out / "gaps.json", gaps_to_dicts(self.gaps))

    def _stage_reason(self, number):
        row = self.rows[number - 1]
        if row["status"] == "OK":
            return ""
        error = row["errors"][-1]["traceback"].strip().splitlines()[-1] if row["errors"] else ""
        return f"stage {number} {row['status']}: {row['reason'] or error}"

    def _literal_reason(self, index, edge):
        report = self.literal_report
        if report is None or report.get("aborted"):
            return self._stage_reason(6) or "literal report unavailable or aborted"
        if index in report["resolved_edges"]:
            value = report["resolved_edges"][index]
            return f"literal resolved_edges: {_json(value)}"
        if index in report["env_refs"]:
            return f"literal env_var: {_json(report['env_refs'][index])}"
        if index in report["pending_constprop"]:
            return f"literal pending_constprop: {_json(report['pending_constprop'][index])}"
        # Samples use only file:line, so they cannot identify multiple calls on
        # one line. Never assign an aggregate outcome to an ambiguous edge.
        location = f"{edge.file}:{edge.line}"
        same_line = sum(e.file == edge.file and e.line == edge.line for e in self.edges)
        outcomes = [outcome for outcome, samples in report["samples"].items() if location in samples]
        if same_line == 1 and len(outcomes) == 1:
            return f"literal outcome={outcomes[0]}"
        return "literal outcome unavailable per edge (report has only capped location samples)"

    def summary(self):
        from analyzer_new.gaps import summarize_gaps

        setters = {}
        for row in self.rows:
            previous = row["edges_before"]["hosts"]
            for index, host in enumerate(row["edges_after"]["hosts"]):
                old = previous[index] if index < len(previous) else None
                if host != old:
                    if host:
                        setters[index] = HOST_STAGES.get(row["stage"], "UNKNOWN")
                    else:
                        setters.pop(index, None)
        edges = self.edges or []
        settled = Counter()
        trace = []
        remaining = unattributed = 0
        for index, edge in enumerate(edges):
            if edge.status != "external_candidate":
                continue
            owner = setters.get(index, "UNKNOWN") if edge.host else "UNKNOWN"
            if edge.host and owner in HOST_STAGES.values():
                settled[owner] += 1
            elif edge.host:
                unattributed += 1
            else:
                remaining += 1
            reasons = [self._literal_reason(index, edge)]
            pending = (self.constprop_report or {}).get("still_pending", {}).get(index)
            if pending is not None:
                reasons.append(f"constprop failure kind={pending}")
            elif self._stage_reason(7):
                reasons.append(self._stage_reason(7))
            candidates = (self.apply_report or {}).get("joern_candidates", {}).get(index, [])
            if not edge.host:
                reasons.append("Joern " + self._joern_reason(index, edge, candidates))
                if self._stage_reason(12):
                    reasons.append(self._stage_reason(12))
            trace.append(dict(edge_index=index, file=edge.file, line=edge.line,
                              callee=edge.callee_expr, host=edge.host, settled_by=owner,
                              reason="; ".join(reasons)))
        external_count = len(trace)
        accounted = sum(settled.values()) + remaining
        funnel = {"total_edges": len(edges), "external_candidate": external_count,
                  "hosts_set_by_literal": settled["literal"],
                  "hosts_set_by_constprop": settled["constprop"],
                  "hosts_set_by_config": settled["config"],
                  "hosts_set_by_joern": settled["joern"],
                  "external_candidate_without_host": remaining,
                  "hosts_without_known_setter": unattributed, "accounted": accounted,
                  "check": "OK" if accounted == external_count and not unattributed else "MISMATCH"}
        by_stage = defaultdict(list)
        for gap in self.gaps:
            by_stage[gap.stage].append(gap)
        gap_report = {stage: summarize_gaps(values) for stage, values in by_stage.items()}
        statuses = [dict(stage=row["stage"], name=row["name"], status=row["status"],
                         reason=row["reason"], errors=row["errors"])
                    for row in self.rows if row["status"] in {"FAILED", "SKIPPED"}]
        summary = {"environment": self.environment, "source": self.args.source, "root": self.root,
                   "timings": [dict(stage=row["stage"], name=row["name"], seconds=row["seconds"])
                               for row in self.rows],
                   "funnel": funnel, "gaps_by_stage_and_kind": gap_report,
                   "gaps_added_by_pipeline_stage": {str(row["stage"]): row["gaps_added_by_kind"]
                                                    for row in self.rows},
                   "gaps_by_kind": summarize_gaps(self.gaps),
                   "failed_or_skipped": statuses, "edge_trace_count": len(trace)}
        print(f"\n=== FINAL SUMMARY ({self.environment}) ===")
        print("Timing table: stage | name | seconds")
        for row in self.rows:
            print(f"{row['stage']:2d} | {row['name']} | {row['seconds']:.6f}")
        print(f"Funnel: {_json(funnel)}")
        print(f"funnel check: {funnel['check']}" +
              (f" (external_candidate={external_count}, accounted={accounted}, "
               f"unattributed={unattributed})" if funnel["check"] == "MISMATCH" else ""))
        print(f"Edge trace: {len(trace)} external_candidate edges")
        for item in trace[:200]:
            print(f"{item['file']}:{item['line']} {item['callee']} "
                  f"settled by {item['settled_by']}; {item['reason']}")
        if len(trace) > 200:
            print(f"{len(trace) - 200} more edge trace lines saved in edge_trace.json")
        print(f"Gaps by stage and kind (Gap.stage): {_json(gap_report)}")
        print(f"Gaps added by pipeline stage: {_json(summary['gaps_added_by_pipeline_stage'])}")
        print(f"FAILED or SKIPPED stages: {_json(statuses)}")
        _write_json(self.out / "edge_trace.json", trace)
        _write_json(self.out / "summary.json", summary)

    def run(self):
        actions = (self.clone, self.parse, self.index, self.resolve, self.patterns,
                   self.literal, self.constprop, self.config, self.make_tasks,
                   self.slices, self.run_joern, self.apply, self.finish_gaps)
        try:
            for number, action in enumerate(actions, 1):
                self._stage(number, action)
                if number == 11 and self.args.probe_keyword:
                    try:
                        self.probe()
                    except Exception:
                        print("Probe diagnostics failed:\n" + traceback.format_exc())
            self.summary()
            if self.root is None:
                print(f"unusable source: {self.args.source}")
                return 2
            return 0
        finally:
            if self.temporary is not None:
                self.temporary.cleanup()
            # The original source/checkout remains available through J4 and
            # afterwards. Never call cleanup_workspace here.


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run and diagnose all 13 real pipeline stages")
    parser.add_argument("--source", required=True, help="GitHub URL or local folder")
    parser.add_argument("--out", required=True, help="output directory outside the source folder")
    parser.add_argument("--max-joern-groups", type=int, default=3)
    parser.add_argument("--joern-timeout", type=int, default=300)
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--probe-keyword", action="store_true")
    args = parser.parse_args(argv)
    if args.max_joern_groups < 0 or args.joern_timeout <= 0 or args.max_files <= 0:
        parser.error("--max-joern-groups must be >= 0; --joern-timeout and --max-files must be > 0")
    source = Path(args.source).expanduser()
    local_root = None
    is_url = "://" in args.source or args.source.startswith("git@")
    try:
        if source.is_dir():
            local_root = source.resolve()
            # An unreadable folder is an unusable source, rather than an empty
            # successful parse. This is read-only and does not inspect Docker.
            next(local_root.iterdir(), None)
        elif not is_url:
            detail = "is not a directory" if source.exists() else "does not exist"
            parser.error(f"unusable source: source folder {detail}: {args.source}")
        out = Path(args.out).expanduser().resolve()
        if local_root is not None and (out == local_root or local_root in out.parents):
            parser.error("--out must be outside the source folder; the demo must not write inside it")
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        parser.error(f"unusable source or output folder: {error}")
    # Avoid creating/replacing import caches inside either checkout. This is
    # also useful when the demo is invoked against its own source repository.
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with _logging_to_both(out):
            return _Demo(args, out, local_root).run()
    finally:
        sys.dont_write_bytecode = previous_bytecode


if __name__ == "__main__":
    sys.exit(main())
