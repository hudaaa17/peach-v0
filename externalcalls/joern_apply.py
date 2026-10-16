"""Apply real Joern runner results to external-call edges.

This stage never invokes Joern. It validates literal source positions against
Tree-sitter, evaluates those nodes with ``url_eval``, and records evidence in
the report while changing only ``host``, ``needs_review`` and ``review_reason``
on an eligible edge.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Mapping, Sequence

from analyzer_new.gaps import Gap
from externalcalls.ast_access import find_call_node, get_url_argument, load_tree, unwrap
from externalcalls.ast_grep_check import CATALOGUE
from externalcalls.joern_positions import joern_to_parser_col
from externalcalls.url_eval import evaluate, host_from_text, redact_userinfo


_REVIEW_REASON = "joern_result"
_LITERAL_TYPES = {"string", "template_string", "concatenated_string"}
_TEMP_SUBSCRIPT = re.compile(
    r'''^\s*tmp\d+\s*\[\s*(?P<quote>["'])(?P<key>.*?)(?P=quote)\s*\]\s*$'''
)
_OOM_MARKERS = ("outofmemory", "out of memory", "java heap space", "gc overhead limit")
_TRUNCATION_REASONS = {
    "budget_files", "budget_lines", "caller_limit", "sink_file_too_large",
}
_IMPACTS = {
    "joern_timeout": "Joern timed out; the external-call host is unknown.",
    "joern_out_of_memory": "Joern ran out of memory; the external-call host is unknown.",
    "joern_failed": "Joern did not produce a usable result; the external-call host is unknown.",
    "joern_incomplete": "The Joern flow result is incomplete; the external-call host is unknown.",
    "joern_slice_truncated": "The Joern source slice was truncated; the external-call host is unknown.",
    "joern_no_sources_found": "No Joern source was reported; this is UNKNOWN, not evidence that no source exists.",
    "joern_ambiguous_hosts": "Joern found multiple possible hosts; no host was selected.",
    "joern_source_unreadable": "A Joern source position could not be verified in the repository source.",
    "joern_apply_failed": "Joern results could not be applied to this edge.",
}


def _relative(value):
    value = str(value or "").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _gap(gaps, kind, *, path="", detail="", samples=()):
    path = _relative(path)
    scope = "file" if path else "repo"
    gap = Gap(
        stage="joern",
        scope=scope,
        kind=kind,
        path=path,
        detail=str(detail),
        impact=_IMPACTS[kind],
        count=1,
        samples=list(samples)[:5],
    )
    key = (gap.stage, gap.scope, gap.kind, gap.path, gap.detail)
    if not any((old.stage, old.scope, old.kind, old.path, old.detail) == key
               for old in gaps):
        gaps.append(gap)
    return gap


def _line_text(repo_root, relative, line_number):
    if type(line_number) is not int or line_number < 1:
        return None
    text = (Path(repo_root) / relative).read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if line_number > len(lines):
        return None
    return lines[line_number - 1]


def _slice_files(task, task_slices):
    source_slice = (task_slices or {}).get(task.task_id)
    files = list(getattr(source_slice, "files", ()) or ())
    if task.file not in files:
        files.insert(0, task.file)
    return [_relative(path) for path in files]


def _resolve_source_file(repo_root, task, source, task_slices):
    reported = _relative(source.get("file", ""))
    if not reported:
        return "", None
    root = Path(repo_root).resolve()
    candidates = []
    for relative in _slice_files(task, task_slices):
        pure = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if (pure.is_absolute() or windows.is_absolute() or ".." in pure.parts
                or (pure.parts and pure.parts[0].endswith(":"))):
            continue
        matches = (relative == reported or relative.endswith("/" + reported)
                   or reported.endswith("/" + relative)
                   or ("/" not in reported and pure.name == reported))
        if matches and relative not in candidates:
            candidates.append(relative)

    existing = []
    for relative in candidates:
        full = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if (full == root or root in full.parents) and full.is_file():
            existing.append(relative)
    if len(existing) == 1:
        return existing[0], root / Path(*PurePosixPath(existing[0]).parts)
    if len(existing) > 1:
        return reported, None

    pure = PurePosixPath(reported)
    if (not pure.is_absolute() and ".." not in pure.parts
            and not (pure.parts and pure.parts[0].endswith(":"))):
        full = (root / Path(*pure.parts)).resolve()
        if full != root and root in full.parents and full.is_file():
            return pure.as_posix(), full
    return candidates[0] if len(candidates) == 1 else reported, None


def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _literal_at(tree, line, byte_column):
    expected = (line - 1, byte_column)
    matches = [node for node in _walk(tree.root_node)
               if node.type in _LITERAL_TYPES and tuple(node.start_point) == expected]
    if not matches:
        return None
    # Prefer the actual string leaf over a same-position concatenation wrapper.
    matches.sort(key=lambda node: (node.type == "concatenated_string",
                                  node.end_byte - node.start_byte))
    return matches[0]


def _source_subscript_keys(source):
    keys = set()
    for element in source.get("elements", ()) or ():
        if not isinstance(element, dict):
            continue
        code = str(element.get("code", ""))
        match = _TEMP_SUBSCRIPT.fullmatch(code)
        if match is None:
            continue
        quoted = match.group("quote") + match.group("key") + match.group("quote")
        try:
            key = ast.literal_eval(quoted)
        except (SyntaxError, ValueError):
            continue
        if isinstance(key, str):
            keys.add(key)
    return keys


def _sink_key(task, edge, repo_root, cache):
    """Return (kind, key): none, static literal key, or dynamic key."""
    entry = next((candidate for candidate in CATALOGUE
                  if candidate.id == getattr(edge, "external_pattern", None)), None)
    if entry is None:
        return "none", None
    tree, source, error = load_tree(repo_root, task.file, cache)
    if error:
        raise ValueError(f"sink source parse failed for {task.file}: {error}")
    sink = find_call_node(tree, task.line, task.callee_col)
    if isinstance(sink, str):
        raise ValueError(f"sink node unavailable for {task.file}:{task.line}: {sink}")
    argument, error = get_url_argument(sink, entry, str(task.language).lower())
    if error or argument is None:
        raise ValueError(f"URL argument unavailable for {task.file}:{task.line}: {error}")
    argument = unwrap(argument)
    if argument is None or argument.type not in ("subscript", "subscript_expression"):
        return "none", None
    key_node = argument.child_by_field_name(
        "subscript" if argument.type == "subscript" else "index")
    key_node = unwrap(key_node)
    if key_node is None or key_node.type != "string":
        return "dynamic", None
    value = evaluate(key_node, source, str(task.language).lower())
    if not value["complete"] or value["blocked_by"] is not None:
        return "dynamic", None
    return "literal", value["text"]


def _result_sources(result):
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        raise TypeError("runner result sources must be a list")
    return sources


def _read_candidate(source, task, repo_root, task_slices, tree_cache):
    """Read and evaluate one source exclusively from its repository AST node."""
    relative, full_path = _resolve_source_file(repo_root, task, source, task_slices)
    line = source.get("line")
    joern_column = source.get("column")
    if full_path is None:
        return {
            "candidate": {"literal": None, "host": None, "file": relative,
                          "line": line, "column": None, "key_filtered": False},
            "error": f"source file is unreadable or ambiguous: {source.get('file')!r}",
        }
    try:
        line_text = _line_text(repo_root, relative, line)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        line_text = None
        line_error = f"{type(error).__name__}: {error}"
    else:
        line_error = "line number is unavailable or outside the source file"
    if line_text is None:
        return {
            "candidate": {"literal": None, "host": None, "file": relative,
                          "line": line, "column": None, "key_filtered": False},
            "error": line_error,
        }
    byte_column, position_error = joern_to_parser_col(
        line_text, joern_column, str(task.language).lower())
    if position_error:
        return {
            "candidate": {"literal": None, "host": None, "file": relative,
                          "line": line, "column": None, "key_filtered": False},
            "error": f"Joern column {joern_column!r} could not be converted: {position_error}",
        }
    tree, source_bytes, parse_error = load_tree(repo_root, relative, tree_cache)
    if parse_error:
        return {
            "candidate": {"literal": None, "host": None, "file": relative,
                          "line": line, "column": byte_column, "key_filtered": False},
            "error": f"source parse failed: {parse_error}",
        }
    node = _literal_at(tree, line, byte_column)
    if node is None:
        return {
            "candidate": {"literal": None, "host": None, "file": relative,
                          "line": line, "column": byte_column, "key_filtered": False},
            "error": f"no Tree-sitter string literal begins at {relative}:{line}:{byte_column}",
        }
    evaluated = evaluate(node, source_bytes, str(task.language).lower())
    host, host_reason = host_from_text(evaluated["text"], evaluated["complete"])
    if host_reason == "port_boundary_host":
        host = None
    return {
        "candidate": {"literal": redact_userinfo(evaluated["text"]),
                      "host": host, "file": relative, "line": line,
                      "column": byte_column, "key_filtered": False},
        "error": None,
    }


def _task_truncated(task, task_slices, result):
    source_slice = (task_slices or {}).get(task.task_id)
    omitted = getattr(source_slice, "omitted", ()) or ()
    reasons = [str(item[1]) for item in omitted
               if isinstance(item, (tuple, list)) and len(item) > 1]
    reason = str(result.get("reason") or "")
    return (any(item in _TRUNCATION_REASONS for item in reasons)
            or "truncat" in reason.lower()
            or any(item in reason.lower() for item in _TRUNCATION_REASONS))


def _failure_kind(reason):
    normalized = str(reason or "").lower()
    if "timeout" in normalized or "timed out" in normalized:
        return "joern_timeout"
    if any(marker in normalized for marker in _OOM_MARKERS):
        return "joern_out_of_memory"
    return "joern_failed"


def apply_joern_results(
    edges: Sequence,
    tasks: Sequence,
    results,
    repo_root,
    *,
    task_slices: Mapping[str, object] | None = None,
    gaps: list[Gap] | None = None,
):
    """Apply runner results to edges without launching Joern.

    ``task_slices`` maps task ids to the real ``Slice`` objects used for the
    runner call; it disambiguates Joern's basename-only file names. The caller
    may pass a shared gap list so retries remain idempotent.
    """
    edges = list(edges)
    tasks = list(tasks)
    report_gaps = gaps if gaps is not None else []
    if isinstance(results, Mapping):
        results_by_id = dict(results)
    else:
        results_by_id = {
            item["task_id"]: item for item in results
            if isinstance(item, dict) and item.get("task_id") is not None
        }
    report = {
        "joern_candidates": {},
        "gaps": report_gaps,
        "counters": {
            "edges_considered": len(edges),
            "host_set": 0,
            "ambiguous": 0,
            "unknown": 0,
            "failed": 0,
            "incomplete": 0,
            "skipped": {},
        },
        "resolved_edges": {},
        "outcomes": {},
    }
    counters = report["counters"]
    skip_counts = counters["skipped"]
    outcomes = report["outcomes"]
    task_by_edge = {}
    for task in tasks:
        index = getattr(task, "edge_index", None)
        if type(index) is int and 0 <= index < len(edges):
            task_by_edge.setdefault(index, []).append(task)

    def count_skip(reason):
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    def count_outcome(name):
        outcomes[name] = outcomes.get(name, 0) + 1

    tree_cache = {}
    root = Path(repo_root)
    for edge_index, edge in enumerate(edges):
        edge_tasks = task_by_edge.get(edge_index, [])
        if not edge_tasks:
            count_skip("no_task")
            continue
        if getattr(edge, "status", None) != "external_candidate":
            count_skip("not_external_candidate")
            continue
        if getattr(edge, "host", None) not in (None, ""):
            count_skip("has_host")
            continue
        task = edge_tasks[0]
        if len(edge_tasks) > 1:
            count_skip("multiple_tasks_for_edge")
            continue
        task_id = getattr(task, "task_id", None)
        edge_candidates = []
        report["joern_candidates"][edge_index] = edge_candidates
        original_result = results_by_id.get(task_id)
        if original_result is None:
            result = {"status": "failed", "reason": f"runner result missing for task {task_id!r}",
                      "sources": []}
        else:
            result = original_result
        try:
            status = result.get("status")
            reason = str(result.get("reason") or "")
            if status == "failed":
                kind = _failure_kind(reason)
                _gap(report_gaps, kind, path=task.file, detail=reason)
                counters["failed"] += 1
                count_outcome(kind)
                continue
            if status == "no_sources_found":
                _gap(report_gaps, "joern_no_sources_found", path=task.file,
                     detail=f"{reason} This result is UNKNOWN, not evidence that no source exists.")
                counters["unknown"] += 1
                count_outcome("no_sources_found")
                continue
            if status not in ("complete", "incomplete"):
                raise ValueError(f"unsupported runner status: {status!r}")

            key_kind, selected_key = _sink_key(task, edge, root, tree_cache)
            unsafe_key = key_kind == "dynamic"
            if (key_kind == "literal"
                    and str(task.language).lower() in ("javascript", "typescript", "js", "ts")):
                # The saved Joern evidence only establishes Python's tmpN["key"] form.
                unsafe_key = True
            if unsafe_key:
                edge.needs_review = True

            for source in _result_sources(result):
                if not isinstance(source, dict):
                    raise TypeError("runner source entry must be an object")
                candidate_info = _read_candidate(
                    source, task, root, task_slices, tree_cache)
                candidate = candidate_info["candidate"]
                if candidate_info["error"]:
                    edge_candidates.append(candidate)
                    _gap(report_gaps, "joern_source_unreadable", path=task.file,
                         detail=(f"{candidate_info['error']} (Joern position "
                                 f"{source.get('file')!r}:{source.get('line')!r}:"
                                 f"{source.get('column')!r})"),
                         samples=[candidate.get("file", "")])
                    continue

                matching_keys = _source_subscript_keys(source)
                if key_kind == "literal" and not unsafe_key and matching_keys:
                    if selected_key not in matching_keys:
                        continue
                    candidate["key_filtered"] = True
                edge_candidates.append(candidate)

            candidate_hosts = sorted({candidate["host"] for candidate in edge_candidates
                                      if candidate.get("host")})
            if len(candidate_hosts) > 1:
                counters["ambiguous"] += 1
                _gap(report_gaps, "joern_ambiguous_hosts", path=task.file,
                     detail="Distinct Joern candidate hosts: " + ", ".join(candidate_hosts),
                     samples=candidate_hosts)
                count_outcome("ambiguous_hosts")

            if status == "incomplete":
                counters["incomplete"] += 1
                gap_kind = ("joern_slice_truncated" if _task_truncated(task, task_slices, result)
                            else "joern_incomplete")
                _gap(report_gaps, gap_kind, path=task.file,
                     detail=reason or "Joern marked this result incomplete.",
                     samples=[candidate.get("file", "") for candidate in edge_candidates])
                count_outcome("incomplete")
                continue

            if not edge_candidates:
                counters["unknown"] += 1
                count_outcome("unknown")
                continue

            if unsafe_key:
                if len(candidate_hosts) == 0:
                    counters["unknown"] += 1
                count_outcome("dynamic_key_unknown")
                continue

            if len(candidate_hosts) == 1:
                host = candidate_hosts[0]
                edge.host = host
                edge.needs_review = True
                edge.review_reason = _REVIEW_REASON
                counters["host_set"] += 1
                report["resolved_edges"][edge_index] = {"host": host}
                count_outcome("host_set")
            elif not candidate_hosts:
                counters["unknown"] += 1
                count_outcome("unknown")
            else:
                count_outcome("ambiguous_hosts")
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            _gap(report_gaps, "joern_apply_failed", path=getattr(task, "file", ""),
                 detail=detail)
            counters["failed"] += 1
            count_outcome("apply_failed")

    # Invalid task indices have no corresponding edge path to attach a file gap to.
    for task in tasks:
        index = getattr(task, "edge_index", None)
        if type(index) is int and 0 <= index < len(edges):
            continue
        detail = f"task edge_index is outside the edge list: {index!r}"
        _gap(report_gaps, "joern_apply_failed", detail=detail)
        counters["failed"] += 1
        count_outcome("apply_failed")
    return report


__all__ = ["apply_joern_results"]
