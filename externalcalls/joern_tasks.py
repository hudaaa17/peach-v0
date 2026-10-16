"""Create Joern tasks from parser-aligned external-call stage reports.

``config_report`` is accepted for a later phase and is not used in Phase 1.
"""

from dataclasses import dataclass
from typing import Optional

from externalcalls.ast_grep_check import CATALOGUE
from externalcalls.edge_pairing import pair_edges_with_calls
from externalcalls.slice_builder import JoernTask


_IN_SCOPE = {
    "imported_constant",
    "parameter",
    "multiple_assignments",
    "not_found",
    "complex_binding",
}


@dataclass
class JoernLimits:
    # placeholder, not enforced; set after J5 measurements
    max_runs: Optional[int] = None
    max_total_seconds: Optional[int] = None


def _result(tasks=None, skipped=None, counts=None, problem=None):
    return {
        "tasks": tasks if tasks is not None else [],
        "skipped": skipped if skipped is not None else {},
        "counts": counts if counts is not None else {
            "considered": 0, "tasks": 0, "skipped": {}},
        "problem": problem,
    }


def _records_by_path(records):
    if isinstance(records, dict):
        return {getattr(record, "path", path): record for path, record in records.items()}
    return {record.path: record for record in records}


def build_joern_tasks(edges, records, literal_report, constprop_report, config_report=None):
    """Build ordered JoernTask values without mutating inputs or reading files.

    ``config_report`` is currently reserved for later configuration-aware
    filtering; it is deliberately unused in this phase.
    """
    if (literal_report is None or constprop_report is None
            or literal_report.get("aborted", False)
            or constprop_report.get("aborted", False)):
        return _result(problem="report_missing_or_aborted")

    calls, problem = pair_edges_with_calls(edges, records)
    if problem:
        return _result(problem=problem)

    by_path = _records_by_path(records)
    entries = {entry.id: entry for entry in CATALOGUE if entry.kind == "external_call"}
    literal_pending = literal_report.get("pending_constprop", {})
    literal_env = literal_report.get("env_refs", {})
    constprop_env = constprop_report.get("env_refs", {})
    still_pending = constprop_report.get("still_pending", {})

    tasks = []
    skipped = {}
    counts = {"considered": 0, "tasks": 0, "skipped": {}}

    def skip(index, reason):
        skipped[index] = reason
        reason_counts = counts["skipped"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for index, (edge, call) in enumerate(zip(edges, calls)):
        if edge.status != "external_candidate":
            continue
        counts["considered"] += 1
        if edge.host is not None:
            skip(index, "has_host")
            continue
        if index in literal_env or index in constprop_env:
            skip(index, "env_var_config_owned")
            continue
        record = by_path.get(edge.file)
        if record is None:
            skip(index, "no_record")
            continue
        entry = entries.get(edge.external_pattern)
        if entry is None:
            skip(index, "no_catalogue_entry")
            continue
        if call.callee_col < 0:
            skip(index, "no_position")
            continue
        outcome = still_pending.get(index)
        if outcome is None:
            skip(index, "no_reason_recorded")
            continue
        if outcome not in _IN_SCOPE:
            skip(index, f"reason_not_in_scope:{outcome}")
            continue
        pending = literal_pending.get(index, {})
        variable = pending.get("name") if isinstance(pending, dict) else None
        if variable is None:
            skip(index, "no_variable_name")
            continue
        tasks.append(JoernTask(
            task_id=f"edge-{index}",
            edge_index=index,
            file=edge.file,
            line=call.callee_line,
            callee_expr=edge.callee_expr,
            callee_col=call.callee_col,
            reason=outcome,
            variable=variable,
            language=record.language,
            url_arg_index=entry.url_arg_index,
            url_arg_keyword=entry.url_arg_keyword,
            url_arg_property=entry.url_arg_property,
        ))

    counts["tasks"] = len(tasks)
    return _result(tasks=tasks, skipped=skipped, counts=counts)
