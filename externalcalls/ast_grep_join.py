"""Join ast-grep calls to resolver edges using the original CallSite order.

Pairing validation, positional joins, and degraded-language regex fallback;
no pipeline wiring. Pass the same records used by resolve_calls. Positions are
1-based lines and 0-based UTF-8 byte columns, with exclusive range ends.
hits_ambiguous counts EDGES left unresolved because of ambiguity, not hits.
Fallback covers all still-unresolved eligible edges of a degraded language,
not just reason sample files. It checks callee spelling only, so aliases such
as r.get are not recognized, and constructor calls can match by spelling.
Telemetry failures never abort or discard the join.
"""

from collections import defaultdict
from pathlib import Path
import re

from analyzer_new.parser import LANG_BY_EXT
from .ast_grep_check import CATALOGUE, MAX_SAMPLES, _FAMILY


def _report():
    return {"aborted": False, "edges_total": 0, "edges_eligible": 0,
            "edges_without_position": 0, "hits_total": 0, "hits_marked": 0,
            "hits_unmatched": 0, "hits_ambiguous": 0, "per_pattern": {},
            "fallback_languages": [], "fallback_marked": {}, "reasons": [], "marked": {}}


def _note(report, kind, cause, detail, samples=(), language="", count=1):
    key = kind, cause, language, detail
    for reason in report["reasons"]:
        if tuple(reason[k] for k in ("kind", "cause", "language", "detail")) == key:
            reason["count"] += count
            reason["samples"] = list(dict.fromkeys(reason["samples"] + list(samples)))[:MAX_SAMPLES]
            return
    report["reasons"].append(dict(kind=kind, cause=cause, language=language,
                                  detail=detail, count=count,
                                  samples=list(dict.fromkeys(samples))[:MAX_SAMPLES]))


def _pair(edges, records, report):
    calls = [call for rec in records for call in rec.calls]
    if len(edges) != len(calls):
        report["aborted"] = True
        _note(report, "join_aborted", "edge_count_mismatch",
              f"edges={len(edges)}, CallSites={len(calls)}")
        return None
    for index, (edge, call) in enumerate(zip(edges, calls)):
        for field in ("file", "line", "callee_expr", "caller"):
            edge_value, call_value = getattr(edge, field), getattr(call, field)
            if edge_value != call_value:
                report["aborted"] = True
                _note(report, "join_aborted", "pair_mismatch",
                      f"index={index}, {field}: edge={edge_value!r}, CallSite={call_value!r}",
                      [f"{call.file}:{call.line} {call.callee_expr}"])
                return None
    return calls


def _sample(hit):
    return f"{hit.file}:{hit.start_line}:{hit.start_col} {hit.callee_text} {hit.pattern_id}"


def _plan(edges, calls, hits, report):
    entries = {entry.id: entry for entry in CATALOGUE}
    by_file = defaultdict(lambda: defaultdict(list))
    valid_hits = {}
    for index, hit in enumerate(hits):
        entry = entries.get(hit.pattern_id)
        if entry is None or entry.kind != "external_call":
            _note(report, "hit_ignored", "unknown_pattern_id" if entry is None else "non_external_call",
                  "hit does not identify an external_call catalogue entry", [_sample(hit)])
            continue
        valid_hits[index] = hit
        by_file[hit.file][hit.callee_text].append(index)

    matched, claimed, ambiguous = set(), set(), set()
    files_with_calls = {call.file for call in calls}
    for edge_index, (edge, call) in enumerate(zip(edges, calls)):
        has_position = call.callee_col >= 0
        positioned = has_position and "<expr>" not in call.callee_expr
        eligible = edge.status == "unresolved" and positioned
        if edge.status == "unresolved":
            report["edges_eligible" if positioned else "edges_without_position"] += 1
        if not has_position:
            continue
        position = call.callee_line, call.callee_col
        candidates = [i for i in by_file.get(edge.file, {}).get(edge.callee_expr, ())
                      if (hits[i].start_line, hits[i].start_col) <= position
                      < (hits[i].end_line, hits[i].end_col)]
        matched.update(candidates)  # Other statuses still account for their hits.
        if not eligible or not candidates:
            continue

        def rank(index):
            hit = hits[index]
            return -hit.start_line, -hit.start_col, hit.end_line, hit.end_col

        best = min(map(rank, candidates))
        winners = [i for i in candidates if rank(i) == best]
        if len(winners) != 1 or winners[0] in claimed:
            cause = "equal_range_tie" if len(winners) != 1 else "claimed_twice"
            report["hits_ambiguous"] += 1
            ambiguous.add(edge_index)
            _note(report, "hit_ambiguous", cause,
                  "innermost hit is tied or already claimed",
                  [_sample(hits[i]) for i in winners])
            continue
        chosen = winners[0]
        claimed.add(chosen)
        hit = hits[chosen]
        report["marked"][edge_index] = {"pattern_id": hit.pattern_id, "via": "ast_grep",
                                        "via_alias": bool(getattr(hit, "via_alias", False))}
        report["hits_marked"] += 1
        counts = report["per_pattern"]
        counts[hit.pattern_id] = counts.get(hit.pattern_id, 0) + 1

    for index, hit in valid_hits.items():
        if index not in matched:
            report["hits_unmatched"] += 1
            cause = "no_matching_edge" if hit.file in files_with_calls else "file_without_call_sites"
            _note(report, "hit_without_edge", cause,
                  "hit has no CallSite with matching callee text and position", [_sample(hit)])
    return ambiguous


def _fallback(edges, calls, run_report, report, ambiguous):
    languages = {LANG_BY_EXT.get(Path(edge.file).suffix.lower()) for edge in edges}
    languages = sorted(language for language in languages if language in _FAMILY)
    reasons = run_report["reasons"]
    all_degraded = run_report["binary"] is None or any(
        reason["kind"] in {"ast_grep_unavailable", "stage_failed"} for reason in reasons)
    degraded = [language for language in languages if all_degraded or any(
        reason["kind"] == "ast_grep_language_failed" and reason["language"] in ("", language)
        for reason in reasons)]
    report["fallback_languages"] = degraded
    report["fallback_marked"] = dict.fromkeys(degraded, 0)
    patterns = {language: [(entry, re.compile(entry.callee_regex)) for entry in CATALOGUE
                           if entry.kind == "external_call" and entry.language == _FAMILY[language]]
                for language in degraded}
    for index, (edge, call) in enumerate(zip(edges, calls)):
        language = LANG_BY_EXT.get(Path(edge.file).suffix.lower())
        if (language not in patterns or edge.status != "unresolved" or index in report["marked"]
                or call.callee_col < 0 or "<expr>" in edge.callee_expr):
            continue
        matches = [entry for entry, regex in patterns[language] if regex.search(edge.callee_expr)]
        if len(matches) > 1:
            ambiguous.add(index)
            _note(report, "hit_ambiguous", "multiple_catalogue_entries",
                  "callee spelling matches multiple external_call catalogue entries",
                  [f"{edge.file}:{call.callee_line}:{call.callee_col} {edge.callee_expr}"], language)
        elif len(matches) == 1:
            entry = matches[0]
            ambiguous.discard(index)
            report["marked"][index] = {"pattern_id": entry.id, "via": "regex_fallback", "via_alias": False}
            report["hits_marked"] += 1
            report["per_pattern"][entry.id] = report["per_pattern"].get(entry.id, 0) + 1
            report["fallback_marked"][language] += 1
    report["hits_ambiguous"] = len(ambiguous)
    for language in degraded:
        _note(report, "regex_fallback_used",
              "binary_missing" if run_report["binary"] is None else "language_failed",
              "matched by callee spelling only; aliases and call shape are not checked",
              language=language, count=report["fallback_marked"][language])


def _notify(ctx, report):
    if ctx is not None:
        for key, value in report.items():
            if type(value) is int and value:
                ctx.bump(f"ast_grep_join.{key}", value)
        for reason in report["reasons"]:
            ctx.bump(f"ast_grep_join.reason.{reason['cause']}", reason["count"])


def join_ast_grep_hits(edges, records, hits, run_report, ctx=None):
    """Mark only unresolved edges after validating every pair.

    Plan all marks before committing the three external-call fields. Unexpected
    exceptions abort with stage_failed and roll back any partial commit. The
    complete marked side table carries alias provenance without changing edges.
    run_report controls fallback for degraded parser languages.
    """
    report = _report()
    backups = []
    try:
        report["edges_total"], report["hits_total"] = len(edges), len(hits)
        calls = _pair(edges, records, report)
        if calls is not None:
            ambiguous = _plan(edges, calls, hits, report)
            _fallback(edges, calls, run_report, report, ambiguous)
        try:
            _notify(ctx, report)
        except Exception:
            pass  # Telemetry must never affect analysis or edge mutations.
        for index in report["marked"]:
            edge = edges[index]
            backups.append((edge, {key: getattr(edge, key) for key in
                                   ("status", "external_pattern", "external_via")}))
        for index, mark in report["marked"].items():
            edge = edges[index]
            edge.status = "external_candidate"
            edge.external_pattern = mark["pattern_id"]
            edge.external_via = mark["via"]
    except Exception as exc:
        for edge, previous in backups:
            vars(edge).update(previous)  # Restore CallEdge fields without setters.
        totals = report["edges_total"], report["hits_total"]
        report = _report()
        report["edges_total"], report["hits_total"] = totals
        report["aborted"] = True
        _note(report, "stage_failed", "exception", f"{type(exc).__name__}: {exc}")
    return report
