"""Inspect catalogue URL arguments without changing edge classification."""

from .ast_access import find_call_node, get_url_argument, load_tree
from .ast_grep_check import CATALOGUE
from .edge_pairing import pair_edges_with_calls
from .url_eval import evaluate, host_from_text, redact_userinfo


def run_literal_check(edges, records, repo_root, ctx=None):
    report = {"aborted": False, "edges_total": 0, "edges_considered": 0,
              "already_set": 0, "userinfo_redacted": 0, "outcomes": {}, "per_pattern": {}, "samples": {},
              "reasons": [], "pending_constprop": {}, "env_refs": {}, "resolved_edges": {}}
    try:
        calls, problem = pair_edges_with_calls(edges, records)
        report["edges_total"] = len(edges)
        if problem:
            report["aborted"] = True
            report["reasons"].append(problem)
            return report
        entries = {entry.id: entry for entry in CATALOGUE if entry.kind == "external_call"}
        languages = {rec.path: rec.language for rec in records}
        cache = {}
        fields = ("literal", "literal_method", "host", "needs_review", "review_reason")
        for index, (edge, call) in enumerate(zip(edges, calls)):
            entry = entries.get(edge.external_pattern)
            if edge.status != "external_candidate" or entry is None:
                continue
            report["edges_considered"] += 1
            before = None
            pending = env_ref = resolved = None
            redacted = False
            try:
                before = {field: getattr(edge, field) for field in fields}
                changes = {}
                if edge.literal is not None:
                    outcome = "already_set"
                elif call.callee_col < 0:
                    outcome = "no_position"
                else:
                    tree, source, error = load_tree(repo_root, edge.file, cache)
                    if error:
                        outcome = error
                    else:
                        node = find_call_node(tree, call.callee_line, call.callee_col)
                        if isinstance(node, str):
                            outcome = node
                        else:
                            language = languages[edge.file]
                            argument, error = get_url_argument(node, entry, language)
                            if error:
                                outcome = error
                            else:
                                result = evaluate(argument, source, language)
                                text, complete = result["text"], result["complete"]
                                blocked = result["blocked_by"]
                                host, reason = host_from_text(text, complete)
                                if blocked and blocked["kind"] == "identifier" and (host is None or reason == "port_boundary_host"):
                                    pending = {"name": blocked["name"],
                                               "arg_line": argument.start_point[0] + 1,
                                               "arg_col": argument.start_point[1]}
                                if blocked and blocked["kind"] == "env_var":
                                    env_ref = {"var": blocked["name"]}
                                    outcome = "env_var"
                                elif text:
                                    outcome = ("full_url" if complete else "partial_url") if host else reason
                                    if reason == "port_boundary_host":
                                        outcome = reason
                                    safe_text = redact_userinfo(text)
                                    redacted = safe_text != text
                                    text = safe_text
                                    changes = {"literal": text, "literal_method": "literal", "host": host}
                                    if not complete or host is None:
                                        changes.update(needs_review=True,
                                                       review_reason=(reason or "partial_url") if host else reason)
                                    resolved = {"literal": text, "host": host, "partial": not complete}
                                else:
                                    outcome = blocked["kind"] if blocked else "empty_string"
                for field, value in changes.items():
                    setattr(edge, field, value)
            except Exception as exc:
                if before is not None:
                    for field, value in before.items():
                        setattr(edge, field, value)
                pending = env_ref = resolved = None
                redacted = False
                outcome = "edge_failed"
                report["reasons"].append({"cause": outcome, "detail": f"{type(exc).__name__}: {exc}",
                                          "sample": f"{edge.file}:{edge.line}"})
            if pending is not None:
                report["pending_constprop"][index] = pending
            if env_ref is not None:
                report["env_refs"][index] = env_ref
            if resolved is not None:
                report["resolved_edges"][index] = resolved
            report["already_set"] += outcome == "already_set"
            report["userinfo_redacted"] += redacted
            report["outcomes"][outcome] = report["outcomes"].get(outcome, 0) + 1
            counts = report["per_pattern"].setdefault(entry.id, {})
            counts[outcome] = counts.get(outcome, 0) + 1
            samples = report["samples"].setdefault(outcome, [])
            sample = f"{edge.file}:{edge.line}"
            if sample not in samples and len(samples) < 5:
                samples.append(sample)
    except Exception as exc:
        report["aborted"] = True
        report["reasons"].append({"cause": "stage_failed", "detail": f"{type(exc).__name__}: {exc}"})
    try:
        if ctx is not None:
            ctx.bump("literal_check.edges_considered", report["edges_considered"])
            for outcome, count in report["outcomes"].items():
                ctx.bump(f"literal_check.outcome.{outcome}", count)
    except Exception:
        pass
    return report
