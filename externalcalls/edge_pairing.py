"""Validate resolver edge order against the original parser CallSites."""


def pair_edges_with_calls(edges, records):
    """Return aligned CallSites or a problem, without changing either input.

    Follow resolve_calls' record/call order and ast_grep_join._pair's checks.
    Unexpected input errors become pair_mismatch problems rather than raising.
    """
    try:
        calls = [call for rec in records for call in rec.calls]
        if len(edges) != len(calls):
            return None, {"cause": "edge_count_mismatch",
                          "detail": f"edges={len(edges)}, CallSites={len(calls)}"}
        for index, (edge, call) in enumerate(zip(edges, calls)):
            for field in ("file", "line", "callee_expr", "caller"):
                edge_value, call_value = getattr(edge, field), getattr(call, field)
                if edge_value != call_value:
                    return None, {"cause": "pair_mismatch",
                                  "detail": f"index={index}, {field}: edge={edge_value!r}, CallSite={call_value!r}"}
        return calls, None
    except Exception as exc:
        return None, {"cause": "pair_mismatch", "detail": f"{type(exc).__name__}: {exc}"}
