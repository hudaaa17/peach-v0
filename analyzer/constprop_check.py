"""
Stage 5: Constant-prop check — when the call argument isn't a readable
literal, do a best-effort backward trace within the same file to find the
most recent assignment of the relevant variable to a string literal above
the call site.

Widened slightly from the original: the trace no longer requires the
argument to be a *bare* identifier. `BASE_URL + "/v1/login"` and
`f"{BASE_URL}/v1/login"` are both extremely common and both used to be
dropped on the floor here, even though the thing worth tracing —
`BASE_URL` — is sitting right there in the expression. We now pull the
candidate identifiers out of the expression and try each in turn, most
endpoint-looking name first. A bare identifier is just the one-candidate
case of that.

This stays a *same-file, above-the-call-site* trace. Anything needing to
cross a function or file boundary is still the Joern stage's job.
"""
import re

from .argtext import first_argument, identifier_roots, classify
from .literal_check import _host_from_literal
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("constprop")

_ASSIGN_PATTERN_TMPL = r'''^\s*(?:const\s+|let\s+|var\s+)?{name}\s*(?::\s*\w+\s*)?=\s*['"]([^'"]{{3,200}})['"]'''


def check_constant_propagation(edges, records_by_path, ctx=None):
    """Mutates edges in place. `records_by_path` maps rel path -> FileRecord."""
    attempted = 0
    resolved = 0

    for edge in edges:
        if edge.status != "external_candidate" or edge.literal:
            continue
        attempted += 1

        arg = first_argument(edge.arg_text)
        # Dotted names (`self.base_url`, `cfg.host`) are deliberately left
        # alone here. Their assignment almost always lives in a *different
        # function* — `__init__`, a setter, a factory — which is the Joern
        # stage's job by the pipeline's own division of labour. Claiming
        # them here would resolve some of them, but it would also make the
        # per-stage numbers lie about which technique did the work.
        candidates = [n for n in identifier_roots(arg) if "." not in n]
        if not candidates:
            _bump(ctx, "constprop.bail.no_traceable_identifier")
            if TRACE_EDGES:
                log(LOG, "debug", "no identifier to trace", file=edge.file,
                    line=edge.line, shape=classify(arg), arg=arg)
            continue

        rec = records_by_path.get(edge.file)
        if not rec:
            _bump(ctx, "constprop.bail.no_source_record")
            log(LOG, "warning", "no parsed record for edge file",
                file=edge.file, line=edge.line)
            continue

        hit = None
        for name in candidates:
            pattern = re.compile(_ASSIGN_PATTERN_TMPL.format(name=re.escape(name)))
            best = None
            for line in rec.source_lines[: max(0, edge.line - 1)]:
                m = pattern.match(line)
                if m:
                    best = m.group(1)  # keep the *last* match before the call line
            if best:
                hit = (name, best)
                break

        if not hit:
            _bump(ctx, "constprop.bail.no_assignment_above_call")
            if TRACE_EDGES:
                log(LOG, "debug", "no literal assignment found above call site",
                    file=edge.file, line=edge.line, tried=candidates)
            continue

        name, value = hit
        edge.literal = value
        edge.literal_method = "const_prop"
        edge.host = _host_from_literal(value) or value
        resolved += 1
        _bump(ctx, "constprop.resolved")
        if TRACE_EDGES:
            log(LOG, "debug", "resolved by same-file backward trace",
                file=edge.file, line=edge.line, variable=name, host=edge.host)

    if ctx is not None:
        log(LOG, "info", "constant propagation complete",
            attempted=attempted, resolved=resolved,
            bail_reasons=ctx.counters_with_prefix("constprop.bail."))
    return edges


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)
