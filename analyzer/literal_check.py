"""
Stage 4: Literal check — for external-call candidates, see if the argument
is a string literal we can read directly (e.g. requests.get("https://...")).

Two behaviours worth calling out, because they used to be a source of
wrong-but-confident answers:

1. The check now requires the first argument to *be* a string literal (or
   an f-string / template literal), not merely to *contain* a quoted run.
   The old version searched the whole argument text for any quoted
   fragment, so `BASE_URL + "?key=" + token` resolved to the host `?key=`
   — a resolution that is both wrong and, worse, *blocking*: once
   `edge.host` is set, the const-prop and Joern stages skip the edge
   entirely, so the correct trace never runs.

2. An f-string is only accepted when its *static* part carries the host.
   `f"{base}/v1/users"` tells us nothing about where the request goes, so
   it's left for the backward-tracing stages rather than being claimed
   here. `f"https://api.example.com/v1/{uid}"` is fine — the host is
   static text and the interpolation only affects the path.
"""
import re
from urllib.parse import urlparse

from .argtext import first_argument, string_literal_value, classify, is_bare_identifier
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("literal")

_STRING_LITERAL = re.compile(r'''['"]([^'"]{3,200})['"]''')
_BARE_IDENTIFIER = re.compile(r'^\s*[A-Za-z_$][\w$]*\s*$')

#: An interpolation hole, in either Python f-string or JS template syntax.
_HOLE = re.compile(r"\$?\{[^{}]*\}")


def _host_from_literal(text: str):
    if "://" in text:
        try:
            parsed = urlparse(text)
            if parsed.netloc:
                return parsed.netloc
        except ValueError:
            pass
    return None


def _static_host(body: str):
    """Host from a literal body, ignoring interpolated holes — but only if
    the host itself is static. Returns (host, reason_if_rejected)."""
    if not _HOLE.search(body):
        return _host_from_literal(body), None

    masked = _HOLE.sub("\x00", body)
    host = _host_from_literal(masked)
    if host and "\x00" not in host:
        return host, None
    if host:
        return None, "host_is_interpolated"
    return None, "no_static_host_in_interpolated_literal"


def check_literals(edges, ctx=None):
    """Mutates edges in place. Returns the list for convenience."""
    resolved = 0
    considered = 0

    for edge in edges:
        if edge.status != "external_candidate":
            continue
        considered += 1
        arg = first_argument(edge.arg_text)
        shape = classify(arg)
        _bump(ctx, f"literal.arg_shape.{shape}")

        body = string_literal_value(arg)
        if body is None:
            _bump(ctx, "literal.bail.not_a_literal")
            if TRACE_EDGES:
                log(LOG, "debug", "not a whole-expression literal; deferring",
                    file=edge.file, line=edge.line, shape=shape, arg=arg)
            continue

        host, reject = _static_host(body)
        if reject:
            _bump(ctx, f"literal.bail.{reject}")
            if TRACE_EDGES:
                log(LOG, "debug", "literal has no static host; deferring",
                    file=edge.file, line=edge.line, reason=reject, arg=arg)
            continue

        edge.literal = body
        edge.literal_method = "literal"
        edge.host = host or body
        resolved += 1
        if TRACE_EDGES:
            log(LOG, "debug", "resolved from literal",
                file=edge.file, line=edge.line, host=edge.host)

    if ctx is not None:
        ctx.bump("literal.resolved", resolved)
        log(LOG, "info", "literal check complete",
            candidates=considered, resolved=resolved,
            unresolved=considered - resolved,
            arg_shapes=ctx.counters_with_prefix("literal.arg_shape."))
    return edges


def is_bare_variable(arg_text: str) -> bool:
    """Kept for backward compatibility with callers that pass a full
    argument list rather than a single argument."""
    return is_bare_identifier(first_argument(arg_text))


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)
