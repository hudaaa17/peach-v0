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
import ast

from urllib.parse import urlparse

from .argtext import first_argument, string_literal_value, classify, is_bare_identifier
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("literal")


def _ast_static_dynamic_split(arg: str):
    """Real tier: if `arg` is valid standalone Python source and it's an
    f-string, `ast.parse` gives the *exact* static/dynamic breakdown —
    `ast.Constant` values are always literal text, `ast.FormattedValue`
    nodes are always a real interpolation. This replaces guessing hole
    boundaries from already-flattened text with reading the real parse
    tree, so it isn't fooled by an escaped brace (`f"{{literal}}"`) or a
    nested-brace expression inside the hole itself (`f"{ {'a': 1} }"`),
    both of which a brace-counting regex over the rendered text cannot
    tell apart from a second hole.

    Returns the static text with every interpolation replaced by a single
    sentinel byte, or `None` if `arg` isn't valid Python or isn't an
    f-string at all (not applicable — the caller falls back to the
    depth-aware scanner below, which is what covers every other
    language's template-literal syntax)."""
    try:
        tree = ast.parse(arg, mode="eval")
    except (SyntaxError, ValueError):
        return None
    node = tree.body
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        else:
            parts.append("\x00")
    return "".join(parts)


def _mask_holes_depth_aware(body: str) -> str:
    """Fallback for languages `ast.parse` can't read at all (JS/TS
    template literals, and everything else this pipeline parses with the
    regex-fallback path in parser.py). Replaces every top-level `{...}`
    or `${...}` hole with a single sentinel, tracking brace depth so a
    hole containing its own braces — `${JSON.stringify({a: 1})}` — is
    masked as one hole instead of the old flat regex's failure mode: it
    disallowed any brace inside `[^{}]*`, so it stopped at the *first*
    inner `}` and left the rest of the real hole sitting in the "static"
    text, which could fabricate a host out of interpolated content."""
    out = []
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "{" or (ch == "$" and i + 1 < n and body[i + 1] == "{"):
            i += 2 if ch == "$" else 1
            depth = 1
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            out.append("\x00")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _host_from_literal(text: str):
    if "://" in text:
        try:
            parsed = urlparse(text)
            if parsed.netloc:
                return parsed.netloc
        except ValueError:
            pass
    return None


def _static_host(body: str, arg: str):
    """Host from a literal body, ignoring interpolated holes — but only if
    the host itself is static. Returns (host, reason_if_rejected).

    Hole detection is tried in two tiers, same shape as the rest of this
    pipeline: real `ast` parsing of `arg` when it's valid standalone
    Python (exact — see `_ast_static_dynamic_split`), and a depth-aware
    bracket scanner over `body` for everything `ast.parse` can't read."""
    masked = _ast_static_dynamic_split(arg)
    if masked is None:
        masked = _mask_holes_depth_aware(body)

    if "\x00" not in masked:
        return _host_from_literal(body), None

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

        host, reject = _static_host(body, arg)
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