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

Matching is tried in two tiers per traced identifier, the same shape the
resolver/literal-check stages use:

  1. A real AST backward scan (`_ast_backward_trace`) — for a file
     `ast.parse` can actually read (i.e. Python), walk the *real* parse
     tree for `Assign`/`AnnAssign` nodes targeting the identifier. This
     is what the old single-line regex
     (`{name}\\s*=\\s*['"]([^'"]{{3,200}})['"]`) always should have been:
     it sees an f-string assignment (`BASE_URL = f"http://{host}/api"`,
     silently unmatched before, because the regex has no `f`/`r`/`b`
     prefix handling at all), a triple-quoted or escaped-quote string,
     and — critically — it cannot mistake a commented-out assignment
     for a live one, because a parsed AST doesn't contain comments in
     the first place. The old regex had no such protection: `#
     BASE_URL = "http://old-host"` matched and resolved exactly like
     real code.
  2. The line-regex scan (`_regex_backward_trace`) below — the real
     fallback for every language this pipeline has no AST for (JS, Go,
     Java, Ruby, PHP, C#, Rust). It's hardened to skip whole-line
     comments, which the original never did.

Tier 2 only runs when tier 1 isn't applicable at all (the file doesn't
parse as Python) — not merely when tier 1 came up empty. A file that
*does* parse as Python has already been read completely and correctly;
re-guessing with the regex on top of that would just reintroduce the
comment/prefix bugs tier 1 exists to avoid, for no gain.

This stays a *same-file, above-the-call-site* trace. Anything needing to
cross a function or file boundary is still the Joern stage's job.
"""
import ast
import re

from .argtext import first_argument, identifier_roots, classify
from .literal_check import _host_from_literal
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("constprop")

_ASSIGN_PATTERN_TMPL = r'''^\s*(?:const\s+|let\s+|var\s+)?{name}\s*(?::\s*\w+\s*)?=\s*['"]([^'"]{{3,200}})['"]'''
_WHOLE_LINE_COMMENT = re.compile(r'^\s*(#|//)')


def _split_string_node(value_node):
    """Real value extraction from an assignment's AST node. Returns
    (display_text, masked_text):

    - `display_text` is the value as source-like text (an f-string keeps
      its `{expr}` holes, for readability) — the AST equivalent of what
      the old regex's capture group used to give us for a plain literal.
    - `masked_text` is the same text with every real interpolation
      (`ast.FormattedValue`) replaced by a single sentinel byte, so a
      caller can tell whether a host derived from it is genuinely static.

    Returns (None, None) if the node isn't a string expression at all —
    a number, a call, another variable, ... — the same thing the old
    regex did by simply not matching those lines.
    """
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
        return value_node.value, value_node.value
    if isinstance(value_node, ast.JoinedStr):
        display_parts, masked_parts = [], []
        for v in value_node.values:
            if isinstance(v, ast.Constant):
                display_parts.append(str(v.value))
                masked_parts.append(str(v.value))
            else:
                try:
                    inner = ast.unparse(v.value)
                except Exception:
                    inner = "..."
                display_parts.append("{" + inner + "}")
                masked_parts.append("\x00")
        return "".join(display_parts), "".join(masked_parts)
    return None, None


def _resolve_string_value(value_node):
    """(display_text, host) for a node that's a usable string expression
    with a static host, or None if it isn't a string at all, or is an
    interpolated string with no static host to anchor on — deferred, the
    same way literal_check.py defers those rather than guessing."""
    display, masked = _split_string_node(value_node)
    if display is None:
        return None
    if "\x00" not in masked:
        return display, (_host_from_literal(display) or display)
    host = _host_from_literal(masked)
    if host and "\x00" not in host:
        return display, host
    return None


def _ast_backward_trace(source_text, edge_line, candidates):
    """Tier 1. Returns `(applicable, hit)`:

    - `applicable` is False when `source_text` isn't parseable Python at
      all — the caller should fall back to the regex tier.
    - `applicable` is True whenever it *is* real Python, whether or not a
      hit was found. A parse that succeeds but finds nothing is a
      genuine miss, not a reason to also try the fallback: the AST scan
      already saw every assignment in the file.
    - `hit` is `(name, display_text, host)` for the closest assignment
      above `edge_line` among `candidates` (tried in the caller's given
      order), or `None`.
    """
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError):
        return False, None

    # name -> (line, value_node) of the closest preceding assignment
    best_by_name = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        line = getattr(node, "lineno", None)
        if line is None or line >= edge_line:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in candidates:
                current = best_by_name.get(target.id)
                if current is None or line > current[0]:
                    best_by_name[target.id] = (line, value)

    for name in candidates:
        found = best_by_name.get(name)
        if not found:
            continue
        resolved = _resolve_string_value(found[1])
        if resolved:
            display, host = resolved
            return True, (name, display, host)
    return True, None


def _regex_backward_trace(rec, edge_line, candidates):
    """Tier 2: the line-anchored regex scan, for languages this pipeline
    has no AST for. Returns `(name, value)` or `None`. Whole-line
    comments are skipped so a commented-out assignment can't be read as
    live code — the one bug in the original that had nothing to do with
    which language it was scanning."""
    for name in candidates:
        pattern = re.compile(_ASSIGN_PATTERN_TMPL.format(name=re.escape(name)))
        best = None
        for line in rec.source_lines[: max(0, edge_line - 1)]:
            if _WHOLE_LINE_COMMENT.match(line):
                continue
            m = pattern.match(line)
            if m:
                best = m.group(1)  # keep the *last* match before the call line
        if best:
            return name, best
    return None


def check_constant_propagation(edges, records_by_path, ctx=None):
    """Mutates edges in place. `records_by_path` maps rel path -> FileRecord."""
    attempted = 0
    resolved = 0
    via_counts = {"ast": 0, "regex_fallback": 0}

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

        source_text = "\n".join(rec.source_lines)
        applicable, ast_hit = _ast_backward_trace(source_text, edge.line, candidates)
        if applicable:
            via = "ast"
            hit = ast_hit
        else:
            via = "regex_fallback"
            found = _regex_backward_trace(rec, edge.line, candidates)
            hit = None
            if found:
                name, value = found
                hit = (name, value, _host_from_literal(value) or value)

        if not hit:
            _bump(ctx, "constprop.bail.no_assignment_above_call")
            if TRACE_EDGES:
                log(LOG, "debug", "no literal assignment found above call site",
                    file=edge.file, line=edge.line, tried=candidates, tier=via)
            continue

        name, value, host = hit
        edge.literal = value
        edge.literal_method = "const_prop"
        edge.constprop_via = via
        edge.host = host
        resolved += 1
        via_counts[via] += 1
        _bump(ctx, "constprop.resolved")
        _bump(ctx, f"constprop.resolved.via_{via}")
        if TRACE_EDGES:
            log(LOG, "debug", "resolved by same-file backward trace",
                file=edge.file, line=edge.line, variable=name, host=edge.host, tier=via)

    if ctx is not None:
        log(LOG, "info", "constant propagation complete",
            attempted=attempted, resolved=resolved,
            via_ast=via_counts["ast"], via_regex_fallback=via_counts["regex_fallback"],
            bail_reasons=ctx.counters_with_prefix("constprop.bail."))
    return edges


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)