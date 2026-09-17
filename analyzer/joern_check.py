"""
Stage 7: Escalate to Joern — a scoped, cross-function/cross-file data-flow
trace, reserved for external-call candidates that literal_check,
constprop_check, and config_check all failed to resolve a host for.

A real Joern deployment builds a code-property graph and runs a precise
inter-procedural data-flow query. We don't ship the Joern binary here (it's
a heavy JVM analysis platform), but we play the same *role* in the
pipeline: pick up exactly where the single-file constant-prop check left
off and widen the search past the boundary of one file/function, using
static tracing techniques that are individually cheap but, combined
end-to-end, are more expensive than anything upstream — which is why this
stage only ever runs on the edges that survive every earlier check.

Four widened traces are attempted, in order, stopping at the first hit:

1. Environment-variable lookups (`os.environ.get("KEY")`, `process.env.KEY`)
   — the value isn't in the repo at all, so we resolve to the *key name*
   and let the config check (re-run after this stage) match it against a
   ConfigMap/manifest.
2. `self.attr` instance attributes — the assignment usually lives in
   `__init__`, a different function than the call site, so this is a
   same-file cross-*function* trace.
3. Imported module-level constants — the assignment lives in a different
   *file* entirely, so this is a cross-file trace.
4. Call-site argument propagation — if the enclosing function itself takes
   the unresolved value as a parameter, we look at how *other* functions in
   the repo call it and pull a literal from the caller's side. This is the
   most Joern-like of the four (real inter-procedural data flow) and also
   the most approximate, so it's gated behind a naming heuristic (see
   `_looks_like_endpoint_param`) to avoid over-escalating on unrelated
   parameters — consistent with the "Escalation Is Heuristic" limitation.

Anything still unresolved after all four is left for the LLM fallback.

Observability
-------------
Every trace reports, per edge, whether it *ran* or was *skipped*, and if
skipped, why. That distinction is the whole point: a stage reporting
"0 resolved" is ambiguous between "ran four traces, none matched" and
"declined to run any trace because the input was unusable". The bail-reason
histogram in the stage summary disambiguates the two at a glance.
"""
import re
from collections import Counter, defaultdict

from .argtext import (
    first_argument, identifier_roots, classify, is_bare_identifier,
    split_top_level, string_literal_value,
)
from .literal_check import _host_from_literal
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("joern")

_SELF_ATTR = re.compile(r'^self\.(\w+)\s*$')
_THIS_ATTR = re.compile(r'^this\.(\w+)\s*$')

_SELF_ASSIGN_TMPL = r'''^\s*(?:self|this)\.{name}\s*(?::\s*\w+\s*)?=\s*['"]([^'"]{{3,200}})['"]'''
_MODULE_ASSIGN_TMPL = r'''^\s*(?:const\s+|let\s+|var\s+)?{name}\s*(?::\s*\w+\s*)?=\s*['"]([^'"]{{3,200}})['"]'''

_ENV_ASSIGN_TMPL = (
    r'''^\s*(?:const\s+|let\s+|var\s+)?{name}\s*(?::\s*\w+\s*)?=\s*'''
    r'''(?:os\.environ\.get\(|os\.environ\[|os\.getenv\()\s*['"]([\w.\-]+)['"]'''
)
_ENV_ASSIGN_JS_TMPL = (
    r'''^\s*(?:const\s+|let\s+|var\s+)?{name}\s*(?::\s*\w+\s*)?=\s*'''
    r'''process\.env(?:\.([\w]+)|\[['"]([\w.\-]+)['"]\])'''
)

#: An env lookup used directly as the argument, with no intermediate
#: variable: `requests.get(os.environ["SVC_URL"])`.
_INLINE_ENV = re.compile(
    r'''(?:os\.environ\.get\(|os\.environ\[|os\.getenv\()\s*['"]([\w.\-]+)['"]'''
    r'''|process\.env(?:\.([\w]+)|\[['"]([\w.\-]+)['"]\])'''
)

_ENDPOINT_HINTS = ("url", "host", "endpoint", "service", "base", "addr", "uri", "target", "api")


class TraceResult:
    """Outcome of one of the four traces for one edge.

    `ran` distinguishes "this trace was applicable and looked" from "this
    trace didn't apply to this shape of input". Without it, a stage that
    never even tries looks identical to a stage that tries and misses."""

    __slots__ = ("name", "ran", "resolved", "reason", "detail")

    def __init__(self, name, ran, resolved=False, reason=None, detail=None):
        self.name = name
        self.ran = ran
        self.resolved = resolved
        self.reason = reason
        self.detail = detail

    def as_dict(self):
        return {"trace": self.name, "ran": self.ran, "resolved": self.resolved,
                "reason": self.reason, "detail": self.detail}


def _looks_like_endpoint_param(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _ENDPOINT_HINTS)


def _last_assignment(source_lines, pattern):
    """Scan the whole file (not just above the call site) since the
    assignment this stage looks for lives in a *different* function/scope,
    which may textually appear before or after the call in the file.

    (The old signature took a `before_line` argument that was never read —
    removed rather than left as a lie about what this does.)"""
    best = None
    for line in source_lines:
        m = pattern.match(line)
        if m:
            best = m.group(1)
    return best


def _path_stem(path: str) -> str:
    """Stem of a repo-relative path, tolerant of either separator.

    Paths are normalised to POSIX at parse time, so this should always see
    forward slashes — but this function is the one that silently returned
    a wrong answer rather than raising when that assumption broke, so it
    stays defensive."""
    tail = path.replace("\\", "/").rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[0] if "." in tail else tail


def _resolve_import_source(raw: str, all_paths: list):
    """Best-effort: map an import string to the file it most likely refers
    to, by matching the module segment's stem against files in the repo."""
    module_part = raw.rsplit(".", 1)[0] if "." in raw else raw
    tail = module_part.replace(".", "/").strip("/").split("/")[-1]
    if not tail:
        return None
    for p in all_paths:
        if _path_stem(p) == tail:
            return p
    return None


# ------------------------------------------------------------ traces ----

def _try_env_var(edge, arg, candidates, rec):
    """Trace 1: environment-variable lookup, inline or via a variable."""
    inline = _INLINE_ENV.search(arg or "")
    if inline:
        key = inline.group(1) or inline.group(2) or inline.group(3)
        edge.literal = key
        edge.literal_method = "joern"
        edge.host = key
        return TraceResult("env_var", ran=True, resolved=True, detail=f"inline env key {key}")

    if not rec:
        return TraceResult("env_var", ran=False, reason="no_source_record")
    if not candidates:
        return TraceResult("env_var", ran=False, reason="no_identifier_to_trace")

    for name in candidates:
        pattern_py = re.compile(_ENV_ASSIGN_TMPL.format(name=re.escape(name)))
        pattern_js = re.compile(_ENV_ASSIGN_JS_TMPL.format(name=re.escape(name)))
        key = None
        for line in rec.source_lines:
            m = pattern_py.match(line)
            if m:
                key = m.group(1)
                continue
            m = pattern_js.match(line)
            if m:
                key = m.group(1) or m.group(2)
        if key:
            edge.literal = key
            edge.literal_method = "joern"
            edge.host = key  # bare env-var key, meant to match a manifest below
            return TraceResult("env_var", ran=True, resolved=True,
                               detail=f"{name} <- env {key}")
    return TraceResult("env_var", ran=True, reason="no_env_assignment_found")


def _try_self_attr(edge, arg, rec):
    """Trace 2: `self.attr` / `this.attr` set in a different method."""
    m = _SELF_ATTR.match(arg or "") or _THIS_ATTR.match(arg or "")
    if not m:
        return TraceResult("self_attr", ran=False, reason="arg_is_not_an_instance_attribute")
    if not rec:
        return TraceResult("self_attr", ran=False, reason="no_source_record")
    attr = m.group(1)
    pattern = re.compile(_SELF_ASSIGN_TMPL.format(name=re.escape(attr)))
    best = _last_assignment(rec.source_lines, pattern)
    if not best:
        return TraceResult("self_attr", ran=True, reason="no_literal_assignment_to_attribute")
    edge.literal = best
    edge.literal_method = "joern"
    edge.host = _host_from_literal(best) or best
    return TraceResult("self_attr", ran=True, resolved=True, detail=f"self.{attr} = {best}")


def _try_imported_constant(edge, candidates, records_by_path, all_paths):
    """Trace 3: module-level constant defined in another file."""
    if not candidates:
        return TraceResult("imported_constant", ran=False, reason="no_identifier_to_trace")
    rec = records_by_path.get(edge.file)
    if not rec:
        return TraceResult("imported_constant", ran=False, reason="no_source_record")
    if not rec.imports:
        return TraceResult("imported_constant", ran=False, reason="file_has_no_imports")

    checked_any = False
    for name in candidates:
        if "." in name:
            continue
        for imp in rec.imports:
            tail = imp.raw.replace(".", "/").strip("/").split("/")[-1]
            if tail != name:
                continue
            checked_any = True
            src_path = _resolve_import_source(imp.raw, all_paths)
            target_rec = records_by_path.get(src_path) if src_path else None
            if not target_rec:
                continue
            pattern = re.compile(_MODULE_ASSIGN_TMPL.format(name=re.escape(name)))
            best = _last_assignment(target_rec.source_lines, pattern)
            if best:
                edge.literal = best
                edge.literal_method = "joern"
                edge.host = _host_from_literal(best) or best
                return TraceResult("imported_constant", ran=True, resolved=True,
                                   detail=f"{name} from {src_path} = {best}")
    if not checked_any:
        return TraceResult("imported_constant", ran=True,
                           reason="no_import_matches_the_traced_identifier")
    return TraceResult("imported_constant", ran=True,
                       reason="import_found_but_no_literal_assignment")


#: A value this trace is willing to call a host. Guessed values reach the
#: graph as service nodes, so the bar is "could plausibly be a network
#: destination": a scheme, a dotted name, or a host:port. A bare word like
#: `email` or `admin` is a wrong-argument pickup, not a host.
_PLAUSIBLE_HOST = re.compile(
    r"""^(?:
          [a-zA-Z][\w+.\-]*://.+                 # has a scheme
        | [A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+.*  # dotted name / domain
        | [A-Za-z0-9\-]+:\d{2,5}(?:/.*)?         # host:port
        | (?:localhost|127\.0\.0\.1)(?::\d+)?.*
        )$""",
    re.VERBOSE,
)


def _is_plausible_host(value: str) -> bool:
    value = (value or "").strip()
    if len(value) < 4 or " " in value:
        return False
    return bool(_PLAUSIBLE_HOST.match(value))


def _parameter_index(caller_qn: str, param_name: str, records_by_path):
    """Position of `param_name` in the signature of the function named by
    `caller_qn`, or None if it can't be determined.

    Without this the trace matched a literal to a parameter by hope: it
    took the first quoted string anywhere in the caller's argument list,
    regardless of which parameter it was actually bound to."""
    file_path = caller_qn.split("::", 1)[0]
    rec = records_by_path.get(file_path)
    if not rec:
        return None
    definition = next((d for d in rec.definitions
                       if d.qualified_name == caller_qn), None)
    if not definition:
        return None
    header = " ".join(
        rec.source_lines[definition.start_line - 1: definition.start_line + 4]
    )
    m = re.search(r"\((.*?)\)", header, re.DOTALL)
    if not m:
        return None
    params = [p.strip() for p in split_top_level(m.group(1))]
    names = []
    for p in params:
        # strip defaults, annotations, *args/**kwargs markers
        name = p.split("=", 1)[0].split(":", 1)[0].strip().lstrip("*").strip()
        if name:
            names.append(name)
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names.index(param_name) if param_name in names else None


def _try_cross_function_param(edge, candidates, reverse_calls, records_by_path):
    """Trace 4: inter-procedural — pull the literal from a caller.

    This is the least reliable of the four and the only one that produces
    a *guess* rather than a trace, so it carries two guards the others
    don't need:

      * positional matching — the literal has to sit at the same index in
        the caller's argument list as the traced name does in the callee's
        signature. Previously any quoted string in the caller's args would
        do, which is how `send_reset(email, ...)` yielded `host="email"`.
      * host-shape validation — the winning value has to look like
        something you could send a request to.

    Anything it does resolve is marked `needs_review`, because a literal
    from one call site says nothing about the other call sites."""
    if not candidates:
        return TraceResult("cross_function_param", ran=False, reason="no_identifier_to_trace")
    endpointish = [n for n in candidates
                   if is_bare_identifier(n) and _looks_like_endpoint_param(n)]
    if not endpointish:
        return TraceResult("cross_function_param", ran=False,
                           reason="identifier_failed_endpoint_name_heuristic")
    callers = reverse_calls.get(edge.caller) or []
    if not callers:
        return TraceResult("cross_function_param", ran=True,
                           reason="enclosing_function_has_no_known_callers")

    rejected_shape = 0
    no_position = False
    for name in endpointish:
        index = _parameter_index(edge.caller, name, records_by_path)
        if index is None:
            no_position = True
            continue
        for call_edge in callers:
            args = split_top_level(call_edge.arg_text or "")
            if index >= len(args):
                continue
            value = string_literal_value(args[index].strip())
            if value is None:
                continue
            host = _host_from_literal(value)
            if not host and not _is_plausible_host(value):
                rejected_shape += 1
                continue
            edge.literal = value
            edge.literal_method = "joern"
            edge.host = host or value
            # Inferred from one call site, not proven for all of them.
            edge.needs_review = True
            edge.review_reason = (
                f"Host taken from a single caller at {call_edge.file}:{call_edge.line}, "
                f"where parameter '{name}' (position {index}) was passed the literal "
                f"'{value}'. Other callers may pass a different value — verify."
            )
            return TraceResult(
                "cross_function_param", ran=True, resolved=True,
                detail=f"{name}@{index} <- literal at {call_edge.file}:{call_edge.line}")

    if rejected_shape:
        return TraceResult("cross_function_param", ran=True,
                           reason="caller_literal_did_not_look_like_a_host")
    if no_position:
        return TraceResult("cross_function_param", ran=True,
                           reason="could_not_locate_parameter_in_callee_signature")
    return TraceResult("cross_function_param", ran=True,
                       reason="no_caller_passed_a_literal_at_that_position")


# ------------------------------------------------------------- stage ----

def check_joern_escalation(edges, records_by_path, all_records, ctx=None):
    """Mutates edges in place. Returns the list of edges that were
    escalated to this stage (whether or not it resolved a host)."""
    all_paths = [r.path for r in all_records if not r.is_config]

    reverse_calls = defaultdict(list)
    for e in edges:
        if e.status == "internal":
            for target_qn in e.resolved_targets:
                reverse_calls[target_qn].append(e)

    escalated = []
    resolved_count = 0
    trace_ran = Counter()
    trace_resolved = Counter()
    bail_reasons = Counter()
    arg_shapes = Counter()

    for edge in edges:
        if edge.status != "external_candidate" or edge.host:
            continue
        edge.escalated_to_joern = True
        escalated.append(edge)

        arg = first_argument(edge.arg_text)
        shape = classify(arg)
        arg_shapes[shape] += 1
        rec = records_by_path.get(edge.file)
        candidates = identifier_roots(arg)

        results = []
        # Run the four traces in order, stopping at the first hit, but
        # recording the outcome of each one that actually ran.
        for factory in (
            lambda: _try_env_var(edge, arg, candidates, rec),
            lambda: _try_self_attr(edge, arg, rec),
            lambda: _try_imported_constant(edge, candidates, records_by_path, all_paths),
            lambda: _try_cross_function_param(edge, candidates, reverse_calls, records_by_path),
        ):
            result = factory()
            results.append(result)
            if result.ran:
                trace_ran[result.name] += 1
            if result.reason:
                bail_reasons[f"{result.name}.{result.reason}"] += 1
            if result.resolved:
                trace_resolved[result.name] += 1
                break

        if any(r.resolved for r in results):
            edge.joern_resolved = True
            resolved_count += 1
            winner = next(r for r in results if r.resolved)
            log(LOG, "info", "escalation resolved",
                file=edge.file, line=edge.line, trace=winner.name,
                host=edge.host, detail=winner.detail)
        else:
            # The important log line. If nothing ran at all, the stage was
            # structurally unable to help rather than simply unlucky.
            ran_any = any(r.ran for r in results)
            log(LOG, "info" if ran_any else "warning",
                "escalation exhausted" if ran_any else "escalation could not run any trace",
                file=edge.file, line=edge.line, callee=edge.callee_expr,
                arg_shape=shape, arg=arg, identifiers=candidates,
                traces=[r.as_dict() for r in results])

    if ctx is not None:
        ctx.bump("joern.escalated", len(escalated))
        ctx.bump("joern.resolved", resolved_count)
        for name, n in trace_ran.items():
            ctx.bump(f"joern.trace_ran.{name}", n)
        for name, n in trace_resolved.items():
            ctx.bump(f"joern.trace_resolved.{name}", n)
        for reason, n in bail_reasons.items():
            ctx.bump(f"joern.bail.{reason}", n)
        for shape, n in arg_shapes.items():
            ctx.bump(f"joern.arg_shape.{shape}", n)

        log(LOG, "info", "joern escalation complete",
            escalated=len(escalated), resolved=resolved_count,
            traces_ran=dict(trace_ran), traces_resolved=dict(trace_resolved),
            arg_shapes=dict(arg_shapes))

        if escalated and not resolved_count:
            _explain_total_failure(ctx, escalated, arg_shapes, trace_ran, bail_reasons)

    return escalated


def _explain_total_failure(ctx, escalated, arg_shapes, trace_ran, bail_reasons):
    """Turn '0 resolved' into an actionable sentence.

    The three cases worth separating:
      * no trace ever ran        -> the input was unusable (upstream problem)
      * traces ran, none matched -> genuinely dynamic values (expected)
      * one dominant bail reason -> a specific gap to close
    """
    opaque = arg_shapes.get("opaque_placeholder", 0)
    if opaque:
        msg = (f"All {len(escalated)} escalated call(s) reached this stage with an "
               f"unrecoverable argument ({opaque} opaque). No trace can run without "
               "argument source text — fix argument extraction in the parser first.")
        level = "error"
    elif not sum(trace_ran.values()):
        msg = (f"None of the four traces were applicable to any of the "
               f"{len(escalated)} escalated call(s). Argument shapes seen: "
               f"{dict(arg_shapes)}. This is an input-shape gap, not a miss.")
        level = "warning"
    else:
        top = ", ".join(f"{r} x{n}" for r, n in bail_reasons.most_common(3))
        msg = (f"All {len(escalated)} escalated call(s) were traced and none "
               f"resolved. Top bail reasons: {top}.")
        level = "info"
    ctx.note(level, "joern", msg)
    log(LOG, level if level != "info" else "info", "escalation yielded nothing", summary=msg)
