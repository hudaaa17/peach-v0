"""
Stage 2: SCIP symbol resolution — decide which calls resolve to a symbol
defined inside the repo. Calls that do not resolve internally are passed on to
the ast-grep pattern-match stage as external-call candidates.

Scope: Python, JavaScript and TypeScript (and their frameworks).

Resolution
----------
SCIP is authoritative. For every call in a file the SCIP index covers
(`scip_check.py`), the call is internal if and only if SCIP resolved the
callee to a definition inside the repo. A call SCIP covers but did not resolve
is NOT internal — it is an external-call candidate (`requests.post`, `fetch`,
`axios.get`, a framework method...). There is deliberately no name-matching
rescue for those: matching by bare name is what used to let a repo function
named `post` swallow every `requests.post` in the codebase.

Fallback — only where SCIP did not cover the file
-------------------------------------------------
A file is "not covered" when SCIP failed for its language (indexer or `scip`
CLI missing, timeout, crash, unreadable index) or the index simply does not
contain the file (excluded by tsconfig, over the indexer's size limit...). For
those files, and only those, a deliberately small same-file fallback runs:

  * `foo(...)`        -> `foo` defined in the same file (function or class)
  * `self.x(...)`, `cls.x(...)`, `this.x(...)`
                      -> method `x` of the class the calling method belongs to

Anything else in an uncovered file (calls to imported functions, calls on
other objects, cross-file calls) stays unresolved: when we cannot know, we say
so instead of guessing, and the call proceeds to the external-call stage like
any other unresolved call.

`edge.resolved_via` records which path produced an internal edge:
"scip" or "lexical_fallback".

internal_unmapped
-----------------
SCIP can say "this call goes to code in the repo" for a definition the parser
has no record of: the defining file was skipped (too large, minified, parse
exception), was past the file cap, or the construct is one the parser does not
record. Such a call is INTERNAL; it has no graph node to point at, and it must
NOT be handed to the external-call stages as if it were `requests.post`. Its
status is "internal_unmapped" and `edge.unmapped_def_files` names where it
goes. (A definition inside a vendored directory such as node_modules/ is
third-party code, so a call resolving there stays an ordinary unresolved /
external candidate.)

Gaps
----
Everything this stage could not do is reported as structured `Gap` entries
(gaps.py): a language SCIP failed for, a file the index did not cover, a run
where SCIP was not attempted, and each file that `internal_unmapped` calls
point into. Pass `gaps=[]` (or rely on `ctx.gaps`) and hand the list to the UI.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .gaps import Gap, emit, gap_sink
from .obs import get_logger, log, TRACE_EDGES
from .parser import SKIP_DIRS
from .scip_check import build_reference_map, _LANGUAGE_TO_INDEXER, ScipResult

LOG = get_logger("resolver")

# Receivers that mean "the object this method belongs to".
_SELF_RECEIVERS = ("self", "cls", "this")


@dataclass
class CallEdge:
    caller: str
    callee_expr: str
    file: str
    line: int
    arg_text: str
    resolved_targets: list = field(default_factory=list)   # qualified names, if internal
    status: str = "unresolved"                              # internal | internal_unmapped | external_candidate | unresolved
    resolved_via: Optional[str] = None                       # "scip" | "lexical_fallback", when status == "internal" (or "scip" for internal_unmapped)
    unmapped_def_files: list = field(default_factory=list)   # internal_unmapped: repo files SCIP says the callee is defined in
    external_pattern: Optional[str] = None
    external_via: Optional[str] = None                       # "ast_grep" | "regex_fallback", when status == "external_candidate"
    literal: Optional[str] = None
    literal_method: Optional[str] = None                    # "literal" | "const_prop" | "joern" | "llm_inferred"
    constprop_via: Optional[str] = None                      # "ast" | "regex_fallback", when literal_method == "const_prop"
    host: Optional[str] = None
    config_matches: list = field(default_factory=list)
    # --- stage 7/8 (Joern escalation / SLM fallback) bookkeeping ---
    escalated_to_joern: bool = False   # sent to the scoped cross-function trace stage
    joern_resolved: bool = False       # that trace found a host
    sent_to_llm: bool = False          # sent to the SLM fallback stage
    needs_review: bool = False         # result is inferred/heuristic, not proven — flag for a human
    review_reason: Optional[str] = None


def build_definition_location_index(records):
    """(file, name_line, name_col) -> qualified_name. SCIP reports where a
    symbol is *declared* (its identifier); this translates that position back
    into our own node id, using the Definition records parser.py produced."""
    index = {}
    for rec in records:
        for d in rec.definitions:
            if d.name_col >= 0:
                index[(rec.path, d.name_line, d.name_col)] = d.qualified_name
    return index


def resolve_calls(records, ctx=None, repo_root=None, gaps=None,
                  scip: Optional[ScipResult] = None) -> list:
    """Resolve every call. Returns a list of CallEdge (one per CallSite).

    `repo_root` is required for SCIP to run. If it is omitted, SCIP is not
    attempted and every file is handled by the same-file fallback. This
    function does not raise because SCIP is unavailable or fails.

    `scip` is an already-built `ScipResult` (from `scip_check.run_scip`). When
    given, its `ref_map` / `status` are used as they are and SCIP is not run
    again, so one SCIP run per job can feed this stage and others; `repo_root`
    is then not needed for SCIP. When None (the default) SCIP runs here exactly
    as before.

    What could not be analysed is appended to `gaps` (or `ctx.gaps`) as `Gap`
    entries; the return value is unchanged. The same gaps are emitted whether
    SCIP ran here or was passed in."""
    sink = gap_sink(ctx, gaps)
    if scip is not None:
        scip_refs, scip_status = scip.ref_map, scip.status
    else:
        scip_refs, scip_status = _scip_references(records, repo_root, ctx)
    covered_files = scip_status.get("covered_files", set())
    def_by_location = build_definition_location_index(records)
    defined = {d.qualified_name for rec in records for d in rec.definitions}

    edges = []
    counts = Counter()
    unresolved_heads = Counter()
    unmapped_by_file = Counter()     # defining file -> number of internal_unmapped calls into it

    for rec in records:
        for call in rec.calls:
            edge = CallEdge(
                caller=call.caller, callee_expr=call.callee_expr, file=call.file,
                line=call.line, arg_text=call.arg_text,
            )
            unmapped_files = []
            if call.file in covered_files:
                counts["covered"] += 1
                targets, unmapped_files = _scip_targets(call, scip_refs, def_by_location, counts)
                via = "scip"
            else:
                counts["fallback"] += 1
                targets = _lexical_targets(call, defined)
                via = "lexical_fallback"

            if targets:
                edge.resolved_targets = targets
                edge.status = "internal"
                edge.resolved_via = via
                counts[f"internal_{via}"] += 1
                if TRACE_EDGES:
                    log(LOG, "debug", "internal edge", callee=call.callee_expr, file=call.file,
                        line=call.line, via=via, targets=targets[:3])
            elif unmapped_files:
                # SCIP: internal. Parser: no node for it. Not an external call.
                edge.status = "internal_unmapped"
                edge.resolved_via = "scip"
                edge.unmapped_def_files = unmapped_files
                counts["internal_unmapped"] += 1
                for path in unmapped_files:
                    unmapped_by_file[path] += 1
            else:
                unresolved_heads[call.callee_expr.split(".")[0]] += 1
            edges.append(edge)

    _emit_gaps(sink, records, repo_root, scip_status, covered_files, unmapped_by_file,
               scip_given=scip is not None)
    _report(ctx, edges, counts, scip_status, unresolved_heads)
    return edges


def _scip_references(records, repo_root, ctx):
    """Run SCIP. Never raises; on any problem returns an empty map and a status
    with no covered files, which sends every call to the fallback."""
    empty = {"indexed": [], "skipped": {}, "covered_files": set(), "scip_print_available": False}
    if repo_root is None:
        log(LOG, "warning", "repo_root not provided; SCIP not attempted, using same-file fallback")
        return {}, empty
    try:
        return build_reference_map(records, repo_root, ctx=ctx)
    except Exception as exc:  # noqa: BLE001 - scip_check already guards; this is belt and braces
        log(LOG, "error", "SCIP stage raised; using same-file fallback for all calls",
            error=f"{type(exc).__name__}: {exc}")
        return {}, empty


def _is_vendored(path):
    """True if a repo-relative POSIX path lies inside a vendored/build directory."""
    return any(part in SKIP_DIRS for part in path.split("/"))


def _scip_targets(call, scip_refs, def_by_location, counts):
    """(targets, unmapped_files) for one call.

    targets: qualified names SCIP resolved this call to, or [].
    unmapped_files: only when targets is empty - the repo files SCIP says the
    callee is defined in although the parser recorded no Definition there.
    Vendored locations (node_modules/, venv/ ...) are third-party code and are
    left out, so a call resolving only there stays an ordinary external
    candidate.

    Looks up the exact position of the callee identifier, so another call on
    the same line can never be mistaken for this one."""
    locations = scip_refs.get((call.file, call.callee_line, call.callee_col))
    if not locations:
        return [], []
    targets = []
    for location in locations:
        qualified = def_by_location.get(location)
        if qualified and qualified not in targets:
            targets.append(qualified)
    if targets:
        return targets, []
    # SCIP says the callee is defined in the repo, but parser.py has no
    # Definition at that spot (defining file skipped by the parser, beyond the
    # file cap, a property assigned a function, a dynamic construct...). Don't
    # invent a target - and don't call it external either.
    counts["scip_unmapped_definition"] += 1
    unmapped = sorted({loc[0] for loc in locations if not _is_vendored(loc[0])})
    return [], unmapped


def _lexical_targets(call, defined):
    """Same-file fallback for files SCIP did not cover. See module docstring."""
    parts = call.callee_expr.split(".")
    if len(parts) == 1:
        qualified = f"{call.file}::{parts[0]}"
    elif len(parts) == 2 and parts[0] in _SELF_RECEIVERS:
        owner = _enclosing_class(call.caller)
        if owner is None:
            return []
        qualified = f"{call.file}::{owner}.{parts[1]}"
    else:
        return []
    return [qualified] if qualified in defined else []


def _enclosing_class(caller):
    """Class a calling method belongs to. A method's qualified name is
    "<file>::<Class>.<method>"; a plain function is "<file>::<name>" and
    module-level code is "<file>::<module>" — neither has a class."""
    tail = caller.partition("::")[2]
    if "." in tail and not tail.startswith("<"):
        return tail.rsplit(".", 1)[0]
    return None


def _emit_gaps(sink, records, repo_root, scip_status, covered_files, unmapped_by_file,
               scip_given=False):
    """Turn what SCIP could not do into structured Gap entries. This is the single
    emitter of SCIP gaps. `scip_given` means a prebuilt ScipResult was supplied, so
    SCIP did run and a missing `repo_root` is not a "not attempted" gap."""
    if sink is None:
        return
    indexable = [r for r in records
                 if not r.skipped and not r.is_config and r.language in _LANGUAGE_TO_INDEXER]
    if not indexable:
        return

    if repo_root is None and not scip_given:
        emit(sink, Gap(
            stage="scip", scope="repo", kind="scip_not_attempted", count=len(indexable),
            detail="SCIP was not run because no repo_root was given.",
            impact=("Only same-file and self/this calls were resolved. Cross-file internal calls "
                    "are reported as unresolved and may be misreported as external calls.")))
        return

    failed = scip_status.get("skipped", {})
    for lang, reason in sorted(failed.items()):
        n = sum(1 for r in indexable if r.language == lang)
        emit(sink, Gap(
            stage="scip", scope="language", kind="scip_language_failed", count=n,
            detail=f"SCIP could not index {lang}: {reason}",
            impact=(f"{n} {lang} file(s) were resolved with the same-file fallback only. Cross-file "
                    "internal calls in them are reported as unresolved and may be misreported as "
                    "external calls.")))

    for rec in indexable:
        if rec.language in failed or rec.path in covered_files:
            continue
        emit(sink, Gap(
            stage="scip", scope="file", kind="scip_not_covered", path=rec.path,
            detail=("SCIP ran but its index does not contain this file (for example excluded by "
                    "tsconfig, or over the indexer's size limit)."),
            impact=("Only same-file and self/this calls in it were resolved. Cross-file internal "
                    "calls from it are reported as unresolved and may be misreported as "
                    "external calls.")))

    skipped_by_path = {r.path: r for r in records if r.skipped}
    for path, n in sorted(unmapped_by_file.items()):
        rec = skipped_by_path.get(path)
        if rec is not None:
            why = f"the parser skipped that file ({rec.skip_reason})"
        else:
            why = ("the parser recorded no definition at that position (file past the file cap, "
                   "or a construct the parser does not record)")
        emit(sink, Gap(
            stage="scip", scope="file", kind="internal_unmapped", path=path, count=n,
            detail=f"{n} call(s) resolve (per SCIP) to code in this file, but {why}.",
            impact=("These calls are internal and are NOT treated as external API calls, but "
                    "there is no graph node for their target.")))


def _report(ctx, edges, counts, scip_status, unresolved_heads):
    internal = counts["internal_scip"] + counts["internal_lexical_fallback"]
    unmapped = counts["internal_unmapped"]
    if ctx is not None:
        ctx.bump("resolve.edges", len(edges))
        ctx.bump("resolve.internal", internal)
        ctx.bump("resolve.internal.via_scip", counts["internal_scip"])
        ctx.bump("resolve.internal.via_lexical_fallback", counts["internal_lexical_fallback"])
        ctx.bump("resolve.internal_unmapped", unmapped)
        ctx.bump("resolve.unresolved", len(edges) - internal - unmapped)
        ctx.bump("resolve.scip.calls_covered", counts["covered"])
        ctx.bump("resolve.fallback.calls", counts["fallback"])
        ctx.bump("resolve.scip.unmapped_definition", counts["scip_unmapped_definition"])
        if counts["fallback"]:
            ctx.note("warning", "resolve",
                     f"{counts['fallback']} of {len(edges)} call(s) are in files SCIP did not "
                     "cover; only same-file and self/this calls were resolved for them, so "
                     "cross-file internal calls there are reported as unresolved.")
    log(LOG, "info", "symbol resolution complete",
        edges=len(edges), internal=internal,
        via_scip=counts["internal_scip"], via_lexical_fallback=counts["internal_lexical_fallback"],
        internal_unmapped=unmapped,
        unresolved=len(edges) - internal - unmapped,
        calls_in_scip_covered_files=counts["covered"], calls_in_fallback_files=counts["fallback"],
        scip_unmapped_definition=counts["scip_unmapped_definition"],
        scip_indexed_languages=scip_status.get("indexed", []),
        scip_skipped_languages=list(scip_status.get("skipped", {}).keys()),
        top_unresolved_roots=dict(unresolved_heads.most_common(8)))