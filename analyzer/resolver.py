"""
Stage 2: SCIP symbol resolution — confirm which calls resolve to a symbol
defined inside the repo. Calls that don't resolve internally are passed on
to the ast-grep pattern-match stage as external-call candidates.

Resolution is tried in two tiers per call site:

  1. Real SCIP resolution (`scip_check.py`) — an actual SCIP indexer's
     output for that call's language, if one is installed and indexing
     succeeded. This is scope- and type-aware, the way the "SCIP symbol
     resolution" stage name implies: it comes from the language's own
     tooling (pyright, tsserver, ...), not from matching identifiers.
  2. The bare tail-name heuristic — match the call's last dotted segment
     against every definition in the repo with that name, preferring a
     same-file match. This is a real fallback, not a placeholder: several
     of the languages this pipeline parses (Ruby, PHP, C#) have no
     mainstream SCIP indexer at all, and it's also what covers a call
     when SCIP tooling for its language isn't installed in this
     environment. It has a known failure mode — a repo-local function
     named e.g. `post` will capture `requests.post` as "internal" purely
     because the tail names match — which is exactly the imprecision tier
     1 exists to avoid whenever it's available.

`edge.resolved_via` records which tier actually resolved a given internal
edge, so that distinction is visible in the graph/meta rather than
collapsed into one undifferentiated "internal" bucket.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .obs import get_logger, log, TRACE_EDGES
from .scip_check import build_reference_map

LOG = get_logger("resolver")


@dataclass
class CallEdge:
    caller: str
    callee_expr: str
    file: str
    line: int
    arg_text: str
    resolved_targets: list = field(default_factory=list)   # qualified names, if internal
    status: str = "unresolved"                              # internal | external_candidate | unresolved
    resolved_via: Optional[str] = None                       # "scip" | "tail_name", when status == "internal"
    external_pattern: Optional[str] = None
    literal: Optional[str] = None
    literal_method: Optional[str] = None                    # "literal" | "const_prop" | "joern" | "llm_inferred"
    host: Optional[str] = None
    config_matches: list = field(default_factory=list)
    # --- stage 7/8 (Joern escalation / SLM fallback) bookkeeping ---
    escalated_to_joern: bool = False   # sent to the scoped cross-function trace stage
    joern_resolved: bool = False       # that trace found a host
    sent_to_llm: bool = False          # sent to the SLM fallback stage
    needs_review: bool = False         # result is inferred/heuristic, not proven — flag for a human
    review_reason: Optional[str] = None


def build_symbol_index(records):
    """name -> list of (file, qualified_name) definitions across the repo.
    Backs the bare tail-name fallback (tier 2)."""
    index = {}
    for rec in records:
        for d in rec.definitions:
            index.setdefault(d.name, []).append((rec.path, d.qualified_name))
    return index


def build_definition_location_index(records):
    """(file, start_line) -> qualified_name. Backs tier 1: SCIP gives us a
    *location* a call resolves to, and this is how that location is
    translated back into our own graph node id scheme, keyed off the
    same Definition records parser.py already produced."""
    index = {}
    for rec in records:
        for d in rec.definitions:
            index[(rec.path, d.start_line)] = d.qualified_name
    return index


def resolve_calls(records, ctx=None, repo_root=None) -> list:
    """Two-tier resolution pass (real SCIP, then bare tail-name). Returns
    a list of CallEdge. `repo_root` is optional — pass it to enable real
    SCIP resolution; omit it (or leave SCIP tooling uninstalled) and every
    call falls back to tier 2, same as before this stage had a real SCIP
    integration."""
    symbol_index = build_symbol_index(records)
    edges = []
    unresolved_heads = Counter()
    cross_file_ambiguous = 0
    scip_resolved = 0
    tail_resolved = 0

    scip_ref_map, scip_status = {}, {"indexed": [], "skipped": {}}
    def_by_location = {}
    if repo_root is not None:
        scip_ref_map, scip_status = build_reference_map(records, repo_root, ctx=ctx)
        if scip_ref_map:
            def_by_location = build_definition_location_index(records)

    for rec in records:
        for call in rec.calls:
            tail = call.callee_expr.split(".")[-1]
            edge = CallEdge(
                caller=call.caller, callee_expr=call.callee_expr, file=call.file,
                line=call.line, arg_text=call.arg_text,
            )

            # --- tier 1: real SCIP resolution ---
            scip_def_loc = scip_ref_map.get((call.file, call.line))
            scip_qn = def_by_location.get(scip_def_loc) if scip_def_loc else None
            if scip_qn:
                edge.resolved_targets = [scip_qn]
                edge.status = "internal"
                edge.resolved_via = "scip"
                scip_resolved += 1
                edges.append(edge)
                continue

            # --- tier 2: bare tail-name fallback ---
            candidates = symbol_index.get(tail, [])
            if candidates:
                # prefer a definition in the same file, else take all matches
                same_file = [qn for f, qn in candidates if f == rec.path]
                edge.resolved_targets = same_file if same_file else [qn for _, qn in candidates]
                edge.status = "internal"
                edge.resolved_via = "tail_name"
                tail_resolved += 1
                if not same_file and len(candidates) > 1:
                    cross_file_ambiguous += 1
                    if TRACE_EDGES:
                        log(LOG, "debug", "ambiguous internal resolution",
                            callee=call.callee_expr, file=call.file, line=call.line,
                            candidates=len(candidates))
                # Resolution is by *bare tail name*, so a repo-local
                # function named e.g. `post` or `get` will capture
                # `requests.post` as internal and hide it from the
                # external-call stage entirely. Worth seeing when it happens.
                if "." in call.callee_expr:
                    _bump(ctx, "resolve.internal.matched_on_tail_of_dotted")
                    if TRACE_EDGES:
                        log(LOG, "debug", "dotted call resolved internally by tail name",
                            callee=call.callee_expr, tail=tail,
                            targets=edge.resolved_targets[:3])
            else:
                unresolved_heads[call.callee_expr.split(".")[0]] += 1
            edges.append(edge)

    if ctx is not None:
        internal = sum(1 for e in edges if e.status == "internal")
        ctx.bump("resolve.edges", len(edges))
        ctx.bump("resolve.internal", internal)
        ctx.bump("resolve.internal.via_scip", scip_resolved)
        ctx.bump("resolve.internal.via_tail_name", tail_resolved)
        ctx.bump("resolve.unresolved", len(edges) - internal)
        ctx.bump("resolve.ambiguous_cross_file", cross_file_ambiguous)
        log(LOG, "info", "symbol resolution complete",
            symbols=len(symbol_index), edges=len(edges),
            internal=internal, via_scip=scip_resolved, via_tail_name=tail_resolved,
            unresolved=len(edges) - internal, ambiguous=cross_file_ambiguous,
            scip_indexed_languages=scip_status.get("indexed", []),
            scip_skipped_languages=list(scip_status.get("skipped", {}).keys()),
            top_unresolved_roots=dict(unresolved_heads.most_common(8)))
    return edges


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)