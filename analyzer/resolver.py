"""
Stage 2: SCIP symbol resolution — confirm which calls resolve to a symbol
defined inside the repo. Calls that don't resolve internally are passed on
to the ast-grep pattern-match stage as external-call candidates.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .obs import get_logger, log, TRACE_EDGES

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
    """name -> list of (file, qualified_name) definitions across the repo."""
    index = {}
    for rec in records:
        for d in rec.definitions:
            index.setdefault(d.name, []).append((rec.path, d.qualified_name))
    return index


def resolve_calls(records, ctx=None) -> list:
    """SCIP-style resolution pass. Returns a list of CallEdge."""
    symbol_index = build_symbol_index(records)
    edges = []
    unresolved_heads = Counter()
    cross_file_ambiguous = 0

    for rec in records:
        for call in rec.calls:
            tail = call.callee_expr.split(".")[-1]
            edge = CallEdge(
                caller=call.caller, callee_expr=call.callee_expr, file=call.file,
                line=call.line, arg_text=call.arg_text,
            )
            candidates = symbol_index.get(tail, [])
            if candidates:
                # prefer a definition in the same file, else take all matches
                same_file = [qn for f, qn in candidates if f == rec.path]
                edge.resolved_targets = same_file if same_file else [qn for _, qn in candidates]
                edge.status = "internal"
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
        ctx.bump("resolve.unresolved", len(edges) - internal)
        ctx.bump("resolve.ambiguous_cross_file", cross_file_ambiguous)
        log(LOG, "info", "symbol resolution complete",
            symbols=len(symbol_index), edges=len(edges),
            internal=internal, unresolved=len(edges) - internal,
            ambiguous=cross_file_ambiguous,
            top_unresolved_roots=dict(unresolved_heads.most_common(8)))
    return edges


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)
