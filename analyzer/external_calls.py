"""
Stage 3: ast-grep pattern match — flag calls that hit a known external-client
API shape (HTTP clients, socket clients, etc). Only calls that the SCIP
resolution stage left unresolved are candidates: anything that already
resolved to a symbol inside the repo is internal, not external.

Matching is tried in two tiers per unresolved call, mirroring the tier
structure resolver.py uses for symbol resolution:

  1. Real ast-grep pattern matching (`ast_grep_check.py`) — an actual
     structural search over the language's own parse tree, if `ast-grep`
     is installed and a rule set exists for the call's language. This is
     what the "ast-grep pattern match" stage name has always implied: a
     match against real call syntax, not text.
  2. The `re`-based catalogue below (`EXTERNAL_PATTERNS`) — matched
     against the already-flattened `callee_expr` string. This is a real
     fallback, not a placeholder: it's what covers a call when `ast-grep`
     isn't installed in this environment, or when the call's language has
     no rule set yet. It has known failure modes — an aliased import
     (`import requests as r; r.get(...)`) or a call ast-grep just wasn't
     asked about won't match — which is exactly the imprecision tier 1
     exists to avoid whenever it's available.

`edge.external_via` records which tier actually matched a given external
candidate, so that distinction is visible in the graph/meta rather than
collapsed into one undifferentiated "external_candidate" bucket.
"""
import re
from collections import Counter

from .ast_grep_check import build_external_match_map
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("external")

# Tier-2 fallback catalogue: (pattern over the dotted call expression,
# human label). Kept in sync with EXTERNAL_RULES in ast_grep_check.py so
# coverage doesn't silently shrink when a call falls back to this tier.
EXTERNAL_PATTERNS = [
    (re.compile(r'^requests\.(get|post|put|delete|patch|head|request)$'), "python-requests"),
    (re.compile(r'^requests\.Session$'), "python-requests"),
    (re.compile(r'^httpx\.(get|post|put|delete|patch|request|Client|AsyncClient)$'), "python-httpx"),
    (re.compile(r'^urllib\.request\.urlopen$'), "python-urllib"),
    (re.compile(r'^http\.client\.HTTPS?Connection$'), "python-http.client"),
    (re.compile(r'^aiohttp\.ClientSession$'), "python-aiohttp"),
    (re.compile(r'^boto3\.client$'), "aws-sdk"),
    (re.compile(r'^fetch$'), "js-fetch"),
    (re.compile(r'^axios\.(get|post|put|delete|patch|request)$'), "js-axios"),
    (re.compile(r'^axios$'), "js-axios"),
    (re.compile(r'^\$\.ajax$'), "js-jquery"),
    (re.compile(r'^(http|https)\.request$'), "node-http"),
    (re.compile(r'^(http|https)\.get$'), "node-http"),
    (re.compile(r'^superagent\.(get|post|put|delete)$'), "js-superagent"),
    (re.compile(r'^grpc\.Client$'), "grpc-client"),
    (re.compile(r'^(HttpClient|RestTemplate|OkHttpClient)$'), "java-http-client"),
    (re.compile(r'^http\.(Get|Post|NewRequest)$'), "go-net-http"),
    (re.compile(r'^resty\.New$'), "go-resty"),
    (re.compile(r'^(Net::HTTP)$'), "ruby-net-http"),
]


# Call roots that are very likely to be network/IO clients but have no
# entry in EXTERNAL_PATTERNS above. Purely for reporting: these are *not*
# flagged as external, they're just counted so a suspiciously low
# "external calls found" number points at the catalogue rather than at
# the repo. Only checked for calls that actually fell to tier 2, since
# tier 1 already gave those a real, trustworthy answer.
_UNCOVERED_HINTS = (
    "smtplib", "smtp", "firestore", "firebase_admin", "boto3", "psycopg2",
    "pymongo", "redis", "kafka", "sqlalchemy", "elasticsearch", "grpc",
    "websocket", "websockets", "ftplib", "paramiko", "urllib3", "httplib2",
    "google.cloud", "azure", "stripe", "twilio", "sendgrid", "openai",
)


def flag_external_calls(edges, records=None, ctx=None, repo_root=None) -> list:
    """Mutates edges in place: unresolved calls matching a known client
    shape become 'external_candidate'. Returns the list for convenience.

    `records` and `repo_root` are optional — pass both to enable real
    ast-grep matching (tier 1). Omit either (or leave `ast-grep`
    uninstalled) and every call falls back to tier 2, same as before
    this stage had a real ast-grep integration."""
    matched_by_label = Counter()
    matched_by_tier = Counter()
    uncovered = Counter()

    match_map, astgrep_status = {}, {"scanned": [], "skipped": {}}
    if records is not None and repo_root is not None:
        match_map, astgrep_status = build_external_match_map(records, repo_root, ctx=ctx)

    for edge in edges:
        if edge.status == "internal":
            continue

        # --- tier 1: real ast-grep pattern match ---
        label = match_map.get((edge.file, edge.line))
        if label:
            edge.status = "external_candidate"
            edge.external_pattern = label
            edge.external_via = "ast_grep"
            matched_by_label[label] += 1
            matched_by_tier["ast_grep"] += 1
            if TRACE_EDGES:
                log(LOG, "debug", "external candidate (ast-grep)", callee=edge.callee_expr,
                    pattern=label, file=edge.file, line=edge.line)
            continue

        # --- tier 2: re-based catalogue fallback ---
        hit = False
        for pattern, fallback_label in EXTERNAL_PATTERNS:
            if pattern.match(edge.callee_expr):
                edge.status = "external_candidate"
                edge.external_pattern = fallback_label
                edge.external_via = "regex_fallback"
                matched_by_label[fallback_label] += 1
                matched_by_tier["regex_fallback"] += 1
                hit = True
                if TRACE_EDGES:
                    log(LOG, "debug", "external candidate (regex fallback)", callee=edge.callee_expr,
                        pattern=fallback_label, file=edge.file, line=edge.line)
                break
        if not hit:
            lowered = edge.callee_expr.lower()
            if any(h in lowered for h in _UNCOVERED_HINTS):
                uncovered[edge.callee_expr] += 1

    if ctx is not None:
        total = sum(matched_by_label.values())
        ctx.bump("external.candidates", total)
        ctx.bump("external.candidates.via_ast_grep", matched_by_tier["ast_grep"])
        ctx.bump("external.candidates.via_regex_fallback", matched_by_tier["regex_fallback"])
        for label, n in matched_by_label.items():
            ctx.bump(f"external.pattern.{label}", n)
        log(LOG, "info", "external-call pattern match complete",
            candidates=total, by_pattern=dict(matched_by_label),
            via_ast_grep=matched_by_tier["ast_grep"],
            via_regex_fallback=matched_by_tier["regex_fallback"],
            astgrep_scanned_languages=astgrep_status.get("scanned", []),
            astgrep_skipped_languages=list(astgrep_status.get("skipped", {}).keys()))
        if uncovered:
            ctx.bump("external.uncovered_client_calls", sum(uncovered.values()))
            log(LOG, "warning",
                "calls that look like external clients matched no pattern",
                examples=dict(uncovered.most_common(8)))
            ctx.note("info", "external",
                     "Some calls look like network/IO clients but aren't in "
                     "EXTERNAL_PATTERNS, so they never became external candidates: "
                     + ", ".join(sorted(uncovered)[:8]))
    return edges