"""
Stage 3: ast-grep pattern match — flag calls that hit a known external-client
API shape (HTTP clients, socket clients, etc). Only calls that the SCIP
resolution stage left unresolved are candidates: anything that already
resolved to a symbol inside the repo is internal, not external.
"""
import re
from collections import Counter

from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("external")

# (pattern over the dotted call expression, human label)
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
# the repo.
_UNCOVERED_HINTS = (
    "smtplib", "smtp", "firestore", "firebase_admin", "boto3", "psycopg2",
    "pymongo", "redis", "kafka", "sqlalchemy", "elasticsearch", "grpc",
    "websocket", "websockets", "ftplib", "paramiko", "urllib3", "httplib2",
    "google.cloud", "azure", "stripe", "twilio", "sendgrid", "openai",
)


def flag_external_calls(edges, ctx=None):
    """Mutates edges in place: unresolved calls matching a known client
    pattern become 'external_candidate'. Returns the list for convenience."""
    matched_by_label = Counter()
    uncovered = Counter()

    for edge in edges:
        if edge.status == "internal":
            continue
        hit = False
        for pattern, label in EXTERNAL_PATTERNS:
            if pattern.match(edge.callee_expr):
                edge.status = "external_candidate"
                edge.external_pattern = label
                matched_by_label[label] += 1
                hit = True
                if TRACE_EDGES:
                    log(LOG, "debug", "external candidate", callee=edge.callee_expr,
                        pattern=label, file=edge.file, line=edge.line)
                break
        if not hit:
            lowered = edge.callee_expr.lower()
            if any(h in lowered for h in _UNCOVERED_HINTS):
                uncovered[edge.callee_expr] += 1

    if ctx is not None:
        total = sum(matched_by_label.values())
        ctx.bump("external.candidates", total)
        for label, n in matched_by_label.items():
            ctx.bump(f"external.pattern.{label}", n)
        log(LOG, "info", "external-call pattern match complete",
            candidates=total, by_pattern=dict(matched_by_label))
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
