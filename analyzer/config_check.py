"""
Stage 6: Config check — cross-reference the host resolved for each
external-call candidate against the repo's config files (yaml/json/toml/
.env/.properties/.ini/Dockerfiles/k8s manifests), so an external service
can be linked to the deployment config that declares it.

The old version did this by lowercasing the whole file and asking
`needle in text`. That is a text search, not a config check, and it had
three failure modes that made the resulting links untrustworthy:

1. The service-label needle. `host.split(".")[0]` turns
   `api.example.com` into the needle `api` — and `apiVersion:` appears
   on the first line of *every* k8s manifest ever written, so that edge
   "matched" every manifest in the repo. Same for hosts starting `db`,
   `auth`, `service`, `config`. The single most common host shape in a
   real codebase produced the single least informative match.
2. Comments and keys counted as live config. `# old host:
   payments.internal` matched exactly like a real `PAYMENTS_HOST=`
   assignment, and a *key* named `stripe_api_key` matched the host
   `stripe.com`'s label. A config link is a claim about what the
   deployment is configured to talk to; a key name and a commented-out
   line are neither.
3. The port. `needles = {host}` kept `api.example.com:8443` intact, so a
   manifest that (correctly) splits `host:` and `port:` into two fields
   never matched at all. The one place the port *was* stripped
   (`_LABEL_RE` on `host.split(":")[0]`) fed only the label needle from
   (1).

Matching is now tried in two tiers per config file, the same shape the
resolver / external / literal / const-prop stages use:

  1. A real parse of the file in its own format — `yaml.compose_all`
     (which carries source line marks, so every match reports the line
     and key path it came from), `json.loads`, `tomllib`, a real
     key/value pass for `.env`/`.properties`/`.ini`, and a directive
     pass for Dockerfiles. Every scalar *value* is then read as a value:
     a URL is parsed with `urlsplit` and compared on its netloc, a
     `host:port` is split, a bare hostname is only accepted under a
     host-ish key. Keys are never matched against, comments don't exist
     in a parsed document, and both sides are normalised (scheme,
     userinfo, port, trailing dot, case) before comparison — so
     `https://api.example.com:8443/v1` in code and `api.example.com`
     under `spec.rules[0].host` are recognised as the same host, which
     the substring scan could not do in either direction.

     Tier 1 also does the thing the label needle was *trying* to do,
     properly: it collects service names that the config actually
     declares — `services:` keys in a compose file, `metadata.name` of a
     k8s Service/StatefulSet/Ingress — expands them into the DNS names
     they really resolve to (`payments`, `payments.default`,
     `payments.default.svc.cluster.local`) and matches those. A host
     whose first label equals a declared service name is still reported,
     but as a separate lower-confidence kind rather than as an equal.
     `apiVersion` is not a service name, so it never matches anything.

  2. The line scan (`_text_fallback_hits`) — for a config file this
     pipeline has no parser for (an unknown extension caught by
     CONFIG_NAME_HINTS, a Helm template full of `{{ }}` that isn't valid
     YAML, a malformed file, or YAML at all when PyYAML isn't
     installed). It is the old behaviour, hardened: whole-line and
     inline comments are skipped, only the value side of a `key: value`
     line is searched, the match must be on token boundaries so
     `api.example.com` doesn't match inside `staging.api.example.com`,
     and the service-label needle is gone — it needs structure to be
     meaningful, and tier 2 has none.

Tier 2 runs only when tier 1 could not read the file at all, not when it
read it and found nothing: a file that parsed has already been inspected
completely, and re-scanning it as text would just reintroduce (2).

Each match carries `kind`, `confidence`, `key_path` and `line`, so a
human reviewing the graph can see *why* a service was linked to a config
file rather than having to take the link on faith.
"""
import json
import re
from pathlib import PurePosixPath
from typing import NamedTuple
from urllib.parse import urlsplit

from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("config")

try:  # optional: the only format here that needs a third-party parser
    import yaml
    _YAML_UNAVAILABLE = None
except Exception as exc:  # noqa: BLE001 - any import failure degrades the same way
    yaml = None
    _YAML_UNAVAILABLE = f"PyYAML not importable ({type(exc).__name__}: {exc})"

try:
    import tomllib as _toml
    _TOML_UNAVAILABLE = None
except ModuleNotFoundError:
    try:
        import tomli as _toml
        _TOML_UNAVAILABLE = None
    except ModuleNotFoundError:
        _toml = None
        _TOML_UNAVAILABLE = "neither tomllib (Python 3.11+) nor tomli is installed"


# ------------------------------------------------------------ host parsing ---

_URL_RE = re.compile(r'\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s"\'`,;<>()\[\]{}]+')
# A dotted hostname or an IPv4 literal. Single-label names (`redis`,
# `payments`) are deliberately *not* here: they're only accepted when the
# key they sit under says they're a host — see `_hosts_in_value`.
_DOTTED_HOST_RE = re.compile(
    r'^(?=.{1,253}$)(?:[A-Za-z0-9_](?:[A-Za-z0-9_\-]{0,61}[A-Za-z0-9_])?\.)+'
    r'[A-Za-z][A-Za-z0-9\-]{0,61}$'
)
_IPV4_RE = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
_SINGLE_LABEL_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_\-]{2,62}$')
# Keys whose value is a hostname even when it has no dots and no scheme.
_HOSTISH_KEY_RE = re.compile(
    r'(?:^|[._\-])(host|hostname|server|endpoint|url|uri|addr|address|domain|'
    r'upstream|target|backend|broker|bootstrap_servers)s?$', re.I
)
_PLACEHOLDER_RE = re.compile(r'[{}$<>%]')

_WHOLE_LINE_COMMENT = re.compile(r'^\s*(#|//|;|--)')
_INLINE_COMMENT = re.compile(r'\s+(?:#|//|;)\s')
_KEYVAL_RE = re.compile(r'^\s*(?:-\s*)?(?:export\s+)?["\']?([A-Za-z_][\w.\-]*)["\']?\s*[:=]\s*(.*)$')


def _normalize_host(text):
    """`https://user:pw@API.Example.com:8443/v1` -> `api.example.com`.

    Returns None for anything that isn't a usable static host — empty,
    whitespace-bearing, or still carrying an unexpanded placeholder
    (`${API_HOST}`, `{{ .Values.host }}`, or the const-prop stage's
    interpolation sentinel). Normalising *both* sides through this one
    function is what makes a URL in code comparable to a bare host in a
    manifest."""
    if not text:
        return None
    t = str(text).strip().strip('"\'').strip()
    if not t:
        return None
    if "://" in t:
        try:
            t = urlsplit(t).netloc
        except ValueError:
            return None
    if "@" in t:
        t = t.rsplit("@", 1)[1]
    if t.startswith("[") and "]" in t:            # [::1]:6379
        t = t[1:t.index("]")]
    elif t.count(":") == 1:
        t = t.split(":", 1)[0]
    t = t.strip().rstrip(".").lower()
    if not t or any(c.isspace() for c in t) or "/" in t or "\x00" in t:
        return None
    if _PLACEHOLDER_RE.search(t):
        return None
    return t


def _hosts_in_value(value, key_path=""):
    """Every host a single scalar *value* declares. A connection string
    (`postgres://svc:pw@db.internal:5432/app`) yields `db.internal`; a
    bare `api.example.com` yields itself; a single-label `redis` yields
    itself only under a host-ish key, because bare words are otherwise
    indistinguishable from ordinary prose."""
    out = set()
    if not isinstance(value, str):
        return out
    v = value.strip()
    if not v or len(v) > 2048:
        return out

    for m in _URL_RE.finditer(v):
        host = _normalize_host(m.group(0))
        if host:
            out.add(host)

    host = _normalize_host(v)
    if host:
        leaf = key_path.rsplit(".", 1)[-1] if key_path else ""
        if _DOTTED_HOST_RE.match(host) or _IPV4_RE.match(host) or host == "localhost":
            out.add(host)
        elif _SINGLE_LABEL_RE.match(host) and _HOSTISH_KEY_RE.search(leaf):
            out.add(host)
    return out


def _service_dns_names(name, namespace=None):
    """The names a k8s Service / compose service actually answers to."""
    name = name.lower()
    names = {name}
    if namespace:
        ns = str(namespace).lower()
        names |= {f"{name}.{ns}", f"{name}.{ns}.svc", f"{name}.{ns}.svc.cluster.local"}
    else:
        names |= {f"{name}.default", f"{name}.default.svc", f"{name}.default.svc.cluster.local"}
    return names


# ------------------------------------------------------- per-format parsing ---

class ConfigValue(NamedTuple):
    key_path: str
    value: str
    line: int          # 1-indexed; 0 when the format gives us no line info


class ConfigIndex(NamedTuple):
    via: str                  # "parsed" | "text_fallback"
    fmt: str                  # yaml | json | toml | env | ini | dockerfile | unknown
    reason: str               # why tier 1 didn't apply, "" when it did
    hosts: dict               # host -> ConfigValue (first occurrence)
    aliases: dict             # declared DNS name -> (service_name, reason)
    services: dict            # service name -> reason
    n_values: int


def _format_for(path: str):
    p = PurePosixPath(path)
    name, ext = p.name.lower(), p.suffix.lower()
    if ext in (".yml", ".yaml"):
        return "yaml"
    if ext == ".json":
        return "json"
    if ext == ".toml":
        return "toml"
    if ext == ".env" or name == ".env" or name.startswith(".env."):
        return "env"
    if ext in (".ini", ".properties", ".cfg", ".conf"):
        return "ini"
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"
    return None


def _walk_yaml_node(node, prefix, out):
    if isinstance(node, yaml.MappingNode):
        for key_node, val_node in node.value:
            key = getattr(key_node, "value", "?")
            _walk_yaml_node(val_node, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            _walk_yaml_node(item, f"{prefix}[{i}]", out)
    elif isinstance(node, yaml.ScalarNode):
        out.append(ConfigValue(prefix, node.value, node.start_mark.line + 1))


def _parse_yaml(text):
    """Values come from `compose_all` (nodes carry line marks); the
    plain-object docs come from `safe_load_all` and are only used to find
    declared service names."""
    values = []
    try:
        for node in yaml.compose_all(text):
            if node is not None:
                _walk_yaml_node(node, "", values)
    except yaml.YAMLError as exc:
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return None, None, f"YAML parse failed: {first[:160]}"
    try:
        docs = [d for d in yaml.safe_load_all(text) if d is not None]
    except yaml.YAMLError:
        docs = []   # values are still good; only service-name detection is lost
    return values, docs, None


def _walk_object(obj, prefix, out, lines):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_object(v, f"{prefix}.{k}" if prefix else str(k), out, lines)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_object(v, f"{prefix}[{i}]", out, lines)
    elif isinstance(obj, str):
        out.append(ConfigValue(prefix, obj, _locate_line(lines, obj)))


def _locate_line(lines, needle):
    """Best-effort line number for a scalar from a parser that doesn't
    report one (json/toml). Reported as 0 when it can't be pinned down,
    rather than guessed at."""
    if not needle or len(needle) > 300:
        return 0
    for i, line in enumerate(lines, start=1):
        if needle in line:
            return i
    return 0


def _parse_json(text, lines):
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, None, f"JSON parse failed: {str(exc)[:160]}"
    values = []
    _walk_object(doc, "", values, lines)
    return values, [doc] if isinstance(doc, dict) else [], None


def _parse_toml(text, lines):
    if _toml is None:
        return None, None, _TOML_UNAVAILABLE
    try:
        doc = _toml.loads(text)
    except Exception as exc:  # noqa: BLE001 - TOMLDecodeError differs across backends
        return None, None, f"TOML parse failed: {str(exc)[:160]}"
    values = []
    _walk_object(doc, "", values, lines)
    return values, [doc], None


def _strip_inline_comment(line: str) -> str:
    if line.lstrip().startswith(("#", "//", ";")):
        return ""
    m = _INLINE_COMMENT.search(line)
    return line[: m.start()] if m else line


def _parse_keyvalue(lines, sections=False):
    """Real line-structured parse for .env / .properties / .ini: the key
    and the value are separated before anything is matched, so only the
    value is ever compared against a host, and comments never are."""
    values = []
    section = ""
    for i, raw in enumerate(lines, start=1):
        if _WHOLE_LINE_COMMENT.match(raw):
            continue
        line = _strip_inline_comment(raw).rstrip()
        if not line.strip():
            continue
        if sections:
            sec = re.match(r'^\s*\[([^\]]+)\]\s*$', line)
            if sec:
                section = sec.group(1).strip()
                continue
        m = _KEYVAL_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip().strip('"\'')
        values.append(ConfigValue(f"{section}.{key}" if section else key, value, i))
    return values


_DOCKER_DIRECTIVE = re.compile(r'^\s*(ENV|ARG|LABEL|EXPOSE|FROM|USER|CMD|ENTRYPOINT|RUN)\s+(.*)$', re.I)
_DOCKER_PAIR = re.compile(r'([A-Za-z_][\w.\-]*)=("[^"]*"|\'[^\']*\'|\S+)')


def _parse_dockerfile(lines):
    """Directive parse with line-continuation joining. `ENV` / `ARG` /
    `LABEL` get their `k=v` pairs split out so the key is separated from
    the value; other directives contribute their remainder as one value
    (a `RUN curl https://...` is a real outbound host)."""
    values = []
    buf, start = "", 0
    for i, raw in enumerate(lines, start=1):
        if _WHOLE_LINE_COMMENT.match(raw):
            continue
        line = raw.rstrip()
        if not line.strip():
            continue
        if not buf:
            start = i
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        joined, buf = (buf + line).strip(), ""
        m = _DOCKER_DIRECTIVE.match(joined)
        if not m:
            continue
        directive, rest = m.group(1).upper(), m.group(2).strip()
        if directive in ("ENV", "ARG", "LABEL"):
            pairs = _DOCKER_PAIR.findall(rest)
            if pairs:
                for key, value in pairs:
                    values.append(ConfigValue(f"{directive}.{key}", value.strip('"\''), start))
                continue
            parts = rest.split(None, 1)          # legacy `ENV KEY value` form
            if len(parts) == 2:
                values.append(ConfigValue(f"{directive}.{parts[0]}", parts[1].strip('"\''), start))
                continue
        values.append(ConfigValue(directive, rest, start))
    return values


def _declared_services(docs):
    """Service names the config *declares* — compose `services:` keys and
    k8s Service/StatefulSet/Ingress `metadata.name`. This is the
    structured replacement for the old `host.split(".")[0]` needle: a
    name is only a service name if a document says it is."""
    services, aliases = {}, {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        compose = doc.get("services")
        if isinstance(compose, dict):
            for name in compose:
                if isinstance(name, str) and name:
                    services.setdefault(name.lower(), "docker-compose service")
                    for alias in _service_dns_names(name):
                        aliases.setdefault(alias, (name.lower(), "docker-compose service"))
        kind, meta = doc.get("kind"), doc.get("metadata")
        if isinstance(kind, str) and isinstance(meta, dict):
            name = meta.get("name")
            if kind in ("Service", "StatefulSet", "Ingress", "Deployment") and isinstance(name, str) and name:
                reason = f"k8s {kind} metadata.name"
                services.setdefault(name.lower(), reason)
                for alias in _service_dns_names(name, meta.get("namespace")):
                    aliases.setdefault(alias, (name.lower(), reason))
    return services, aliases


def _build_config_index(rec, ctx=None) -> ConfigIndex:
    """Tier 1 for one config file. Cached on the record: this stage is
    called once after the literal/const-prop checks and again after the
    Joern/LLM stages, and the file hasn't changed in between."""
    cached = getattr(rec, "_peach_config_index", None)
    if cached is not None:
        return cached

    lines = rec.source_lines
    text = "\n".join(lines)
    fmt = _format_for(rec.path)
    values, docs, err = None, [], ""

    if fmt is None:
        err = "no parser for this file type"
    elif fmt == "yaml":
        if yaml is None:
            err = _YAML_UNAVAILABLE
        else:
            values, docs, err = _parse_yaml(text)
            err = err or ""
            docs = docs or []
    elif fmt == "json":
        values, docs, err = _parse_json(text, lines)
        err, docs = err or "", docs or []
    elif fmt == "toml":
        values, docs, err = _parse_toml(text, lines)
        err, docs = err or "", docs or []
    elif fmt == "env":
        values = _parse_keyvalue(lines)
    elif fmt == "ini":
        values = _parse_keyvalue(lines, sections=True)
    elif fmt == "dockerfile":
        values = _parse_dockerfile(lines)

    if values is None:
        index = ConfigIndex(via="text_fallback", fmt=fmt or "unknown", reason=err or "unparsed",
                            hosts={}, aliases={}, services={}, n_values=0)
        _bump(ctx, "config.file.text_fallback")
        _bump(ctx, f"config.file.text_fallback.{fmt or 'unknown'}")
        if TRACE_EDGES:
            log(LOG, "debug", "config file not parseable; using text fallback",
                config=rec.path, fmt=fmt or "unknown", reason=err)
    else:
        hosts = {}
        for cv in values:
            for host in _hosts_in_value(cv.value, cv.key_path):
                hosts.setdefault(host, cv)
        services, aliases = _declared_services(docs)
        index = ConfigIndex(via="parsed", fmt=fmt, reason="", hosts=hosts,
                            aliases=aliases, services=services, n_values=len(values))
        _bump(ctx, "config.file.parsed")
        _bump(ctx, f"config.file.parsed.{fmt}")

    setattr(rec, "_peach_config_index", index)
    return index


# ------------------------------------------------------------- text fallback ---

def _text_fallback_hits(rec, host):
    """Tier 2: the old scan, minus its three bugs. Whole-line and inline
    comments are skipped, the search is limited to the value side of a
    `key: value` line when there is one, and the host must sit on token
    boundaries — so `api.example.com` no longer matches inside
    `staging.api.example.com`, and no service-label needle is used at
    all."""
    pattern = re.compile(r'(?<![\w.\-])' + re.escape(host) + r'(?![\w\-])', re.I)
    for i, raw in enumerate(rec.source_lines, start=1):
        if _WHOLE_LINE_COMMENT.match(raw):
            continue
        line = _strip_inline_comment(raw)
        if not line.strip():
            continue
        m = _KEYVAL_RE.match(line)
        key, haystack = (m.group(1), m.group(2)) if m else ("", line)
        if pattern.search(haystack):
            return ConfigValue(key, haystack.strip()[:200], i)
    return None


# ------------------------------------------------------------------ matching ---

class ConfigMatch(tuple):
    """Still a `(edge, config_path, hit)` 3-tuple, so existing callers
    that unpack it keep working — with the evidence for the match hung
    off it as attributes rather than thrown away."""
    def __new__(cls, edge, config_path, hit, *, kind, confidence,
                key_path, line, value, via):
        obj = tuple.__new__(cls, (edge, config_path, hit))
        obj.edge, obj.config_path, obj.hit = edge, config_path, hit
        obj.kind, obj.confidence, obj.via = kind, confidence, via
        obj.key_path, obj.line, obj.value = key_path, line, value
        return obj

    def __repr__(self):
        where = f"{self.config_path}:{self.line}" if self.line else self.config_path
        return f"<ConfigMatch {self.hit} -> {where} ({self.kind}/{self.confidence})>"


def _match_host(host, index, rec):
    """(kind, confidence, ConfigValue, detail) for the strongest match in
    one config file, or None. Tiers within tier 1 are ordered by how much
    the match actually proves."""
    cv = index.hosts.get(host)
    if cv:
        return "host_value", "high", cv, host

    alias = index.aliases.get(host)
    if alias:
        name, reason = alias
        return "service_dns", "high", ConfigValue(f"<{reason}>", name, 0), name

    # The disciplined version of the old label needle: the host's first
    # label matches a name this config *declares* as a service. Real for
    # `payments.prod.internal` vs a Service named `payments`, but it's an
    # inference, not a sighting, so it's reported as medium.
    first = host.split(".")[0]
    if len(first) > 2 and first in index.services:
        return "service_name", "medium", ConfigValue(f"<{index.services[first]}>", first, 0), first

    if index.via == "text_fallback":
        cv = _text_fallback_hits(rec, host)
        if cv:
            return "text", "low", cv, host
    return None


def check_config_references(edges, config_records, ctx=None, pass_label="initial"):
    """Returns a list of ConfigMatch — a `(edge, config_path, hit)` tuple
    carrying `.kind`, `.confidence`, `.key_path` and `.line`. Does not
    mutate edges beyond appending to `config_matches`.

    Safe to call more than once on the same edge list — e.g. once after
    the literal/const-prop checks, and again after the Joern escalation
    and LLM fallback stages resolve additional hosts — since
    already-recorded (edge, config_path) pairs are skipped rather than
    duplicated, and each file is parsed once and cached.
    """
    if not config_records:
        # Worth a warning rather than a silent no-op: an empty result here
        # means "we never looked", not "we looked and found nothing".
        log(LOG, "warning", "no config files in repo; config stage is a no-op",
            pass_label=pass_label)
        if ctx is not None and pass_label == "initial":
            ctx.note("warning", "config",
                     "No config/manifest files were detected, so external services "
                     "could not be cross-referenced to any deployment config. "
                     "Check CONFIG_EXTS/CONFIG_NAME_HINTS if the repo does have them.")
        return []

    indexes = [(rec, _build_config_index(rec, ctx=ctx)) for rec in config_records]

    matches = []
    hosts_checked = 0
    hosts_unusable = 0
    hosts_matched = 0
    by_kind = {}

    for edge in edges:
        if edge.status != "external_candidate" or not edge.host:
            continue
        hosts_checked += 1
        host = _normalize_host(edge.host)
        if not host:
            # An interpolated or otherwise non-static host reaching this
            # stage is a real signal about the upstream stages, not
            # something to paper over with a substring search.
            hosts_unusable += 1
            _bump(ctx, "config.bail.unusable_host")
            if TRACE_EDGES:
                log(LOG, "debug", "host is not a usable static hostname; skipping",
                    file=edge.file, line=edge.line, host=edge.host)
            continue

        hit_any = False
        for rec, index in indexes:
            if rec.path in edge.config_matches:
                continue
            found = _match_host(host, index, rec)
            if not found:
                continue
            kind, confidence, cv, detail = found
            edge.config_matches.append(rec.path)
            matches.append(ConfigMatch(
                edge, rec.path, detail, kind=kind, confidence=confidence,
                key_path=cv.key_path, line=cv.line, value=cv.value, via=index.via,
            ))
            by_kind[kind] = by_kind.get(kind, 0) + 1
            hit_any = True
            if TRACE_EDGES:
                log(LOG, "debug", "host matched a config file", host=host,
                    config=rec.path, kind=kind, confidence=confidence,
                    key=cv.key_path, config_line=cv.line, tier=index.via)
        if hit_any:
            hosts_matched += 1

    parsed_files = sum(1 for _, ix in indexes if ix.via == "parsed")
    fallback_files = [(r.path, ix.reason) for r, ix in indexes if ix.via == "text_fallback"]

    if ctx is not None:
        ctx.bump(f"config.pass.{pass_label}.hosts_checked", hosts_checked)
        ctx.bump(f"config.pass.{pass_label}.hosts_matched", hosts_matched)
        ctx.bump(f"config.pass.{pass_label}.matches", len(matches))
        for kind, n in by_kind.items():
            ctx.bump(f"config.pass.{pass_label}.matches.{kind}", n)
        if hosts_unusable:
            ctx.bump(f"config.pass.{pass_label}.hosts_unusable", hosts_unusable)
        if fallback_files and pass_label == "initial":
            reasons = sorted({reason for _, reason in fallback_files})
            ctx.note("info", "config",
                     f"{len(fallback_files)} config file(s) could not be parsed in their own "
                     "format and were searched as text instead, which only finds a host "
                     "written out literally: " + "; ".join(reasons[:4]))

    log(LOG, "info", "config cross-reference complete", pass_label=pass_label,
        config_files=len(config_records), parsed_files=parsed_files,
        text_fallback_files=len(fallback_files), hosts_checked=hosts_checked,
        hosts_matched=hosts_matched, hosts_unusable=hosts_unusable,
        new_matches=len(matches), by_kind=by_kind)
    return matches


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)