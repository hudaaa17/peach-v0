"""
Stage 3 (step 1 of the external-call stages): ast-grep runner and callee
catalogue. This module only FINDS candidate call expressions; it does not read
or write CallEdge, and it emits no Gaps. The next step joins the hits to edges
and turns `run_report["reasons"]` into Gap entries.

What it does
------------
`run_patterns(repo_root, files_by_language)` runs every catalogue entry that
applies to a language over exactly the files it is given and returns

    (list[RawHit], run_report)

Files are grouped into buckets, one per ast-grep language: python, javascript
(js/jsx/mjs/cjs), typescript, tsx (`.tsx` is split out of the parser's
"typescript" because the typescript grammar cannot parse JSX). For each bucket
and each chunk of FILES_PER_CALL files ONE ast-grep process runs, with one rule
per catalogue entry of that language family.

Before call scanning, a separate import-discovery scan runs for each bucket
and chunk. It returns per-file bindings, including imports nested in functions;
bindings are per file, not per scope. Plain call scanning still covers all files.
A separate alias pass scans only files with non-trivial bindings, using one
union rule per applicable entry, then filters matches against each file's own
bindings. Alias rules use temporary files (removed in finally); a bucket above
1 MiB of alias rules is reported as alias_rules_too_large and receives no alias scan.
No discovery prefilter is used: not all file bytes are already held before
discovery. Validated bytes of import matches are cached for call scanning.

The catalogue and the rules
---------------------------
`CATALOGUE` is plain data, one `Entry` per external client call:

    id                stable name, e.g. "py.requests" (later stages key on it)
    language          "python" or "javascript" (the JS family: js, jsx, ts, tsx)
    callee_regex      matched against the text of the call's callee
                      (`requests.get`). Built with `_callee(...)`, which also
                      allows whitespace around the dots so a callee split over
                      lines (`axios\\n  .get(url)`, as Prettier writes call
                      chains) still matches. Rust regex syntax (ast-grep's
                      engine): no lookaround, no backreferences.
    url_arg_index     0-based position of the URL argument, or None
    url_arg_keyword   Python keyword-argument name of the URL (`url=`)
    url_arg_property  JS/TS: the URL is the property of this name on the first
                      object-literal argument (`axios({ url: ... })`). Where
                      url_arg_index is also set, that position holds the URL when
                      the call has no such object argument (`axios(url)`).
    description       one line for humans
    module            import module, or None for globals (no alias support)
    members           supported member names; empty for a callable module
    kind              external_call (default), or client_factory (no URL argument)

Each entry becomes one ast-grep rule (see `_rule_doc`): a call node (kind `call`
in Python, `call_expression` in JS/TS/TSX; both have a `function` field) whose
`function` child is captured as `$F`, with a constraint that `$F` matches
callee_regex. For TS/TSX an await_expression in that field is unwrapped and
its identifier/member child is captured instead (ast-grep 0.45.3's awaited
generic-call shape). Nothing about the arguments or type arguments is described, so
`axios.get<User>(url)` matches like `axios.get(url)`. The rules of a bucket are
passed together as `--inline-rules` documents separated by `---`, and each match
comes back with its rule id, which is the catalogue id.

Position convention (same as parser.py's CallSite.callee_line/callee_col)
-------------------------------------------------------------------------
line   1-based
column 0-based offset in UTF-8 BYTES from the start of the line (Tree-sitter's
       `start_point` column), `end_col` exclusive.

RawHit positions are computed here from ast-grep's `range.byteOffset` against
the file's raw bytes. There is no fallback: a match without `byteOffset` is an
"unreadable_output" reason, not a guess. A match whose offsets do not reproduce
ast-grep's matched text exactly is dropped and reported. Only "\\n" ends a line
(as in Tree-sitter). Two things make the raw file differ from what parser.py
parsed, so hits in such a file are dropped and reported rather than risking a
wrong column: a lone "\\r" line break (Python's universal newlines turns it into
"\\n"), and bytes that are not valid UTF-8 (parser.py reads with errors="ignore").
CRLF files are fine.

For that awaited generic-call shape, the validated `$F` byte range supplies
the start, excluding `await`; matched_text is trimmed to the same start and
the call's end is retained. Positions and text still describe the same bytes.

`RawHit.callee_text` is ast-grep's captured `$F` with all whitespace removed
(`axios\\n  .get` -> `axios.get`), which is how parser.py spells `callee_expr`.

run_report
----------
    {"binary": path or None, "version": str,
     "reasons":   [ {kind, cause, language, detail, count, samples} ... ],
     "languages": {parser language: {files, runs, hits, dropped, seconds}}}

Per-language discovery stats also include files_scanned, files_with_aliases,
alias_names (distinct local names across that language's buckets),
discovery_runs, discovery_skipped_files (0 without a prefilter), and
discovery_seconds. Existing runs/seconds counters describe plain call scans.
Factory rules share that plain scan, without extra launches. Their matches
never enter hits: per-language client_factory_hits counts them, while
run_report["client_instances"] contains one record per file, sorted by file,
with language, count, sorted distinct factory IDs, up to three source-position
samples, via_alias_count, kind="client_instance_untracked", and impact="calls made through this
client object are not detected". These records do not change reasons or smoke
exit status; they report the remaining client-call blind spot.
Alias stats include alias_files, alias_names, alias_hits_raw, alias_hits_kept,
alias_hits_dropped (including duplicates), alias_discovery_seconds,
alias_pass_launches, alias_pass_seconds and alias_rule_bytes. Dedupe uses file,
start/end bytes and catalogue id; plain wins over alias for that id. A justified
factory alias supersedes an external result at the same range, so one call
cannot be counted as both kinds. RawHit.via_alias marks retained alias calls;
pattern_id remains canonical and callee_text retains the written spelling.
Trivial bindings are ignored only when all applicable catalogue spellings
already match their corresponding plain regex. Unknown members are retained
as bindings but have no applicable call entry.

Nothing here raises; every problem is a reason. A reason is one aggregated entry
per (kind, cause, language, detail): `count` says how many times it happened
(unit in the table) and `samples` holds up to MAX_SAMPLES example file paths.
`language` is the parser language, or "" for the whole stage. A language with a
reason was NOT fully checked: missing hits there mean "unknown", not "no
external calls".

    kind                       cause               count unit      meaning
    ast_grep_unavailable       binary_missing      1               no usable ast-grep binary
    ast_grep_language_failed   timeout             ast-grep runs   a run exceeded the timeout
                               launch_failed       runs            the process could not start
                               exit_nonzero        runs            non-zero exit (detail: stderr)
                               unreadable_output   runs or matches stdout not JSON lines, or a match
                                                                   lacks file/text/range/byteOffset/
                                                                   metaVariables (detail says what)
                               unknown_rule_id     matches         output names a rule we did not send
                               offset_mismatch     matches         byteOffset does not reproduce text
                               unreadable_file     matches         file cannot be read for positions
                               invalid_utf8        matches         file is not valid UTF-8
                               lone_cr             matches         file has a lone "\\r" line break
                               missing_file        files           listed file does not exist
                               unknown_language    files           no catalogue entries for the language
                               alias_discovery_failed runs/chunks  alias calls may be missed; plain scans still run
    stage_failed               exception           1               unexpected exception (detail: text)

Exit codes: ast-grep exiting 1 with BOTH stdout and stderr empty means "no
matches". Exit 1 with anything on stdout or stderr, or any other non-zero exit,
is an exit_nonzero failure whose detail carries the stderr text.

Binary discovery
----------------
PEACH_AST_GREP_BIN (if set it is the only candidate; there is no fall-through to
PATH), else `ast-grep`, else `sg`. Every candidate must answer `--version` with
text that mentions "ast-grep": on Linux `sg` is usually shadow-utils' "run a
command with another group", which is not ast-grep. PEACH_AST_GREP_TIMEOUT
(seconds, default 60) bounds each ast-grep process and each version probe.

Known blind spots (by design, not detected)
-------------------------------------------
Matching is by the callee's spelling at the call site, so these are NOT found:
  * aliases from assignment (e.g. `const { get } = axios`), Python wildcard
    imports (`from requests import *`) and dynamic import(); JS namespace
    imports (`import * as ax from "axios"`) are supported
  * aliases bind per file, not per scope; scope shadowing and canonical-name
    shadowing such as `import httpx as requests` can produce false positives
  * calls through session / client objects: `s = requests.Session(); s.get(url)`,
    `client = axios.create(); client.get(url)`, `self.http.get(url)`; supported
    factory calls are recorded only as client_instances, and subsequent calls
    through the resulting objects are not detected
  * optional chaining (`axios?.get(url)`) and callees built from call results
    or computed members
  * libraries not in the catalogue (aiohttp, urllib3, http.client,
    XMLHttpRequest, jQuery, ...) and verbs not listed (e.g. httpx.options)

ast-grep behaviour this relies on. Checked against ast-grep 0.45.3 by the
real-binary tests: `range.byteOffset` in the JSON, nested matches are all
reported, `--lang tsx` / `language: tsx`, explicit file paths. NOT verified when
this was written (no ast-grep was available): `scan --inline-rules` accepting
several JSON documents joined by `---`, `ruleId` and `metaVariables.single.F.text`
in the JSON output, `has` with `field` + `pattern: $F`, `constraints`, hint
severity being reported, scan's exit codes. The real-binary tests exercise each;
a wrong assumption shows up as a reason (rule error -> exit_nonzero with stderr,
missing field -> unreadable_output), never as silently empty results.
"""
import ast
import bisect
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .obs import get_logger, log

LOG = get_logger("ast_grep_check")

DEFAULT_TIMEOUT = 60
FILES_PER_CALL = 100            # files passed to one ast-grep process
MAX_SAMPLES = 5                 # example paths kept per reason
MAX_ALIAS_RULE_BYTES = 1024 * 1024
_TEXT_CHARS = 500               # how much ast-grep stderr goes into a reason / log line

# ast-grep language -> node kind of a call (both kinds have a `function` field).
_CALL_KIND = {"python": "call", "javascript": "call_expression",
              "typescript": "call_expression", "tsx": "call_expression"}

# Parser language -> catalogue language.
_FAMILY = {"python": "python", "javascript": "javascript", "typescript": "javascript"}

_bin_cache = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ catalogue ----

@dataclass(frozen=True)
class Entry:
    id: str
    language: str                           # "python" | "javascript" (js, jsx, ts, tsx)
    callee_regex: str
    url_arg_index: Optional[int] = 0
    url_arg_keyword: Optional[str] = None   # Python: keyword argument name
    url_arg_property: Optional[str] = None  # JS/TS: property of the first object-literal argument
    description: str = ""
    module: Optional[str] = None
    members: tuple[str, ...] = ()           # empty: callable module itself
    kind: str = "external_call"


def _callee(*parts):
    """Anchored regex for a dotted callee: parts are regex fragments joined by a
    dot that may have whitespace around it. _callee("axios", "(get|post)") ->
    ^axios\\s*\\.\\s*(get|post)$"""
    return "^" + r"\s*\.\s*".join(parts) + "$"


_PY_VERBS = "(get|post|put|delete|patch|head)"
_JS_VERBS = "(get|post|put|patch|delete|head)"

CATALOGUE = (
    Entry("py.requests", "python", _callee("requests", "(get|post|put|delete|patch|head|options)"),
          module='requests', members=('get', 'post', 'put', 'delete', 'patch', 'head', 'options'),
          url_arg_index=0, url_arg_keyword="url",
          description="requests.get/post/put/delete/patch/head/options(url, ...)"),
    Entry("py.requests.request", "python", _callee("requests", "request"),
          module='requests', members=('request',),
          url_arg_index=1, url_arg_keyword="url", description="requests.request(method, url, ...)"),
    Entry("py.httpx", "python", _callee("httpx", _PY_VERBS),
          module='httpx', members=('get', 'post', 'put', 'delete', 'patch', 'head'),
          url_arg_index=0, url_arg_keyword="url", description="httpx.get/post/put/delete/patch/head(url, ...)"),
    Entry("py.httpx.request", "python", _callee("httpx", "request"),
          module='httpx', members=('request',),
          url_arg_index=1, url_arg_keyword="url", description="httpx.request(method, url, ...)"),
    Entry("py.httpx.stream", "python", _callee("httpx", "stream"),
          module='httpx', members=('stream',),
          url_arg_index=1, url_arg_keyword="url", description="httpx.stream(method, url, ...)"),
    Entry("py.urllib.urlopen", "python", _callee("urllib", "request", "urlopen"),
          module='urllib.request', members=('urlopen',),
          url_arg_index=0, url_arg_keyword="url", description="urllib.request.urlopen(url_or_request)"),

    Entry("js.fetch", "javascript", _callee("fetch"),
          url_arg_index=0, description="fetch(url, init)"),
    Entry("js.window.fetch", "javascript", _callee("window", "fetch"),
          url_arg_index=0, description="window.fetch(url, init)"),
    Entry("js.globalThis.fetch", "javascript", _callee("globalThis", "fetch"),
          url_arg_index=0, description="globalThis.fetch(url, init)"),
    Entry("js.axios.method", "javascript", _callee("axios", "(get|post|put|delete|patch|head|options)"),
          module='axios', members=('get', 'post', 'put', 'delete', 'patch', 'head', 'options'),
          url_arg_index=0, description="axios.get/post/put/delete/patch/head/options(url, ...)"),
    Entry("js.axios.request", "javascript", _callee("axios", "request"),
          module='axios', members=('request',),
          url_arg_index=None, url_arg_property="url", description="axios.request(config)"),
    Entry("js.axios", "javascript", _callee("axios"),
          module='axios', members=(),
          url_arg_index=0, url_arg_property="url", description="axios(url) or axios(config)"),
    Entry("js.got", "javascript", _callee("got"),
          module='got', members=(),
          url_arg_index=0, url_arg_property="url", description="got(url) or got(options)"),
    Entry("js.got.method", "javascript", _callee("got", _JS_VERBS),
          module='got', members=('get', 'post', 'put', 'patch', 'delete', 'head'),
          url_arg_index=0, description="got.get/post/put/patch/delete/head(url, ...)"),
    Entry("js.ky", "javascript", _callee("ky"),
          module='ky', members=(),
          url_arg_index=0, description="ky(url, options)"),
    Entry("js.ky.method", "javascript", _callee("ky", _JS_VERBS),
          module='ky', members=('get', 'post', 'put', 'patch', 'delete', 'head'),
          url_arg_index=0, description="ky.get/post/put/patch/delete/head(url, ...)"),
    Entry("js.superagent", "javascript", _callee("superagent", _JS_VERBS),
          module='superagent', members=('get', 'post', 'put', 'patch', 'delete', 'head'),
          url_arg_index=0, description="superagent.get/post/put/patch/delete/head(url)"),
    Entry("js.http.request", "javascript", _callee("http", "request"),
          module='http', members=('request',),
          url_arg_index=0, description="http.request(url_or_options, ...)"),
    Entry("js.https.request", "javascript", _callee("https", "request"),
          module='https', members=('request',),
          url_arg_index=0, description="https.request(url_or_options, ...)"),
    Entry("js.http.get", "javascript", _callee("http", "get"),
          module='http', members=('get',),
          url_arg_index=0, description="http.get(url_or_options, ...)"),
    Entry("js.https.get", "javascript", _callee("https", "get"),
          module='https', members=('get',),
          url_arg_index=0, description="https.get(url_or_options, ...)"),
    Entry("py.requests.Session", "python", _callee("requests", "Session"),
          module="requests", members=("Session",), kind="client_factory",
          url_arg_index=None, description="requests.Session()"),
    Entry("py.requests.session", "python", _callee("requests", "session"),
          module="requests", members=("session",), kind="client_factory",
          url_arg_index=None, description="requests.session()"),
    Entry("py.httpx.Client", "python", _callee("httpx", "Client"),
          module="httpx", members=("Client",), kind="client_factory",
          url_arg_index=None, description="httpx.Client(...)"),
    Entry("py.httpx.AsyncClient", "python", _callee("httpx", "AsyncClient"),
          module="httpx", members=("AsyncClient",), kind="client_factory",
          url_arg_index=None, description="httpx.AsyncClient(...)"),
    Entry("js.axios.create", "javascript", _callee("axios", "create"),
          module="axios", members=("create",), kind="client_factory",
          url_arg_index=None, description="axios.create(config)"),
    Entry("js.got.extend", "javascript", _callee("got", "extend"),
          module="got", members=("extend",), kind="client_factory",
          url_arg_index=None, description="got.extend(options)"),
    Entry("js.ky.create", "javascript", _callee("ky", "create"),
          module="ky", members=("create",), kind="client_factory",
          url_arg_index=None, description="ky.create(options)"),
    Entry("js.ky.extend", "javascript", _callee("ky", "extend"),
          module="ky", members=("extend",), kind="client_factory",
          url_arg_index=None, description="ky.extend(options)"),
    Entry("js.superagent.agent", "javascript", _callee("superagent", "agent"),
          module="superagent", members=("agent",), kind="client_factory",
          url_arg_index=None, description="superagent.agent()"),
)


@dataclass
class RawHit:
    """One ast-grep match. Positions follow the convention in the module
    docstring (1-based line, 0-based UTF-8 byte column, end exclusive); they
    describe the call from the callee through the arguments, excluding await."""
    file: str            # repo-relative POSIX path
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    matched_text: str    # the whole call, exactly as in the file
    pattern_id: str      # Entry.id (the ast-grep rule id)
    language: str        # parser language of the file: python | javascript | typescript
    callee_text: str     # captured callee with whitespace removed, e.g. "requests.get"
    via_alias: bool = False


def _rule_doc(entry, ast_lang):
    """The ast-grep rule (a dict) for one catalogue entry."""
    function = {"field": "function", "pattern": "$F"}
    if ast_lang in ("typescript", "tsx"):
        # 0.45.3 puts await inside the function field of generic calls.
        # Disjoint branches keep one rule/match and constrain the real callee.
        function = {"field": "function", "any": [
            {"all": [{"not": {"kind": "await_expression"}}, {"pattern": "$F"}]},
            {"kind": "await_expression", "pattern": "$AWAIT",
             "has": {"any": [{"kind": "member_expression"}, {"kind": "identifier"}],
                     "pattern": "$F"}},
        ]}
    return {
        "id": entry.id,
        "language": ast_lang,
        "severity": "hint",
        "message": entry.description,
        "rule": {"kind": _CALL_KIND[ast_lang],
                 "has": function},
        "constraints": {"F": {"regex": entry.callee_regex}},
    }


def _rules_text(entries, ast_lang):
    """All rules for a bucket as one `--inline-rules` value: JSON documents
    (JSON is valid YAML, so no quoting care is needed) separated by `---`."""
    return "\n---\n".join(json.dumps(_rule_doc(e, ast_lang)) for e in entries)


def _decode_rule_id(rule_id):
    """Canonical catalogue id and whether the id denotes an alias rule."""
    if not isinstance(rule_id, str):
        return None, False
    return (rule_id[:-6], True) if rule_id.endswith("#alias") else (rule_id, False)


def _alias_rule_docs(entries, bindings, ast_lang):
    """One union rule per applicable entry; unknown members contribute nothing."""
    docs = []
    for entry in entries:
        spellings = set()
        for items in bindings.values():
            for local, module, member in items:
                if entry.module is None or module != entry.module:
                    continue
                if member is None:
                    suffix = (r"\s*\.\s*(?:" + "|".join(re.escape(m) for m in entry.members) + ")"
                              if entry.members else "")
                    spellings.add(re.escape(local) + suffix)
                elif member in entry.members:
                    spellings.add(re.escape(local))
        if spellings:
            doc = _rule_doc(entry, ast_lang)
            doc["id"] = entry.id + "#alias"
            doc["constraints"]["F"]["regex"] = "^(?:" + "|".join(sorted(spellings)) + ")$"
            docs.append(doc)
    return docs


def _import_rule_docs(ast_lang):
    """Discovery only: one scan of these rules per bucket/chunk, including nested imports."""
    if ast_lang == "python":
        patterns = {"python": {"any": [{"kind": "import_statement"},
                                       {"kind": "import_from_statement"}]}}
    else:
        statement = {"kind": "import_statement", "not": {"regex": r"^import\s+type\b"},
                     "has": {"field": "source", "pattern": "$SOURCE"}}
        clause = {"kind": "import_clause", "inside": statement}
        require = {"kind": "variable_declarator", "has": {
            "field": "value", "kind": "call_expression", "pattern": "require($SOURCE)",
            "has": {"field": "arguments", "has": {"kind": "string"}}}}
        destructuring = {"kind": "object_pattern", "inside": require}
        patterns = {
            "default": {"kind": "identifier", "pattern": "$LOCAL", "inside": clause},
            "namespace": {"kind": "namespace_import",
                          "has": {"kind": "identifier", "pattern": "$LOCAL"}, "inside": clause},
            "named": {"kind": "import_specifier", "not": {"regex": r"^type\s"}, "all": [
                {"has": {"field": "name", "pattern": "$MEMBER"}},
                {"any": [{"has": {"field": "alias", "pattern": "$LOCAL"}},
                         {"all": [{"not": {"has": {"field": "alias", "pattern": "$IGNORED"}}},
                                  {"has": {"field": "name", "pattern": "$LOCAL"}}]}]}],
                      "inside": dict(statement, stopBy="end")},
            "require": dict(require, all=[{"has": {
                "field": "name", "kind": "identifier", "pattern": "$LOCAL"}}]),
            "shorthand": {"kind": "shorthand_property_identifier_pattern", "pattern": "$LOCAL",
                          "inside": destructuring},
            "pair": {"kind": "pair_pattern", "all": [
                {"has": {"field": "key", "pattern": "$MEMBER"}},
                {"has": {"field": "value", "kind": "identifier", "pattern": "$LOCAL"}}],
                     "inside": destructuring},
        }
        if ast_lang in ("typescript", "tsx"):
            patterns["equals"] = {"kind": "import_require_clause", "all": [
                {"has": {"field": "source", "pattern": "$SOURCE"}},
                {"has": {"kind": "identifier", "pattern": "$LOCAL"}}],
                "inside": {"kind": "import_statement", "not": {"regex": r"^import\s+type\b"}}}
    return [{"id": "import." + name, "language": ast_lang, "rule": rule}
            for name, rule in patterns.items()]


def _bindings_from_import(obj, ast_lang, modules):
    """Decode structural captures (Python import statements use the stdlib AST).

    Bindings are per file, not per scope. Unsupported modules, relative imports,
    Python wildcard imports and dynamic loaders do not supply bindings.
    """
    if ast_lang == "python":
        statement, = ast.parse(obj["text"]).body
        bindings = []
        if isinstance(statement, ast.Import):
            for name in statement.names:
                # Unaliased dotted imports keep their canonical prefix for the
                # triviality check below; they never become alias bindings.
                bindings.append((name.asname or name.name, name.name, None))
        elif isinstance(statement, ast.ImportFrom) and not statement.level:
            for name in statement.names:
                if name.name == "*":
                    continue
                module = statement.module
                submodule = module + "." + name.name
                bindings.append((name.asname or name.name,
                                 submodule if submodule in modules else module,
                                 None if submodule in modules else name.name))
        return [b for b in bindings if b[1] in modules]
    captures = obj["metaVariables"]["single"]
    local = captures["LOCAL"]["text"]
    source = captures["SOURCE"]["text"]
    if not isinstance(local, str) or not isinstance(source, str):
        raise ValueError("import captures must be strings")
    module = ast.literal_eval(source)
    if not isinstance(module, str):
        raise ValueError("import source must be a string literal")
    if module in ("node:http", "node:https"):
        module = module.removeprefix("node:")
    member = captures.get("MEMBER", {}).get("text")
    if obj["ruleId"] == "import.shorthand":
        member = local
    if member is not None and not isinstance(member, str):
        raise ValueError("import member must be a string")
    return [(local, module, member)] if module in modules else []


def _binding_is_trivial(binding, entries):
    """True only if every applicable catalogue spelling is already a plain hit.

    A module binding checks all its callable/member entries; a named binding
    checks entries containing that member. No applicable entry is not trivial.
    """
    local, module, member = binding
    matches = []
    for entry in entries:
        if entry.module != module:
            continue
        if member is None:
            spellings = [local + "." + m for m in entry.members] if entry.members else [local]
        elif member in entry.members:
            spellings = [local]
        else:
            continue
        matches.extend(bool(re.fullmatch(entry.callee_regex, s)) for s in spellings)
    return bool(matches) and all(matches)


# ---------------------------------------------------------------- binary / subprocess ----

def _timeout():
    try:
        return int(os.environ.get("PEACH_AST_GREP_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _run(cmd, cwd=None):
    """Run a command. Returns (CompletedProcess, None) or (None, (cause, text)).
    Never raises. Output is decoded as UTF-8 with replacement."""
    timeout = _timeout()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd is not None else None, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return proc, None
    except subprocess.TimeoutExpired:
        return None, ("timeout", f"timed out after {timeout}s")
    except Exception as exc:  # noqa: BLE001 - launch problems must not escape
        return None, ("launch_failed", f"{type(exc).__name__}: {exc}")


def _tail(text):
    text = (text or "").strip()
    return text[:_TEXT_CHARS] + (" ..." if len(text) > _TEXT_CHARS else "")


def _probe(path):
    """(version line, None) if `path --version` identifies ast-grep, else
    (None, why not)."""
    proc, err = _run([path, "--version"])
    if proc is None:
        return None, err[1]
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode == 0 and "ast-grep" in out.lower():
        return out.splitlines()[0].strip(), None
    return None, f"`--version` does not identify ast-grep (exit {proc.returncode}): {_tail(out) or '<no output>'}"


def find_binary():
    """(path, version, tried): the ast-grep executable, its version line, and a
    list of what was rejected and why. path is None if there is none. Cached per
    process for a given PEACH_AST_GREP_BIN / PATH (misses too)."""
    override = os.environ.get("PEACH_AST_GREP_BIN", "").strip()
    key = (override, os.environ.get("PATH", ""))
    with _lock:
        if key in _bin_cache:
            return _bin_cache[key]
    candidates = [override] if override else ["ast-grep", "sg"]
    tried = []
    found = (None, "", tried)
    for name in candidates:
        path = shutil.which(name)
        if not path:
            tried.append(f"{name}: not found")
            continue
        version, why = _probe(path)
        if version:
            found = (path, version, tried)
            break
        tried.append(f"{name} ({path}): {why}")
    with _lock:
        _bin_cache[key] = found
    return found


# ------------------------------------------------------------------- reasons ----

def _bump(ctx, key, n=1):
    bump = getattr(ctx, "bump", None)
    if bump is not None and n:
        bump(key, n)


def _note(report, ctx, kind, cause, language, detail, count=1, samples=()):
    """Record a reason, merging with an existing one that has the same
    (kind, cause, language, detail): counts add up, samples stay unique and
    capped at MAX_SAMPLES."""
    samples = [s for s in dict.fromkeys(samples) if s]
    for r in report["reasons"]:
        if (r["kind"], r["cause"], r["language"], r["detail"]) == (kind, cause, language, detail):
            r["count"] += count
            r["samples"].extend(s for s in samples if s not in r["samples"])
            del r["samples"][MAX_SAMPLES:]
            break
    else:
        report["reasons"].append({"kind": kind, "cause": cause, "language": language,
                                  "detail": detail, "count": count,
                                  "samples": samples[:MAX_SAMPLES]})
    _bump(ctx, f"ast_grep.reason.{cause}", count)


_FILE_PROBLEMS = {
    "unreadable_file": "file could not be read to compute positions",
    "invalid_utf8": "file is not valid UTF-8; columns would not match parser.py",
    "lone_cr": 'file has a lone "\\r" line break; line numbers would not match parser.py',
}


# ------------------------------------------------------------------- positions ----

def _relpath(file, root: Path):
    """Repo-relative POSIX path for a path ast-grep printed, or "" if unusable."""
    if not isinstance(file, str) or not file:
        return ""
    if os.path.isabs(file):
        try:
            return Path(file).resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            return ""
    rel = file.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def _load_view(root: Path, rel, views):
    """((raw_bytes, newline_offsets), None) for a file, or (None, problem key of
    _FILE_PROBLEMS). Cached in `views` for the run."""
    if rel in views:
        return views[rel]
    try:
        raw = (root / rel).read_bytes()
    except Exception:  # noqa: BLE001
        result = (None, "unreadable_file")
    else:
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            result = (None, "invalid_utf8")
        else:
            if b"\r" in raw.replace(b"\r\n", b""):
                result = (None, "lone_cr")
            else:
                result = ((raw, [i for i, b in enumerate(raw) if b == 0x0A]), None)
    views[rel] = result
    return result


def _position(newlines, offset):
    """(1-based line, 0-based byte column) of a byte offset."""
    n = bisect.bisect_left(newlines, offset)        # newlines strictly before offset
    line_start = newlines[n - 1] + 1 if n else 0
    return n + 1, offset - line_start


def _hit_from_json(obj, entries_by_id, parser_lang, root, views):
    """(RawHit, None) for one ast-grep JSON match, or (None, (cause, detail, file))
    when it cannot be used."""
    if not isinstance(obj, dict):
        return None, ("unreadable_output", "match is not a JSON object", "")
    rel = _relpath(obj.get("file"), root)
    text, rng = obj.get("text"), obj.get("range")
    if not rel or not isinstance(text, str) or not isinstance(rng, dict):
        return None, ("unreadable_output", "match lacks file, text or range", rel)
    entry = entries_by_id.get(obj.get("ruleId"))
    if entry is None:
        return None, ("unknown_rule_id", f"ast-grep reported rule id {obj.get('ruleId')!r}, which was not sent", rel)
    bo = rng.get("byteOffset")
    if not (isinstance(bo, dict) and isinstance(bo.get("start"), int) and isinstance(bo.get("end"), int)):
        return None, ("unreadable_output", "missing range.byteOffset", rel)
    captured = ((obj.get("metaVariables") or {}).get("single") or {}).get("F")
    if not (isinstance(captured, dict) and isinstance(captured.get("text"), str)):
        return None, ("unreadable_output", "missing metaVariables.single.F.text", rel)
    view, problem = _load_view(root, rel, views)
    if view is None:
        return None, (problem, _FILE_PROBLEMS[problem], rel)
    raw, newlines = view
    start, end = bo["start"], bo["end"]
    if not (0 <= start <= end <= len(raw)) or raw[start:end] != text.encode("utf-8"):
        return None, ("offset_mismatch", "range.byteOffset does not reproduce the matched text", rel)
    if "AWAIT" in ((obj.get("metaVariables") or {}).get("single") or {}):
        # Keep call positions/text consistent with ordinary calls: exclude await.
        fbo = (captured.get("range") or {}).get("byteOffset")
        if not (isinstance(fbo, dict) and isinstance(fbo.get("start"), int)
                and isinstance(fbo.get("end"), int)):
            return None, ("unreadable_output", "missing awaited callee range.byteOffset", rel)
        fs, fe = fbo["start"], fbo["end"]
        if not (start < fs < fe <= end) or raw[fs:fe] != captured["text"].encode("utf-8"):
            return None, ("offset_mismatch", "awaited callee offsets do not reproduce captured text", rel)
        start = fs
        text = raw[start:end].decode("utf-8")
    (sl, sc), (el, ec) = _position(newlines, start), _position(newlines, end)
    return RawHit(file=rel, start_line=sl, start_col=sc, end_line=el, end_col=ec,
                  matched_text=text, pattern_id=entry.id, language=parser_lang,
                  callee_text=re.sub(r"\s+", "", captured["text"])), None


# ---------------------------------------------------------------------- running ----

def _parse_json_lines(stdout):
    """One JSON value per non-empty line (`--json=stream`). Raises ValueError."""
    return [json.loads(line) for line in (stdout or "").splitlines() if line.strip()]


def _run_one(binary, rules, files, root: Path, rules_file=False):
    """(json values, None) or ([], (cause, detail)) for one ast-grep process."""
    temporary = None
    try:
        if rules_file:
            # Alias unions can exceed Windows command-line limits. ast-grep
            # 0.45.3 accepts multi-document rules through --rule as well.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yml", delete=False) as fh:
                temporary = Path(fh.name)
                fh.write(rules)
            cmd = [binary, "scan", "--rule", str(temporary), "--json=stream", "--", *files]
        else:
            cmd = [binary, "scan", "--inline-rules", rules, "--json=stream", "--", *files]
        proc, err = _run(cmd, cwd=root)
    except OSError as exc:
        return [], ("launch_failed", f"{type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink()
    if proc is None:
        return [], err
    out, errtext = (proc.stdout or "").strip(), (proc.stderr or "").strip()
    if proc.returncode == 1 and not out and not errtext:
        return [], None                     # exit 1 with no output at all: no matches
    if proc.returncode != 0:
        detail = f"exit {proc.returncode}: {_tail(errtext) or '<no stderr>'}"
        if out:
            detail += " (stdout was not empty)"
        return [], ("exit_nonzero", detail)
    try:
        return _parse_json_lines(proc.stdout), None
    except ValueError as exc:
        return [], ("unreadable_output", f"stdout is not JSON lines: {exc}")


def _discover_bindings(binary, parser_lang, ast_lang, files, root, report, ctx=None, views=None):
    """Return file -> non-trivial binding triples and discovery stats.

    A failed chunk invalidates discovery for the entire bucket; plain call
    scanning still runs. File bytes are loaded only for validating matches and
    shared with call scanning. No prefilter or extra prefilter read is used.
    """
    views = {} if views is None else views
    entries = [e for e in CATALOGUE if e.language == _FAMILY[parser_lang]]
    modules = {e.module for e in entries if e.module is not None}
    docs = _import_rule_docs(ast_lang)
    rule_ids = {d["id"] for d in docs}
    rules = "\n---\n".join(json.dumps(d) for d in docs)
    bindings = {f: set() for f in files}
    started = time.monotonic()
    runs = 0
    failed = False
    for i in range(0, len(files), FILES_PER_CALL):
        chunk = files[i:i + FILES_PER_CALL]
        runs += 1
        objs, failure = _run_one(binary, rules, chunk, root)
        if failure is None:
            try:
                for obj in objs:
                    if not isinstance(obj, dict) or obj.get("ruleId") not in rule_ids:
                        raise ValueError("unknown import rule id or non-object match")
                    rel = _relpath(obj.get("file"), root)
                    if rel not in chunk:
                        raise ValueError("import match is outside the requested chunk")
                    view, problem = _load_view(root, rel, views)
                    if view is None:
                        raise ValueError(problem)
                    raw, _ = view
                    bo = obj["range"]["byteOffset"]
                    start, end = bo["start"], bo["end"]
                    if not (isinstance(start, int) and isinstance(end, int)
                            and 0 <= start <= end <= len(raw)
                            and raw[start:end] == obj["text"].encode("utf-8")):
                        raise ValueError("import byte offsets do not reproduce matched text")
                    for binding in _bindings_from_import(obj, ast_lang, modules):
                        if not _binding_is_trivial(binding, entries):
                            bindings[rel].add(binding)
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                failure = ("unreadable_output", f"{type(exc).__name__}: {exc}")
        if failure:
            failed = True
            cause, detail = failure
            _note(report, ctx, "ast_grep_language_failed", "alias_discovery_failed", parser_lang,
                  f"--lang {ast_lang}: {cause}: {detail}; alias calls may be missed", samples=chunk)
    if failed:
        bindings = {f: set() for f in files}
    result = {f: sorted(items, key=lambda b: (b[0], b[1], b[2] or ""))
              for f, items in bindings.items()}
    stats = {"files_scanned": len(files), "files_with_aliases": sum(bool(v) for v in result.values()),
             "alias_names": len({b[0] for items in result.values() for b in items}),
             "discovery_runs": runs, "discovery_skipped_files": 0,
             "discovery_seconds": round(time.monotonic() - started, 3)}
    _bump(ctx, "ast_grep.discovery_runs", runs)
    _bump(ctx, "ast_grep.discovery_files", len(files))
    return result, stats


def _run_alias_pass(binary, parser_lang, ast_lang, bindings, root, report, ctx=None):
    """Collect candidate matches only from binding files; filtering is separate."""
    started = time.monotonic()
    files = sorted(f for f, items in bindings.items() if items)
    entries = [e for e in CATALOGUE if e.language == _FAMILY[parser_lang]]
    rules = "\n---\n".join(json.dumps(d) for d in _alias_rule_docs(entries, bindings, ast_lang))
    stats = {"alias_files": len(files), "alias_rule_bytes": len(rules.encode("utf-8")),
             "alias_pass_launches": 0, "alias_pass_seconds": 0.0}
    matches = []
    if stats["alias_rule_bytes"] > MAX_ALIAS_RULE_BYTES:
        _note(report, ctx, "ast_grep_language_failed", "alias_rules_too_large", parser_lang,
              f"--lang {ast_lang}: {stats['alias_rule_bytes']} bytes exceeds {MAX_ALIAS_RULE_BYTES}; "
              "alias calls may be missed", samples=files)
    elif rules:
        for i in range(0, len(files), FILES_PER_CALL):
            chunk = files[i:i + FILES_PER_CALL]
            stats["alias_pass_launches"] += 1
            objs, failure = _run_one(binary, rules, chunk, root, rules_file=True)
            if failure:
                cause, detail = failure
                _note(report, ctx, "ast_grep_language_failed", cause, parser_lang,
                      f"--lang {ast_lang} alias pass: {detail}", samples=chunk)
            else:
                matches.extend(objs)
    stats["alias_pass_seconds"] = round(time.monotonic() - started, 3)
    return matches, stats


def _buckets(root: Path, files_by_language, report, ctx):
    """[(parser_language, ast_grep_language, [files])]: existing files only,
    de-duplicated and sorted; `.tsx` split out of typescript."""
    out = []
    for lang in sorted(files_by_language):
        wanted = sorted(set(files_by_language[lang] or []))
        if lang not in _FAMILY:
            if wanted:
                _note(report, ctx, "ast_grep_language_failed", "unknown_language", lang,
                      f"no external-call entries exist for language '{lang}'",
                      count=len(wanted), samples=wanted)
            continue
        present = [f for f in wanted if (root / f).is_file()]
        present_set = set(present)
        missing = [f for f in wanted if f not in present_set]
        if missing:
            _note(report, ctx, "ast_grep_language_failed", "missing_file", lang,
                  "file does not exist under repo_root", count=len(missing), samples=missing)
        groups = {}
        for f in present:
            ast_lang = "tsx" if lang == "typescript" and f.lower().endswith(".tsx") else lang
            groups.setdefault(ast_lang, []).append(f)
        for ast_lang in sorted(groups):
            out.append((lang, ast_lang, groups[ast_lang]))
    return out


def _match_key(obj, hit):
    bo = obj["range"]["byteOffset"]
    single = obj["metaVariables"]["single"]
    start = single["F"]["range"]["byteOffset"]["start"] if "AWAIT" in single else bo["start"]
    return hit.file, start, bo["end"], hit.pattern_id


def _record_factory(hit, report):
    instance = next((r for r in report["client_instances"] if r["file"] == hit.file), None)
    if instance is None:
        instance = {"file": hit.file, "language": hit.language, "count": 0, "via_alias_count": 0,
                    "factories": [], "samples": [], "kind": "client_instance_untracked",
                    "impact": "calls made through this client object are not detected"}
        report["client_instances"].append(instance)
    instance["count"] += 1
    instance["via_alias_count"] += int(hit.via_alias)
    instance["factories"] = sorted(set(instance["factories"]) | {hit.pattern_id})
    instance["samples"].append(f"{hit.start_line}:{hit.start_col} {hit.callee_text}")
    instance["samples"] = sorted(instance["samples"], key=lambda s: (
        tuple(map(int, s.split(" ", 1)[0].split(":"))), s))[:3]


def _alias_filter_reason(hit, entry, bindings):
    head, dot, member_text = hit.callee_text.partition(".")
    local = [b for b in bindings.get(hit.file, []) if b[0] == head]
    if not local:
        return "check 1: head has no binding in this file"
    same_module = [b for b in local if b[1] == entry.module]
    if not same_module:
        return "check 2: binding module differs from entry module"
    if not entry.members:
        if not dot and any(b[2] is None for b in same_module):
            return None
        return "check 3: callable module requires a module binding and a bare callee"
    for _, _, member in same_module:
        if member is None and dot and member_text in entry.members:
            return None
        if member is not None and not dot and member in entry.members:
            return None
    return "check 4: callee or imported member is not an entry member"


def _process_alias_matches(objs, parser_lang, bindings, root, report, hits, views, seen, ctx=None,
                           factory_ranges=None):
    factory_ranges = set() if factory_ranges is None else factory_ranges
    by_id = {e.id + "#alias": e for e in CATALOGUE if e.language == _FAMILY[parser_lang]}
    stats = {"alias_hits_raw": len(objs), "alias_hits_kept": 0, "alias_hits_dropped": 0,
             "client_factory_hits": 0, "hits": 0}
    # A factory import can shadow a canonical callable (create as axios).
    # Resolve factories first so a range cannot become both kinds of result.
    ordered = sorted(objs, key=lambda o: not (isinstance(o, dict)
                     and o.get("ruleId") in by_id and by_id[o["ruleId"]].kind == "client_factory"))
    for obj in ordered:
        hit, problem = _hit_from_json(obj, by_id, parser_lang, root, views)
        if problem:
            stats["alias_hits_dropped"] += 1
            cause, detail, rel = problem
            _note(report, ctx, "ast_grep_language_failed", cause, parser_lang, detail, samples=[rel])
            continue
        entry = by_id[obj["ruleId"]]
        reason = _alias_filter_reason(hit, entry, bindings)
        key = _match_key(obj, hit)
        if (reason is not None or key in seen
                or (entry.kind != "client_factory" and key[:3] in factory_ranges)):
            stats["alias_hits_dropped"] += 1
            continue
        seen.add(key)
        hit.via_alias = True
        stats["alias_hits_kept"] += 1
        if entry.kind == "client_factory":
            factory_ranges.add(key[:3])
            # The plain callable rule may already have reported this same
            # range. A justified factory binding supersedes that external hit.
            overlapping = [h for h in hits if (h.file, h.start_line, h.start_col, h.end_line, h.end_col)
                           == (hit.file, hit.start_line, hit.start_col, hit.end_line, hit.end_col)]
            for old in overlapping:
                hits.remove(old)
            stats["hits"] -= len(overlapping)
            _record_factory(hit, report)
            stats["client_factory_hits"] += 1
        else:
            hits.append(hit)
            stats["hits"] += 1
    _bump(ctx, "ast_grep.hits", stats["hits"])
    _bump(ctx, "ast_grep.client_factory_hits", stats["client_factory_hits"])
    return stats


def _run_bucket(binary, parser_lang, ast_lang, files, root, ctx, report, hits, views, seen=None,
                factory_ranges=None):
    seen = set() if seen is None else seen
    factory_ranges = set() if factory_ranges is None else factory_ranges
    entries = [e for e in CATALOGUE if e.language == _FAMILY[parser_lang]]
    by_id = {e.id: e for e in entries}
    rules = _rules_text(entries, ast_lang)
    chunks = [files[i:i + FILES_PER_CALL] for i in range(0, len(files), FILES_PER_CALL)]
    started = time.monotonic()
    runs = failed = found = dropped = factory_hits = 0
    for chunk in chunks:
        runs += 1
        objs, failure = _run_one(binary, rules, chunk, root)
        if failure:
            failed += 1
            cause, detail = failure
            _note(report, ctx, "ast_grep_language_failed", cause, parser_lang,
                  f"--lang {ast_lang}: {detail}", samples=chunk)
            log(LOG, "error", "ast-grep run failed", language=parser_lang, ast_lang=ast_lang,
                cause=cause, error=detail, files=len(chunk))
            continue
        for obj in objs:
            hit, problem = _hit_from_json(obj, by_id, parser_lang, root, views)
            if hit is not None:
                key = _match_key(obj, hit)
                if key in seen:
                    continue
                seen.add(key)
                if by_id[hit.pattern_id].kind == "client_factory":
                    factory_ranges.add(key[:3])
                    factory_hits += 1
                    _record_factory(hit, report)
                    continue
                hits.append(hit)
                found += 1
            else:
                dropped += 1
                cause, detail, rel = problem
                _note(report, ctx, "ast_grep_language_failed", cause, parser_lang, detail, samples=[rel])
    seconds = time.monotonic() - started

    stats = report["languages"].setdefault(
        parser_lang, {"files": 0, "runs": 0, "hits": 0, "dropped": 0, "seconds": 0.0})
    stats["files"] += len(files)
    stats["runs"] += runs
    stats["hits"] += found
    stats["dropped"] += dropped
    stats["client_factory_hits"] = stats.get("client_factory_hits", 0) + factory_hits
    stats["seconds"] = round(stats["seconds"] + seconds, 3)
    _bump(ctx, "ast_grep.files", len(files))
    _bump(ctx, "ast_grep.runs", runs)
    _bump(ctx, "ast_grep.hits", found)
    _bump(ctx, "ast_grep.client_factory_hits", factory_hits)
    _bump(ctx, "ast_grep.failed_runs", failed)
    _bump(ctx, "ast_grep.dropped", dropped)
    log(LOG, "info", "ast-grep language done", language=parser_lang, ast_lang=ast_lang,
        files=len(files), rules=len(entries), runs=runs, hits=found, failed_runs=failed,
        dropped=dropped, seconds=round(seconds, 3))


def _run_patterns(root: Path, files_by_language, ctx, report, hits):
    binary, version, tried = find_binary()
    report["binary"], report["version"] = binary, version
    if binary is None:
        _note(report, ctx, "ast_grep_unavailable", "binary_missing", "",
              "no ast-grep executable found (" + "; ".join(tried) + ")")
        log(LOG, "error", "ast-grep not found; no pattern matching possible", tried=tried)
        return
    log(LOG, "info", "ast-grep binary", binary=binary, version=version)
    views = {}
    alias_names = {}
    seen = set()
    factory_ranges = set()
    for parser_lang, ast_lang, files in _buckets(root, files_by_language, report, ctx):
        bindings, discovery = _discover_bindings(binary, parser_lang, ast_lang, files, root,
                                                report, ctx, views)
        _run_bucket(binary, parser_lang, ast_lang, files, root, ctx, report, hits, views, seen, factory_ranges)
        objs, alias_stats = _run_alias_pass(binary, parser_lang, ast_lang, bindings, root, report, ctx)
        alias_counts = _process_alias_matches(objs, parser_lang, bindings, root, report, hits, views, seen,
                                              ctx, factory_ranges)
        stats = report["languages"][parser_lang]
        for key, value in alias_counts.items():
            stats[key] = stats.get(key, 0) + value
        for key in ("alias_files", "alias_rule_bytes", "alias_pass_launches"):
            stats[key] = stats.get(key, 0) + alias_stats[key]
        stats["alias_pass_seconds"] = round(stats.get("alias_pass_seconds", 0.0)
                                            + alias_stats["alias_pass_seconds"], 3)
        for key in ("files_scanned", "files_with_aliases", "discovery_runs", "discovery_skipped_files"):
            stats[key] = stats.get(key, 0) + discovery[key]
        stats["discovery_seconds"] = round(stats.get("discovery_seconds", 0.0)
                                           + discovery["discovery_seconds"], 3)
        names = alias_names.setdefault(parser_lang, set())
        names.update(b[0] for items in bindings.values() for b in items)
        stats["alias_names"] = len(names)
        stats["alias_discovery_seconds"] = stats["discovery_seconds"]


def run_patterns(repo_root, files_by_language, ctx=None):
    """Run the catalogue over `files_by_language` ({"python": [rel paths],
    "javascript": [...], "typescript": [...]}, paths repo-relative POSIX) and
    return (hits, run_report); see the module docstring for the report. Never
    raises. Hits are sorted by (file, start_line, start_col, pattern_id)."""
    report = {"binary": None, "version": "", "reasons": [], "languages": {}, "client_instances": []}
    hits = []
    try:
        _run_patterns(Path(repo_root), files_by_language or {}, ctx, report, hits)
    except Exception as exc:  # noqa: BLE001 - belt and braces; stages never raise
        detail = f"{type(exc).__name__}: {exc}"
        _note(report, ctx, "stage_failed", "exception", "", detail)
        log(LOG, "error", "ast-grep pattern stage raised", error=detail)
    hits.sort(key=lambda h: (h.file, h.start_line, h.start_col, h.pattern_id))
    report["client_instances"].sort(key=lambda r: r["file"])
    return hits, report
