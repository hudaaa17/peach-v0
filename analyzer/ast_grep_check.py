"""
Stage 3 (real): ast-grep pattern match.

`flag_external_calls` in external_calls.py used to decide "does this call
hit a known external-client API shape?" by running Python `re` patterns
against `edge.callee_expr` — a flattened, already-lossy string like
`"requests.get"` that parser.py built by walking dotted-name nodes (or,
for the regex-fallback languages, by a generic brace/indent scan). That's
a workaround with real, named failure modes:

  * It only ever sees the *tail* shape of the call target as text, so an
    aliased import (`import requests as r; r.get(url)`, `from axios
    import get as fetchGet`) silently doesn't match, even though the
    call is exactly the client call the pattern is meant to catch.
  * It has no idea whether `callee_expr` came from inside a string,
    comment, or docstring versus real call syntax — it's regex over text
    that already lost its AST, not a match against a real call node.
  * It can't express "this call, with this shape of arguments" or
    anything beyond a flat dotted-name string, so multi-line calls and
    calls the upstream extraction couldn't flatten cleanly are missed
    the same way an opaque `arg_text` is missed by the literal stage.

This module shells out to the real `ast-grep` toolchain
(https://github.com/ast-grep/ast-grep) — the actual structural search
tool the "ast-grep pattern match" stage name refers to — to match
against real parse trees instead of flattened strings:

  1. Each known external-client shape becomes a real ast-grep rule: a
     structural pattern (`requests.get($$$ARGS)`, `fetch($$$ARGS)`, ...)
     evaluated against the language's own grammar, the same way
     scip_check.py hands resolution to a real language server instead of
     a name-matching heuristic.
  2. Rules are written to a rule directory plus an `sgconfig.yml`
     (`ruleDirs: [...]`) and run in a single `ast-grep scan --json`
     pass over the repo — one subprocess + `json.loads`, the same shape
     every other real-tool integration in this pipeline uses (SCIP,
     Joern).
  3. Every match's file+start-line is recorded against the rule's label,
     giving `external_calls.py` a real, syntax-aware answer per call
     site instead of a boolean regex `.match()`.

ast-grep also happens to cover several languages no mainstream SCIP
indexer exists for (Ruby, PHP, C#), so this stage's coverage doesn't
depend on tier 1 of the resolver stage having succeeded.

Like scip_check.py, this degrades per language rather than per run:
`ast-grep` might not be installed, or a language's tree-sitter grammar
might not ship with this build of it. `external_calls.py` tries the
real scan first and only falls back to the `re`-based heuristic for
languages this module couldn't scan (not installed, grammar missing,
scan failed, or a language nothing in `EXTERNAL_RULES` covers yet) —
"degrade a level, don't go silent."
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from .obs import get_logger, log

LOG = get_logger("ast_grep_check")

SCAN_TIMEOUT = int(os.environ.get("PEACH_ASTGREP_SCAN_TIMEOUT", "300"))

# (label, [ast-grep patterns for that label]) per language. This is a
# structural mirror of EXTERNAL_PATTERNS in external_calls.py — same
# labels, same intent — but expressed as real call-shape patterns
# instead of a regex over a flattened dotted name. `$$$ARGS` matches
# zero-or-more arguments so call sites aren't missed over arg count.
EXTERNAL_RULES = {
    "python": [
        ("python-requests", [
            "requests.get($$$ARGS)", "requests.post($$$ARGS)",
            "requests.put($$$ARGS)", "requests.delete($$$ARGS)",
            "requests.patch($$$ARGS)", "requests.head($$$ARGS)",
            "requests.request($$$ARGS)", "requests.Session($$$ARGS)",
        ]),
        ("python-httpx", [
            "httpx.get($$$ARGS)", "httpx.post($$$ARGS)", "httpx.put($$$ARGS)",
            "httpx.delete($$$ARGS)", "httpx.patch($$$ARGS)", "httpx.request($$$ARGS)",
            "httpx.Client($$$ARGS)", "httpx.AsyncClient($$$ARGS)",
        ]),
        ("python-urllib", ["urllib.request.urlopen($$$ARGS)"]),
        ("python-http.client", [
            "http.client.HTTPConnection($$$ARGS)", "http.client.HTTPSConnection($$$ARGS)",
        ]),
        ("python-aiohttp", ["aiohttp.ClientSession($$$ARGS)"]),
        ("aws-sdk", ["boto3.client($$$ARGS)"]),
    ],
    "javascript": [
        ("js-fetch", ["fetch($$$ARGS)"]),
        ("js-axios", [
            "axios.get($$$ARGS)", "axios.post($$$ARGS)", "axios.put($$$ARGS)",
            "axios.delete($$$ARGS)", "axios.patch($$$ARGS)", "axios.request($$$ARGS)",
            "axios($$$ARGS)",
        ]),
        ("js-jquery", ["$.ajax($$$ARGS)"]),
        ("node-http", [
            "http.request($$$ARGS)", "https.request($$$ARGS)",
            "http.get($$$ARGS)", "https.get($$$ARGS)",
        ]),
        ("js-superagent", [
            "superagent.get($$$ARGS)", "superagent.post($$$ARGS)",
            "superagent.put($$$ARGS)", "superagent.delete($$$ARGS)",
        ]),
        ("grpc-client", ["new grpc.Client($$$ARGS)"]),
    ],
    "typescript": [
        ("js-fetch", ["fetch($$$ARGS)"]),
        ("js-axios", [
            "axios.get($$$ARGS)", "axios.post($$$ARGS)", "axios.put($$$ARGS)",
            "axios.delete($$$ARGS)", "axios.patch($$$ARGS)", "axios.request($$$ARGS)",
            "axios($$$ARGS)",
        ]),
        ("node-http", [
            "http.request($$$ARGS)", "https.request($$$ARGS)",
            "http.get($$$ARGS)", "https.get($$$ARGS)",
        ]),
        ("js-superagent", [
            "superagent.get($$$ARGS)", "superagent.post($$$ARGS)",
            "superagent.put($$$ARGS)", "superagent.delete($$$ARGS)",
        ]),
        ("grpc-client", ["new grpc.Client($$$ARGS)"]),
    ],
    "go": [
        ("go-net-http", [
            "http.Get($$$ARGS)", "http.Post($$$ARGS)", "http.NewRequest($$$ARGS)",
        ]),
        ("go-resty", ["resty.New($$$ARGS)"]),
    ],
    "java": [
        ("java-http-client", [
            "new HttpClient($$$ARGS)", "new RestTemplate($$$ARGS)", "new OkHttpClient($$$ARGS)",
        ]),
        ("grpc-client", ["new grpc.Client($$$ARGS)"]),
    ],
    "ruby": [
        ("ruby-net-http", [
            "Net::HTTP.get($$$ARGS)", "Net::HTTP.post($$$ARGS)",
            "Net::HTTP.start($$$ARGS)", "Net::HTTP.new($$$ARGS)",
        ]),
    ],
    # PHP and C# have no SCIP indexer at all (see scip_check.py), but
    # ast-grep ships grammars for both, so this stage's coverage of them
    # doesn't depend on tier 1 of the resolver stage succeeding.
    "php": [
        ("php-curl", ["curl_init($$$ARGS)"]),
        ("php-guzzle", ["new Client($$$ARGS)"]),
    ],
    "csharp": [
        ("java-http-client", ["new HttpClient($$$ARGS)", "new RestClient($$$ARGS)"]),
    ],
}

_lock = threading.Lock()
_bin_cache = {}
_astgrep_available = None  # None = unchecked, True/False once probed


def _find_binary(name):
    if name in _bin_cache:
        return _bin_cache[name]
    astgrep_home = os.environ.get("ASTGREP_HOME")
    found = None
    if astgrep_home:
        candidate = Path(astgrep_home) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found = str(candidate)
    if not found:
        found = shutil.which(name)
    with _lock:
        _bin_cache[name] = found
    return found


def _check_astgrep_available():
    global _astgrep_available
    if _astgrep_available is None:
        _astgrep_available = _find_binary("ast-grep") is not None
    return _astgrep_available


def _write_rules(languages, workdir: Path):
    """One rule file per label, keyed by a generated rule id, plus an
    sgconfig.yml pointing `ruleDirs` at them — a single `ast-grep scan`
    then covers every language and every label in one pass. Returns
    {rule_id: label} so scan output can be mapped back to a label."""
    rules_dir = workdir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rule_id_to_label = {}

    for language in languages:
        for label, patterns in EXTERNAL_RULES.get(language, []):
            rule_id = f"ext-{language}-{label}-{uuid.uuid4().hex[:8]}"
            rule_id_to_label[rule_id] = label
            any_patterns = "\n".join(f"      - pattern: {p}" for p in patterns)
            rule_yaml = (
                f"id: {rule_id}\n"
                f"language: {language}\n"
                f"message: {label}\n"
                f"severity: info\n"
                f"rule:\n"
                f"  any:\n"
                f"{any_patterns}\n"
            )
            (rules_dir / f"{rule_id}.yml").write_text(rule_yaml, encoding="utf-8")

    sgconfig = workdir / "sgconfig.yml"
    sgconfig.write_text(f"ruleDirs:\n  - {rules_dir}\n", encoding="utf-8")
    return sgconfig, rule_id_to_label


def _run_scan(sgconfig: Path, repo_root: Path):
    astgrep_bin = _find_binary("ast-grep")
    cmd = [astgrep_bin, "scan", "--json", "-c", str(sgconfig), str(repo_root)]
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=SCAN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"ast-grep scan timed out after {SCAN_TIMEOUT}s"
    except OSError as exc:
        return None, f"failed to launch ast-grep: {exc}"
    # ast-grep exits non-zero when rules match (that's its "lint found
    # issues" convention) — only treat it as a real failure if we got no
    # parseable stdout at all.
    if not proc.stdout.strip():
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            detail = tail[-1][:200] if tail else "no output"
            return None, f"ast-grep scan failed (exit {proc.returncode}): {detail}"
        return [], None

    text = proc.stdout.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, None
    except json.JSONDecodeError:
        pass

    matches = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            matches.append(json.loads(line))
        except json.JSONDecodeError:
            return None, "could not parse `ast-grep scan --json` output as JSON or NDJSON"
    return matches, None


def _match_line(match):
    """ast-grep JSON matches carry a 0-indexed `range.start.line`; +1 to
    line up with our 1-indexed CallSite/CallEdge line numbers (the same
    conversion scip_check.py's `_occurrence_line` does)."""
    rng = match.get("range") or {}
    start = rng.get("start") or {}
    line = start.get("line")
    return None if line is None else line + 1


def build_external_match_map(records, repo_root, ctx=None):
    """Runs a single real `ast-grep scan` covering every language present
    in `records` that `EXTERNAL_RULES` has patterns for, and returns
    `(match_map, status)`:

    - `match_map`: `{(file, line): label}` for every call site ast-grep
      matched to a known external-client shape.
    - `status`: which languages got scanned and which were skipped (and
      why), so `external_calls.py` can say *why* a call fell back to the
      `re` heuristic instead of quietly losing precision.
    """
    languages = sorted({r.language for r in records if not r.is_config})
    status = {"scanned": [], "skipped": {}, "astgrep_available": _check_astgrep_available()}
    match_map = {}

    def _finish():
        if ctx is not None:
            ctx.bump("external.astgrep.languages_scanned", len(status["scanned"]))
            ctx.bump("external.astgrep.languages_skipped", len(status["skipped"]))
            ctx.bump("external.astgrep.matches", len(match_map))
            if status["skipped"]:
                ctx.note("info", "external",
                         "ast-grep pattern match unavailable for: " +
                         ", ".join(f"{lang} ({reason})" for lang, reason in status["skipped"].items()) +
                         ". Those calls use the regex fallback instead.")
        return match_map, status

    if not status["astgrep_available"]:
        status["skipped"]["*"] = (
            "`ast-grep` CLI not found on $PATH/$ASTGREP_HOME; install it from "
            "https://github.com/ast-grep/ast-grep"
        )
        return _finish()

    coverable = [l for l in languages if l in EXTERNAL_RULES]
    uncovered = [l for l in languages if l not in EXTERNAL_RULES]
    for lang in uncovered:
        status["skipped"][lang] = "no external-call rules authored for this language yet"
    if not coverable:
        return _finish()

    workdir = Path(tempfile.mkdtemp(prefix="peach_astgrep_"))
    try:
        sgconfig, rule_id_to_label = _write_rules(coverable, workdir)
        if not rule_id_to_label:
            return _finish()

        matches, err = _run_scan(sgconfig, Path(repo_root))
        if err:
            for lang in coverable:
                status["skipped"][lang] = err
            log(LOG, "warning", "ast-grep scan failed for repo", reason=err)
            return _finish()

        for m in matches:
            rule_id = m.get("ruleId") or m.get("rule_id")
            label = rule_id_to_label.get(rule_id)
            if not label:
                continue
            rel_path = m.get("file")
            line = _match_line(m)
            if rel_path and line is not None:
                match_map[(rel_path, line)] = label

        status["scanned"] = coverable
        log(LOG, "info", "ast-grep scan complete", languages=coverable,
            rules=len(rule_id_to_label), matches=len(match_map))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return _finish()