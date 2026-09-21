"""
Stage 7 (real): Joern escalation — inter-procedural data-flow and
control-flow extraction over a real Code Property Graph.

This stage used to be four hand-written regex "traces" (`self.attr`
assignment lines, module-level `NAME = "..."` lines, `os.environ.get`
lines, and a positional guess at what a caller passed) with a docstring
that said outright: *"We don't ship the Joern binary here ... but we play
the same role."* It didn't play the same role. A line-anchored regex has
no notion of a def-use chain, a call graph, a branch, or an assignment
that happens anywhere other than the start of a line, so the stage named
after inter-procedural data flow was the one stage in the pipeline doing
no data-flow analysis at all.

This module runs the real thing:

  1. `joern-parse` builds a Code Property Graph (AST + CFG + PDG + call
     graph) for the repo, once per language present, using that
     language's real Joern frontend (`pysrc2cpg`, `jssrc2cpg`,
     `javasrc2cpg`, `gosrc2cpg`, `phpparser`, `rubysrc2cpg`,
     `csharpsrc2cpg`).
  2. `joern --script` runs `_QUERY_SCRIPT` (CPGQL/Scala) against that
     CPG. The script applies the OSS data-flow layer (`run.ossdataflow`)
     and, for each escalated call site, asks `reachableByFlows` which
     sources actually reach the first argument of that call. That is a
     real inter-procedural, flow-sensitive query over the PDG: it
     crosses functions and files by construction, follows assignments
     through intermediate variables, string building and containers, and
     needs no naming heuristic to decide whether a parameter is "endpoint
     shaped" — either the value flows there or it doesn't.
  3. The same script reads *control* flow for each result:
     `controlledBy.isControlStructure` gives the branch conditions that
     dominate the call, so a host that is only reached under
     `if env == "prod"` is reported as conditional instead of as the
     answer.
  4. Python side (`_interpret`) only *classifies* what Joern returned —
     literal source, environment-variable source, or a source outside the
     analysed code (a parameter/unknown). It does no tracing of its own.

There is no fallback tier here, by design. Every other stage in this
pipeline degrades to a cheaper technique, because a cheaper technique
still answers the same question less precisely. There is no cheap version
of inter-procedural data flow — the regex traces that used to live here
produced *different, wrong* answers, not less precise ones — so when
Joern can't run (not installed, no frontend for a language, parse failed,
query timed out) this stage resolves nothing and says loudly why, in the
logs, the counters and a `ctx.note`. A silent miss and an unavailable
analysis are never conflated.

Requirements
------------
* `joern` and `joern-parse` on `$PATH`, or `JOERN_HOME` pointing at the
  Joern install directory (https://joern.io).
* `repo_root` must be passed in by the pipeline — a CPG is built from
  source on disk, not from the parsed records.

Tunables (environment):
  PEACH_JOERN_PARSE_TIMEOUT   per-language CPG build timeout (default 900s)
  PEACH_JOERN_QUERY_TIMEOUT   per-language query timeout      (default 600s)
  PEACH_JOERN_MAX_HEAP        JVM heap for both               (default 4g)
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import signal      
import threading
import time


from .argtext import first_argument, classify
from .literal_check import _host_from_literal
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("joern")

PARSE_TIMEOUT = int(os.environ.get("PEACH_JOERN_PARSE_TIMEOUT", "900"))
QUERY_TIMEOUT = int(os.environ.get("PEACH_JOERN_QUERY_TIMEOUT", "600"))
MAX_HEAP = os.environ.get("PEACH_JOERN_MAX_HEAP", "4g")

#: Our language names -> the Joern frontend language id understood by
#: `joern-parse --language`. Languages with no Joern frontend are absent
#: and are reported as skipped rather than quietly handled some other way.
_JOERN_LANGUAGES = {
    "python": "PYTHONSRC",
    "javascript": "JSSRC",
    "typescript": "JSSRC",
    "java": "JAVASRC",
    "go": "GOLANG",
    "php": "PHP",
    "ruby": "RUBYSRC",
    "csharp": "CSHARP",
    # rust: no Joern frontend at the time of writing.
}

#: Extracts the variable name from an env-lookup *node Joern already
#: identified as the source of the flow*. This is not a trace — the
#: data-flow question was answered by the CPG query; this only reads the
#: key out of the source node's own source text.
_ENV_KEY_RE = re.compile(
    r'''(?:os\.environ\.get|os\.getenv|System\.getenv|os\.Getenv|getenv|Environment\.GetEnvironmentVariable)'''
    r'''\s*\(\s*['"]([\w.\-]+)['"]'''
    r'''|os\.environ\s*\[\s*['"]([\w.\-]+)['"]\s*\]'''
    r'''|process\.env\.([A-Za-z_]\w*)'''
    r'''|process\.env\s*\[\s*['"]([\w.\-]+)['"]\s*\]'''
    r'''|ENV\s*\[\s*['"]([\w.\-]+)['"]\s*\]''',
)

#: A value we're willing to call a network destination. Joern tells us a
#: literal reaches the call; this decides whether that literal is a host
#: or an unrelated string (a header name, a log message, a JSON key).
_PLAUSIBLE_HOST = re.compile(
    r"""^(?:
          [a-zA-Z][\w+.\-]*://.+                 # has a scheme
        | [A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+.*  # dotted name / domain
        | [A-Za-z0-9\-]+:\d{2,5}(?:/.*)?         # host:port
        | (?:localhost|127\.0\.0\.1)(?::\d+)?.*
        )$""",
    re.VERBOSE,
)


# --------------------------------------------------------- the CPG query ----

_QUERY_SCRIPT = r'''
// Peach stage 7: inter-procedural data-flow + control-flow extraction.
//
// For every (file, line) sink handed in, find the call at that location,
// take its first argument, and ask the data-flow engine which literal /
// environment / parameter sources reach it. Also record the control
// structures that dominate the call, so a branch-dependent host is
// reported as branch-dependent instead of as fact.
import io.shiftleft.codepropertygraph.generated.nodes.{AstNode, CfgNode}
import scala.util.Try

// These three are substituted by _run_query() before the script is
// written to disk (see PEACH_CPG_FILE / PEACH_SINKS_FILE / PEACH_OUT_FILE
// sentinels below) — baked in as literals rather than passed via
// `--param`, because this installed Joern version's `--param` parser
// rejects every key we tried (`Option --param failed when given ...`),
// which left cpgFile/sinksFile/outFile unbound and crashed the script
// with `NoSuchElementException: None.get` before it ever touched the
// CPG. A literal in the script body has no option-parser to disagree
// with.
// Peach stage 7: inter-procedural data-flow + control-flow extraction.
import io.shiftleft.codepropertygraph.generated.nodes.{AstNode, CfgNode}
import scala.util.Try

val cpgFile: String = "PEACH_CPG_FILE"
val sinksFile: String = "PEACH_SINKS_FILE"
val outFile: String = "PEACH_OUT_FILE"

// --- PROGRESS MARKERS: These printlns are streamed to the Python logger 
// so you can see exactly where Joern freezes if it hangs.
println("peach-joern: importing CPG...")
importCpg(cpgFile)

println("peach-joern: CPG imported, running ossdataflow (this can take a while)...")
Try(run.ossdataflow)   

println("peach-joern: ossdataflow complete, starting sink queries...")

def shorten(s: String): String = {
    val t = if (s == null) "" else s.replaceAll("\\s+", " ").trim
    if (t.length > 300) t.take(300) + "..." else t
}
def norm(s: String): String = if (s == null) "" else s.replace("\\", "/")
def fileOf(n: AstNode): String = Try(norm(n.location.filename)).getOrElse("")
def lineOf(n: AstNode): Int = Try(n.lineNumber.map(_.toInt).getOrElse(-1)).getOrElse(-1)
def describe(n: AstNode) = ujson.Obj(
    "code" -> shorten(n.code), "file" -> fileOf(n), "line" -> lineOf(n),
    "method" -> Try(n.location.methodFullName).getOrElse("")
)
def conditionsOf(n: AstNode): List[String] =
    Try(n.asInstanceOf[CfgNode].controlledBy.isControlStructure.map(cs => shorten(cs.code)).l)
    .getOrElse(Nil).distinct

def literalSources: Iterator[CfgNode] = cpg.literal
def envSources: Iterator[CfgNode] = cpg.call.filter { c =>
    val cd = Try(c.code.replaceAll("\\s+", " ")).getOrElse("")
    cd.contains("os.environ") || cd.contains("os.getenv(") ||
    cd.contains("process.env") || cd.contains("System.getenv(") ||
    cd.contains("os.Getenv(") || cd.contains("Environment.GetEnvironmentVariable(") ||
    cd.startsWith("ENV[")
}
def paramSources: Iterator[CfgNode] = cpg.method.parameter

val spec = ujson.read(os.read(os.Path(sinksFile))).arr
val totalSinks = spec.size
println(s"peach-joern: querying $totalSinks sink(s)...")

val results = spec.zipWithIndex.map { case (s, idx) =>
    if (idx % 10 == 0) println(s"peach-joern: query progress $idx / $totalSinks")
    val path = s("file").str
    val line = s("line").num.toInt
    val calls = cpg.call.filter { c =>
        lineOf(c) == line && {
            val f = fileOf(c)
            f.nonEmpty && (f.endsWith(path) || path.endsWith(f))
        }
    }.l
    val args = calls.flatMap(c => Try(c.argument.argumentIndex(1).l).getOrElse(Nil))
    val sinkNodes: List[CfgNode] =
        (if (args.nonEmpty) args.map(_.asInstanceOf[CfgNode]) else calls.map(_.asInstanceOf[CfgNode]))

    def flowsFrom(sources: => Iterator[CfgNode], kind: String): List[ujson.Obj] =
        Try {
            sinkNodes.iterator.reachableByFlows(sources).take(40).l.map { p =>
                val els = p.elements
                ujson.Obj(
                    "kind" -> kind, "source" -> describe(els.head), "sink" -> describe(els.last),
                    "steps" -> ujson.Arr.from(els.map(describe)),
                    "conditions" -> ujson.Arr.from((conditionsOf(els.head) ++ conditionsOf(els.last)).distinct)
                )
            }
        }.getOrElse(Nil)

    val flows =
        flowsFrom(literalSources, "literal") ++
        flowsFrom(envSources, "env") ++
        (if (calls.isEmpty) Nil else flowsFrom(paramSources, "parameter"))

    ujson.Obj(
        "file" -> path, "line" -> line, "found_call" -> calls.nonEmpty,
        "found_argument" -> args.nonEmpty, "call_code" -> (if (calls.isEmpty) "" else shorten(calls.head.code)),
        "method" -> (if (calls.isEmpty) "" else Try(calls.head.method.fullName).getOrElse("")),
        "conditions" -> ujson.Arr.from(calls.headOption.map(c => conditionsOf(c)).getOrElse(Nil)),
        "flows" -> ujson.Arr.from(flows)
    )
}

os.write.over(os.Path(outFile), ujson.write(ujson.Arr.from(results)))
println(s"peach-joern: wrote ${results.size} sink result(s)")
'''


# ------------------------------------------------------------ subprocess ----

_bin_cache = {}


def _find_binary(name):
    if name in _bin_cache:
        return _bin_cache[name]
    found = None
    joern_home = os.environ.get("JOERN_HOME")
    if joern_home:
        for candidate in (Path(joern_home) / name, Path(joern_home) / "bin" / name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                found = str(candidate)
                break
    if not found:
        found = shutil.which(name)
    _bin_cache[name] = found
    return found


def _jvm_env():
    env = dict(os.environ)
    opts = env.get("JAVA_OPTS", "")
    if "-Xmx" not in opts:
        env["JAVA_OPTS"] = (opts + f" -Xmx{MAX_HEAP}").strip()
    return env


#: Matches the line(s) in JVM stderr that actually say what went wrong.
#: A JVM stack trace prints the exception type + message *first*, then
#: "at ..." frames walking outward, ending with the entry point last
#: (for `joern`/`joern-parse` that's always `ReplBridge.main`, or the
#: process launcher — never the cause). Taking the last line therefore
#: reliably returns the one line that is guaranteed *not* to be the
#: cause. This regex instead pulls out exception/error/"Caused by"
#: lines, wherever they are, plus any compiler-error banner.
_ERROR_LINE_RE = re.compile(
    r"""^(?:.*\bException\b.*|.*\bError\b:.*|Caused\ by:.*|error:.*|
          .*Compilation\ Failed.*)""",
    re.VERBOSE,
)


def _extract_error(output: str, limit: int = 800) -> str:
    """Best-effort pull of the actually-useful line(s) out of a failed
    subprocess's stderr/stdout, instead of blindly taking the last line
    (which for a JVM stack trace is boilerplate, not the cause)."""
    text = (output or "").strip()
    if not text:
        return "no output"
    lines = text.splitlines()
    hits = [l.strip() for l in lines if _ERROR_LINE_RE.match(l.strip())]
    if hits:
        # Keep order of first appearance, dedupe, cap how much we log.
        seen = []
        for h in hits:
            if h not in seen:
                seen.append(h)
        return " | ".join(seen)[:limit]
    # No recognizable exception line at all (e.g. a plain non-JVM
    # failure) — fall back to the first non-empty line, which for
    # ordinary CLI errors *is* usually the message, unlike the last
    # line of a stack trace.
    return lines[0].strip()[:limit]


def _scala_string_literal(path) -> str:
    """Render a filesystem path as a Scala double-quoted string literal.
    Paths are attacker-uncontrolled (they're our own tempdir/output
    names), but escaping backslashes and quotes properly still matters on
    Windows workdirs and matches what a literal is actually supposed to
    mean here — this isn't string interpolation, just embedding a value
    the CLI's own option parser can no longer mangle."""
    s = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _render_query_script(cpg_path: Path, sinks_path: Path, out_path: Path) -> str:
    """Bake the three run-specific paths into `_QUERY_SCRIPT` as literals.
    Each call gets its own rendered script — cheap, since the script is
    only a few KB and is already written to disk per run."""
    return (
        _QUERY_SCRIPT
        .replace('"PEACH_CPG_FILE"', _scala_string_literal(cpg_path))
        .replace('"PEACH_SINKS_FILE"', _scala_string_literal(sinks_path))
        .replace('"PEACH_OUT_FILE"', _scala_string_literal(out_path))
    )


# ... (keep _extract_error, _scala_string_literal, _render_query_script as they are) ...

def _run_subprocess(cmd, cwd, env, timeout, label):
    """Run a subprocess, streaming output to logs and enforcing a hard timeout.
    Kills the entire process tree on timeout to prevent zombie JVM hangs."""
    log(LOG, "info", f"{label} starting", cmd=" ".join(cmd), cwd=cwd, timeout=timeout)
    try:
        kwargs = {}
        if os.name != 'nt':
            kwargs['start_new_session'] = True  # Allows killing the whole process tree
            
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **kwargs
        )
    except OSError as exc:
        log(LOG, "error", f"{label} failed to launch", error=str(exc))
        return None, f"failed to launch {label}: {exc}", ""

    log(LOG, "info", f"{label} running", pid=proc.pid)
    
    output_lines = []
    timed_out = False
    
    def reader():
        try:
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                # Stream our custom progress markers directly to INFO
                if line.startswith("peach-joern:"):
                    log(LOG, "info", f"{label} progress", msg=line)
                elif "Exception" in line or "Error" in line or "Caused by" in line:
                    log(LOG, "warning", f"{label} jvm-error", msg=line)
                elif TRACE_EDGES:
                    log(LOG, "debug", f"{label} debug", msg=line)
        except Exception:
            pass

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    
    start_time = time.time()
    while proc.poll() is None:
        if time.time() - start_time > timeout:
            log(LOG, "error", f"{label} timed out after {timeout}s, killing process tree", pid=proc.pid)
            timed_out = True
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                try: proc.kill()
                except Exception: pass
            break
        time.sleep(1)
        
    proc.wait()
    reader_thread.join(timeout=5)
    
    full_output = "\n".join(output_lines)
    
    if timed_out:
        return None, f"{label} timed out after {timeout}s", full_output
    if proc.returncode != 0:
        return None, f"{label} failed (exit {proc.returncode})", full_output
        
    return full_output, None, full_output

def _build_cpg(language, joern_language, repo_root: Path, workdir: Path):
    joern_parse = _find_binary("joern-parse")
    out_path = workdir / f"cpg-{language}.bin"
    cmd = [joern_parse, str(repo_root), "--language", joern_language, "--output", str(out_path)]
    
    full_output, err, _ = _run_subprocess(cmd, str(repo_root), _jvm_env(), PARSE_TIMEOUT, f"joern-parse ({language})")
    
    if err:
        detail = _extract_error(full_output)
        log(LOG, "error", "CPG construction failed", language=language, reason=err, detail=detail)
        return None, f"joern-parse failed: {detail}"
    if not out_path.exists():
        return None, f"joern-parse finished but output file {out_path} was not created"
    
    log(LOG, "info", "CPG built successfully", language=language, size=out_path.stat().st_size)
    return out_path, None

def _run_query(cpg_path: Path, sinks, workdir: Path, tag: str):
    joern = _find_binary("joern")
    sinks_path = workdir / f"sinks-{tag}.json"
    out_path = workdir / f"flows-{tag}.json"
    script_path = workdir / f"peach_dataflow-{tag}.sc"
    
    sinks_path.write_text(json.dumps(sinks), encoding="utf-8")
    script_path.write_text(_render_query_script(cpg_path, sinks_path, out_path), encoding="utf-8")
    
    cmd = [joern, "--script", str(script_path)]
    full_output, err, _ = _run_subprocess(cmd, str(workdir), _jvm_env(), QUERY_TIMEOUT, f"joern-query ({tag})")
    
    if err:
        detail = _extract_error(full_output)
        log(LOG, "error", "data-flow query failed", tag=tag, reason=err, detail=detail)
        return None, f"joern query failed: {detail}"
    if not out_path.exists():
        return None, f"joern query finished but output file {out_path} was not created"
    
    try:
        return json.loads(out_path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read query output: {exc}"


def run_dataflow(records_by_path, sink_edges, repo_root, ctx=None):
    """Build a CPG per language and query it for every escalated sink.

    Returns `(flows_by_location, status)` where `flows_by_location` maps
    `(file, line)` to the raw query result, and `status` records which
    languages were analysed and which were skipped and why — so "resolved
    nothing" can always be told apart from "never ran"."""
    status = {"analyzed": [], "skipped": {}, "joern_available": True}

    for binary in ("joern-parse", "joern"):
        if not _find_binary(binary):
            status["joern_available"] = False
            status["skipped"]["*"] = (
                f"`{binary}` not found on $PATH/$JOERN_HOME. This stage performs "
                "inter-procedural data-flow analysis on a Joern CPG and has no "
                "substitute for it; install Joern from https://joern.io"
            )
            return {}, status

    if repo_root is None:
        status["joern_available"] = False
        status["skipped"]["*"] = ("repo_root was not passed to this stage; a CPG is built "
                                  "from source on disk and cannot be built without it")
        return {}, status

    # Group sinks by the language of the file they live in: each Joern
    # frontend produces its own CPG, so a sink is only queryable against
    # the CPG built for its own language.
    by_language = defaultdict(list)
    for edge in sink_edges:
        rec = records_by_path.get(edge.file)
        language = rec.language if rec else None
        by_language[language].append({"file": edge.file, "line": edge.line})

    flows = {}
    workdir = Path(tempfile.mkdtemp(prefix="peach_joern_"))
    try:
        for language, sinks in sorted(by_language.items(), key=lambda kv: str(kv[0])):
            joern_language = _JOERN_LANGUAGES.get(language)
            if not joern_language:
                status["skipped"][language or "unknown"] = (
                    f"no Joern frontend for language {language!r} "
                    f"({len(sinks)} call site(s) not analysed)")
                continue

            cpg_path, err = _build_cpg(language, joern_language, Path(repo_root), workdir)
            if err:
                status["skipped"][language] = err
                log(LOG, "error", "CPG construction failed", language=language, reason=err)
                continue

            results, err = _run_query(cpg_path, sinks, workdir, tag=language)
            if err:
                status["skipped"][language] = err
                log(LOG, "error", "data-flow query failed", language=language, reason=err)
                continue

            for res in results or []:
                flows[(res.get("file"), res.get("line"))] = res
            status["analyzed"].append(language)
            log(LOG, "info", "CPG data-flow query complete", language=language,
                sinks=len(sinks), results=len(results or []))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if ctx is not None:
        ctx.bump("joern.languages_analyzed", len(status["analyzed"]))
        ctx.bump("joern.languages_skipped", len(status["skipped"]))
    return flows, status


# -------------------------------------------------------- interpretation ----

def _is_plausible_host(value: str) -> bool:
    value = (value or "").strip()
    if len(value) < 4 or " " in value:
        return False
    return bool(_PLAUSIBLE_HOST.match(value))


def _literal_text(code: str):
    """Source text of a literal node -> its string value, or None if the
    node isn't a string literal (a number, a bool, a Scala-side oddity)."""
    if code is None:
        return None
    text = code.strip()
    for prefix in ("f", "r", "b", "u", "rb", "br", "fr", "rf"):
        if text[:len(prefix)].lower() == prefix and len(text) > len(prefix) and text[len(prefix)] in "\"'":
            text = text[len(prefix):]
            break
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1]
    return None


def _env_key(code: str):
    m = _ENV_KEY_RE.search(code or "")
    if not m:
        return None
    return next((g for g in m.groups() if g), None)


class FlowOutcome:
    """What the CPG query produced for one edge, and what we made of it.

    `analyzed` separates "Joern looked at this call site" from "Joern
    never saw it" — the same distinction the old `TraceResult.ran` flag
    carried, kept because it is the only thing that makes a zero in this
    stage interpretable."""

    __slots__ = ("analyzed", "resolved", "kind", "reason", "detail", "conditions")

    def __init__(self, analyzed, resolved=False, kind=None, reason=None,
                 detail=None, conditions=None):
        self.analyzed = analyzed
        self.resolved = resolved
        self.kind = kind
        self.reason = reason
        self.detail = detail
        self.conditions = conditions or []

    def as_dict(self):
        return {"analyzed": self.analyzed, "resolved": self.resolved,
                "source_kind": self.kind, "reason": self.reason,
                "detail": self.detail, "conditions": self.conditions}


def _interpret(edge, result):
    """Classify the flows Joern returned for one call site and, if one of
    them carries a host, write it onto the edge. No tracing happens here:
    every fact used comes from the CPG query."""
    if result is None:
        return FlowOutcome(analyzed=False, reason="call_site_not_present_in_cpg")
    if not result.get("found_call"):
        return FlowOutcome(analyzed=False, reason="no_call_node_at_this_location")
    if not result.get("found_argument"):
        return FlowOutcome(analyzed=True, reason="call_has_no_first_argument_node")

    flows = result.get("flows") or []
    if not flows:
        return FlowOutcome(analyzed=True, reason="no_source_reaches_this_argument")

    sink_conditions = result.get("conditions") or []

    # --- literal sources: a string constant that actually flows here ---
    hosts = []          # (host, literal, flow)
    for flow in flows:
        if flow.get("kind") != "literal":
            continue
        value = _literal_text((flow.get("source") or {}).get("code"))
        if value is None:
            continue
        host = _host_from_literal(value)
        if not host and not _is_plausible_host(value):
            continue
        hosts.append((host or value, value, flow))

    if hosts:
        distinct = sorted({h for h, _, _ in hosts})
        host, value, flow = hosts[0]
        edge.literal = value
        edge.literal_method = "joern"
        edge.host = host
        source = flow.get("source") or {}
        conditions = (flow.get("conditions") or []) + sink_conditions
        _attach(edge, flow, "literal", conditions)

        # Control flow is part of the answer: a host that is only reached
        # under a branch, or a call site several literals reach, is a
        # conditional result, not a fact.
        if len(distinct) > 1:
            edge.needs_review = True
            edge.review_reason = (
                f"{len(distinct)} different literals reach this call argument "
                f"({', '.join(distinct[:4])}); the CPG says the destination is "
                "branch- or caller-dependent. Reported host is one of them.")
        elif conditions:
            edge.needs_review = True
            edge.review_reason = (
                f"Host '{host}' flows from {source.get('file')}:{source.get('line')} "
                f"but the call is control-dependent on: {'; '.join(conditions[:3])}. "
                "Other branches may reach a different destination.")
        return FlowOutcome(
            analyzed=True, resolved=True, kind="literal", conditions=conditions,
            detail=f"literal at {source.get('file')}:{source.get('line')} -> {host}")

    # --- environment sources: the value isn't in the repo, the key is ---
    for flow in flows:
        if flow.get("kind") != "env":
            continue
        key = _env_key((flow.get("source") or {}).get("code"))
        if not key:
            continue
        source = flow.get("source") or {}
        edge.literal = key
        edge.literal_method = "joern"
        edge.host = key           # matched against manifests by the config stage
        _attach(edge, flow, "env", (flow.get("conditions") or []) + sink_conditions)
        return FlowOutcome(
            analyzed=True, resolved=True, kind="env",
            conditions=flow.get("conditions") or [],
            detail=f"env var {key} read at {source.get('file')}:{source.get('line')}")

    # --- everything else: the value enters from outside the analysed code ---
    params = [f for f in flows if f.get("kind") == "parameter"]
    if params:
        source = params[0].get("source") or {}
        return FlowOutcome(
            analyzed=True, reason="value_enters_from_an_unanalysed_caller",
            detail=(f"reaches the call from parameter {source.get('code')!r} of "
                    f"{source.get('method')} with no literal behind it"))
    return FlowOutcome(analyzed=True, reason="sources_reach_the_call_but_none_is_a_host")


def _attach(edge, flow, kind, conditions):
    """Hang the evidence off the edge. CallEdge is a plain dataclass, so
    these are readable by the graph/report writers; add them to the
    dataclass in resolver.py if you want them declared."""
    edge.joern_resolved = True
    setattr(edge, "joern_source_kind", kind)
    setattr(edge, "joern_source", flow.get("source"))
    setattr(edge, "joern_flow_steps", flow.get("steps") or [])
    setattr(edge, "joern_conditions", list(conditions))


# ------------------------------------------------------------- the stage ----

def check_joern_escalation(edges, records_by_path, all_records, ctx=None, repo_root=None):
    """Mutates edges in place. Returns the list of edges escalated to this
    stage (whether or not the CPG query resolved a host).

    `repo_root` is required: this stage builds a real CPG from source.
    Called without it, or without Joern installed, it resolves nothing and
    reports why — it does not substitute a weaker technique."""
    escalated = [e for e in edges
                 if e.status == "external_candidate" and not e.host]
    for edge in escalated:
        edge.escalated_to_joern = True
    if not escalated:
        log(LOG, "info", "joern escalation complete", escalated=0, resolved=0,
            note="no unresolved external candidates reached this stage")
        return []

    flows, status = run_dataflow(records_by_path, escalated, repo_root, ctx=ctx)

    if not status["joern_available"]:
        reason = status["skipped"].get("*", "Joern unavailable")
        log(LOG, "error", "Joern is unavailable; stage 7 performed no analysis",
            escalated=len(escalated), reason=reason)
        if ctx is not None:
            ctx.bump("joern.escalated", len(escalated))
            ctx.bump("joern.unavailable", len(escalated))
            ctx.note("error", "joern",
                     f"{len(escalated)} call site(s) needed inter-procedural data-flow "
                     f"analysis and did not get it: {reason}. Those calls were passed to "
                     "the SLM fallback unanalysed — their hosts are unverified.")
        for edge in escalated:
            edge.review_reason = (edge.review_reason or
                                  f"Not analysed: {reason}")
        return escalated

    resolved_count = 0
    kinds = Counter()
    reasons = Counter()
    arg_shapes = Counter()
    conditional = 0

    for edge in escalated:
        arg_shapes[classify(first_argument(edge.arg_text))] += 1
        outcome = _interpret(edge, flows.get((edge.file, edge.line)))

        if outcome.resolved:
            resolved_count += 1
            kinds[outcome.kind] += 1
            if outcome.conditions:
                conditional += 1
            log(LOG, "info", "escalation resolved by CPG data flow",
                file=edge.file, line=edge.line, source_kind=outcome.kind,
                host=edge.host, detail=outcome.detail,
                conditions=outcome.conditions[:3])
        else:
            reasons[outcome.reason or "unknown"] += 1
            log(LOG, "info" if outcome.analyzed else "warning",
                "escalation exhausted" if outcome.analyzed
                else "call site was not covered by any CPG",
                file=edge.file, line=edge.line, callee=edge.callee_expr,
                reason=outcome.reason, detail=outcome.detail)
            if not outcome.analyzed:
                edge.review_reason = (edge.review_reason or
                                      f"Not analysed by Joern: {outcome.reason}")
            if TRACE_EDGES:
                log(LOG, "debug", "flow outcome", file=edge.file, line=edge.line,
                    outcome=outcome.as_dict())

    if ctx is not None:
        ctx.bump("joern.escalated", len(escalated))
        ctx.bump("joern.resolved", resolved_count)
        ctx.bump("joern.resolved.control_dependent", conditional)
        for kind, n in kinds.items():
            ctx.bump(f"joern.resolved.source_{kind}", n)
        for reason, n in reasons.items():
            ctx.bump(f"joern.bail.{reason}", n)
        for shape, n in arg_shapes.items():
            ctx.bump(f"joern.arg_shape.{shape}", n)
        if status["skipped"]:
            ctx.note("warning", "joern",
                     "Joern data-flow analysis was unavailable for: " +
                     "; ".join(f"{lang} ({why})" for lang, why in status["skipped"].items()) +
                     ". Call sites in those files reached the SLM fallback unanalysed.")

    log(LOG, "info", "joern escalation complete",
        escalated=len(escalated), resolved=resolved_count,
        control_dependent=conditional, by_source=dict(kinds),
        analyzed_languages=status["analyzed"],
        skipped_languages=list(status["skipped"].keys()),
        bail_reasons=dict(reasons))

    if ctx is not None and escalated and not resolved_count:
        _explain_total_failure(ctx, escalated, status, arg_shapes, reasons)

    return escalated


def _explain_total_failure(ctx, escalated, status, arg_shapes, reasons):
    """Turn '0 resolved' into an actionable sentence. The cases worth
    separating are unchanged from the old stage; what changed is that
    'analysed' now means a CPG query actually ran."""
    unanalysed = sum(n for r, n in reasons.items()
                     if r in ("call_site_not_present_in_cpg", "no_call_node_at_this_location"))
    if status["skipped"] and unanalysed:
        msg = (f"None of the {len(escalated)} escalated call site(s) resolved, and "
               f"{unanalysed} of them were never analysed: " +
               "; ".join(f"{lang} ({why})" for lang, why in list(status["skipped"].items())[:3]) +
               ". This is missing analysis, not a negative result.")
        level = "error"
    elif unanalysed == len(escalated):
        msg = (f"All {len(escalated)} escalated call site(s) were missing from the CPG "
               "even though indexing succeeded — check that repo_root matches the paths "
               "in the parsed records and that the frontend did not skip those files.")
        level = "error"
    else:
        top = ", ".join(f"{r} x{n}" for r, n in reasons.most_common(3))
        msg = (f"All {len(escalated)} escalated call site(s) were analysed by Joern and "
               f"none resolved to a host. Top reasons: {top}. Argument shapes: "
               f"{dict(arg_shapes)}.")
        level = "info"
    ctx.note(level, "joern", msg)
    log(LOG, level, "escalation yielded nothing", summary=msg)


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)