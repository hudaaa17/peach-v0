#!/usr/bin/env python3
"""
Standalone test harness for the Tree-sitter parser (parser.py).

Runs OUTSIDE the pipeline. It loads parser.py with a tiny stand-in for the
project's `obs` module (logger + counters), so nothing else from the project
is needed. It can:

  1. SELF-TEST (default, no arguments): writes a small fixture repo covering
     every supported extension (.py .js .jsx .mjs .cjs .ts .tsx), plus config
     files, a syntax-error file, an unsupported language, and files that must
     be SKIPPED (over the size limit, minified), then asserts the exact
     definitions / calls / imports / skip records that should come out.
  2. REAL REPO: shallow-clones (depth 1) a git URL, or takes a local folder,
     runs parse_repo on it and reports per-extension results.

Usage
-----
    python test_parser.py                                   # self-test
    python test_parser.py https://github.com/pallets/click  # clone + parse
    python test_parser.py ./some/local/folder
    python test_parser.py <repo> --samples 5 --trace --json report.json

Options
-------
    --parser PATH     parser.py to test (default: ./parser.py next to this file)
    --max-files N     file-count cap (default 400)
    --max-source-bytes N   size limit for source files (default: parser's MAX_SOURCE_BYTES)
    --max-other-bytes N    size limit for non-source files (default: parser's MAX_OTHER_BYTES)
    --samples N       definitions/calls/imports to print per extension (default 3)
    --trace           enable the parser's TRACE_EDGES debug logging
    --keep            keep the cloned repo / fixture folder instead of deleting it
    --log-file PATH   full DEBUG log (default: test_parser.log)
    --json PATH       also write the full results as JSON

Requirements:  pip install tree-sitter tree-sitter-language-pack   (and `git` for cloning)

Exit code is 0 only if: all four grammars load, no file fell back to the
line-based parser, and (in self-test mode) every assertion passed.
"""
import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

PKG = "_parser_under_test"                       # synthetic package so `from .obs import ...` works
SUPPORTED_EXTS = [".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"]
GRAMMARS = ["python", "javascript", "typescript", "tsx"]
LOGGER_ROOT = "parser_test"


# ------------------------------------------------------------------ logging ----

def setup_logging(log_file: str, trace: bool) -> logging.Logger:
    """Console at INFO (DEBUG with --trace); the log file always gets DEBUG."""
    root = logging.getLogger(LOGGER_ROOT)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if trace else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)
    fileh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    root.addHandler(fileh)
    return root


class Ctx:
    """Stand-in for the pipeline's run context: counters + notes."""

    def __init__(self):
        self.counters = Counter()
        self.notes = []

    def bump(self, key, n=1):
        self.counters[key] += n

    def count(self, key):
        return self.counters[key]

    def note(self, level, stage, msg):
        self.notes.append((level, stage, msg))
        logging.getLogger(f"{LOGGER_ROOT}.ctx").log(
            getattr(logging, level.upper(), logging.INFO), "[%s] %s", stage, msg)


def load_parser(parser_path: Path, trace: bool):
    """Import parser.py under a synthetic package with a stub `obs` module."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    sys.modules[PKG] = pkg

    obs = types.ModuleType(f"{PKG}.obs")
    obs.TRACE_EDGES = trace
    obs.get_logger = lambda name: logging.getLogger(f"{LOGGER_ROOT}.{name}")

    def log(logger, level, msg, **kw):
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        logger.log(getattr(logging, level.upper(), logging.INFO), "%s %s", msg, extra)

    obs.log = log
    sys.modules[obs.__name__] = obs

    gaps_path = parser_path.with_name("gaps.py")          # parser.py imports `.gaps`
    gspec = importlib.util.spec_from_file_location(f"{PKG}.gaps", gaps_path)
    gmod = importlib.util.module_from_spec(gspec)
    sys.modules[gspec.name] = gmod
    gspec.loader.exec_module(gmod)

    spec = importlib.util.spec_from_file_location(f"{PKG}.parser", parser_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses need this registered before exec
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ self-test data ----

FIXTURES = {
    "sample.py": '''import os
from .auth.funcs import login


class Client:
    def fetch(self, url):
        return requests.get(url)


def run():
    c = Client()
    c.fetch("https://api.example.com")


run()
''',
    "sample.js": '''import axios from 'axios';
const fs = require('fs');
function load(id) { return fetch(buildUrl(id)); }
const post = async (d) => axios.post("/p", d);
class Api { get(u) { return this.client.get(u); } }
''',
    "sample.jsx": '''import React from 'react';
export const App = () => {
  const d = useData();
  return <div onClick={() => go(d)}>{fmt(d)}</div>;
};
''',
    "sample.mjs": '''import { x } from './x.mjs';
export function m() { return x(); }
const lazy = () => import('./lazy.mjs');
''',
    "sample.cjs": '''const a = require('./a.cjs');
module.exports = function handler() { return a.run(); };
''',
    "sample.ts": '''import type { T } from "./types";
interface Repo { find(id: string): T }
enum Kind { A }
export class Service<T> {
  load(id: string): T { return fetch(`/x/${id}`) as any; }
}
export function ex(a: number): number { return id(a); }
''',
    "sample.tsx": '''import { useX } from "./x";
export const Comp = ({ a }: { a: number }) => {
  const v = useX(a);
  return <div>{fmt(v)}</div>;
};
''',
    # path contains "service": must be parsed as Python, not mistaken for config
    "services/user_service.py": '''import requests
def get_user(i):
    return requests.get("https://users.internal/" + str(i))
''',
    # syntax error: tree-sitter must still recover `ok`
    "broken.ts": '''export function ok() { return 1; }
function broken( {
  foo(
''',
    # config files: must NOT be parsed, must be flagged is_config
    ".env": "API_HOST=https://api.example.com\n",
    ".env.production": "API_HOST=https://prod.example.com\n",
    "k8s/deployment": "apiVersion: v1\nkind: Service\n",
    "settings.yaml": "host: example.com\n",
    # out of scope language: reported as "other"
    "Main.java": "public class Main { void f() { g(); } }\n",
    # --- files that must be SKIPPED and passed ahead as skipped records ---
    "big_generated.py": "x = 1\n" * 360_000,                        # ~2.16 MB source > 2 MB limit
    "huge.min.js": "a(b(c(d(1))));" * 6_000,                       # ~84 KB on ONE line -> minified
    "big.json": '{"a": [' + "1," * 230_000 + "1]}",                # ~460 KB non-source > 400 KB limit
    # vendored directory: ignored by design, summarised by ONE repo-level gap
    "node_modules/dep/index.js": "function dep() { return 1; }\n",
    # long line but small: must still be parsed (guards against false "minified")
    "long_line_ok.js": "fetch(buildUrl(1)); " * 60 + "\n",
}

# path -> expectations. defs/imports are exact sets, calls must be a subset.
EXPECT = {
    "sample.py": dict(language="python",
        defs={"sample.py::Client", "sample.py::Client.fetch", "sample.py::run"},
        calls={"requests.get", "Client", "c.fetch", "run"},
        imports={"os", "auth.funcs.login"}),
    "sample.js": dict(language="javascript",
        defs={"sample.js::load", "sample.js::post", "sample.js::Api", "sample.js::Api.get"},
        calls={"require", "fetch", "buildUrl", "axios.post", "this.client.get"},
        imports={"axios", "fs"}),
    "sample.jsx": dict(language="javascript",
        defs={"sample.jsx::App"}, calls={"useData", "go", "fmt"}, imports={"react"}),
    "sample.mjs": dict(language="javascript",
        defs={"sample.mjs::m", "sample.mjs::lazy"}, calls={"x", "import"},
        imports={"./x.mjs", "./lazy.mjs"}),
    "sample.cjs": dict(language="javascript",
        defs={"sample.cjs::handler"}, calls={"require", "a.run"}, imports={"./a.cjs"}),
    "sample.ts": dict(language="typescript",
        defs={"sample.ts::Repo", "sample.ts::Kind", "sample.ts::Service",
              "sample.ts::Service.load", "sample.ts::ex"},
        calls={"fetch", "id"}, imports={"./types"}),
    "sample.tsx": dict(language="typescript",
        defs={"sample.tsx::Comp"}, calls={"useX", "fmt"}, imports={"./x"}),
    "services/user_service.py": dict(language="python",
        defs={"services/user_service.py::get_user"}, calls={"requests.get", "str"},
        imports={"requests"}),
    "broken.ts": dict(language="typescript",
        defs={"broken.ts::ok"}, calls=set(), imports=None, expect_syntax_error=True),
    ".env": dict(is_config=True),
    ".env.production": dict(is_config=True),
    "k8s/deployment": dict(is_config=True),
    "settings.yaml": dict(is_config=True),
    "Main.java": dict(language="other"),
    "big_generated.py": dict(skipped_reason="too_large", language="python", is_config=False),
    "huge.min.js": dict(skipped_reason="minified", language="javascript", is_config=False),
    "big.json": dict(skipped_reason="too_large", language="config", is_config=True),
    "long_line_ok.js": dict(language="javascript", defs=set(), calls={"fetch", "buildUrl"},
                            imports=None),
}


def write_fixtures(root: Path):
    for rel, content in FIXTURES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def run_assertions(records, per_file):
    """Return a list of (label, passed, detail)."""
    by_path = {r.path: r for r in records}
    results = []

    def check(label, ok, detail=""):
        results.append((label, bool(ok), detail))

    for path, exp in EXPECT.items():
        rec = by_path.get(path)
        check(f"{path}: record produced", rec is not None)
        if rec is None:
            continue
        if "skipped_reason" in exp:
            check(f"{path}: skipped ({exp['skipped_reason']}) and passed ahead",
                  rec.skipped and rec.skip_reason == exp["skipped_reason"]
                  and not rec.definitions and not rec.calls and not rec.imports,
                  f"skipped={rec.skipped} reason={rec.skip_reason!r}")
            check(f"{path}: skipped record keeps language/is_config and has a detail",
                  rec.language == exp["language"] and rec.is_config == exp["is_config"]
                  and bool(rec.skip_detail) and rec.size_bytes > 0,
                  f"language={rec.language} is_config={rec.is_config} size={rec.size_bytes}")
            continue
        check(f"{path}: not skipped", not rec.skipped, f"reason={rec.skip_reason!r}")
        if "is_config" in exp:
            check(f"{path}: flagged as config, not parsed",
                  rec.is_config and not rec.definitions and not rec.calls,
                  f"is_config={rec.is_config} language={rec.language}")
            continue
        if "language" in exp:
            check(f"{path}: language == {exp['language']}", rec.language == exp["language"],
                  f"got {rec.language}")
        if exp["language"] == "other":
            continue
        defs = {d.qualified_name for d in rec.definitions}
        calls = {c.callee_expr for c in rec.calls}
        imps = {i.raw for i in rec.imports}
        check(f"{path}: definitions", defs == exp["defs"],
              f"missing={sorted(exp['defs'] - defs)} unexpected={sorted(defs - exp['defs'])}")
        check(f"{path}: calls", exp["calls"] <= calls, f"missing={sorted(exp['calls'] - calls)}")
        if exp.get("imports") is not None:
            check(f"{path}: imports", imps == exp["imports"],
                  f"missing={sorted(exp['imports'] - imps)} unexpected={sorted(imps - exp['imports'])}")
        info = per_file.get(path, {})
        check(f"{path}: parsed by tree-sitter (not fallback)", info.get("engine") == "tree-sitter",
              f"engine={info.get('engine')}")
        if exp.get("expect_syntax_error"):
            check(f"{path}: syntax error detected and recovered from",
                  info.get("syntax_error") and "broken.ts::ok" in defs)
    return results


def run_gap_assertions(mod, records, ctx, workdir):
    """Gap report: everything not analysed must be a structured Gap."""
    results = []

    def check(label, ok, detail=""):
        results.append((label, bool(ok), detail))

    by_path = {r.path: r for r in records}
    gaps = list(getattr(ctx, "gaps", []))
    of = lambda kind: [g for g in gaps if g.kind == kind]

    # --- the self-test repo
    check("gaps: too_large files == big_generated.py + big.json",
          {g.path for g in of("too_large")} == {"big_generated.py", "big.json"}, str([g.path for g in of("too_large")]))
    check("gaps: minified file reported", [g.path for g in of("minified")] == ["huge.min.js"])
    check("gaps: broken.ts reported as syntax_errors", [g.path for g in of("syntax_errors")] == ["broken.ts"])
    check("gaps: no file used the fallback, so no parse_fallback gap", not of("parse_fallback"))
    v = of("vendored_dir")
    check("gaps: vendored dirs -> ONE repo-level gap with count and sample dir names",
          len(v) == 1 and v[0].scope == "repo" and v[0].count == 1 and v[0].samples == ["node_modules"], str(v))
    check("gaps: every skipped record has a matching file gap",
          all(any(g.path == r.path and g.kind == r.skip_reason for g in gaps) for r in records if r.skipped))
    cfg = next((g for g in of("too_large") if g.path == "big.json"), None)
    check("gaps: skipped config's impact says UNKNOWN (not 'nothing found')", cfg is not None and "UNKNOWN" in cfg.impact)
    check("gaps: parse_mode set (treesitter for source, '' for config/other/skipped)",
          by_path["sample.py"].parse_mode == "treesitter" and by_path["settings.yaml"].parse_mode == ""
          and by_path["Main.java"].parse_mode == "" and by_path["big_generated.py"].parse_mode == "")
    check("gaps: syntax-error file carries has_syntax_errors, clean file does not",
          by_path["broken.ts"].has_syntax_errors and not by_path["sample.py"].has_syntax_errors)

    # --- parser exception: skipped record, never dropped
    root = workdir / "gap_exc"
    write_files(root, {"ok.py": "def f():\n    return 1\n", "boom.py": "def z():\n    return 1\n"})
    orig = mod.parse_file

    def exploding(repo_root, path, ctx=None):
        if path.name == "boom.py":
            raise RuntimeError("synthetic parser failure")
        return orig(repo_root, path, ctx=ctx)

    mod.parse_file = exploding
    try:
        g2 = []
        recs = mod.parse_repo(root, ctx=Ctx(), gaps=g2)
    finally:
        mod.parse_file = orig
    boom = {r.path: r for r in recs}.get("boom.py")
    check("gaps: file whose parse raised is returned as a skipped record (parse_exception)",
          boom is not None and boom.skipped and boom.skip_reason == "parse_exception", str(boom))
    check("gaps: ...with a parse_exception gap", [g.path for g in g2 if g.kind == "parse_exception"] == ["boom.py"])

    # --- file cap: counted, not silently cut
    root = workdir / "gap_cap"
    write_files(root, {f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(5)})
    c3 = Ctx()
    recs = mod.parse_repo(root, ctx=c3, max_files=2)      # gaps taken from ctx.gaps
    cap = [g for g in getattr(c3, "gaps", []) if g.kind == "file_cap"]
    check("gaps: file cap -> one repo gap counting the 3 files NOT examined (with samples)",
          len(cap) == 1 and cap[0].count == 3 and len(cap[0].samples) == 3, str(cap))
    check("gaps: exactly max_files files parsed", len([r for r in recs if not r.skipped]) == 2)

    # --- relative-path directory test
    root = workdir / "build" / "proj"
    write_files(root, {"ok.py": "def f():\n    return 1\n"})
    recs = mod.parse_repo(root, ctx=Ctx(), gaps=[])
    check("gaps: a repo located under a folder named 'build' is not skipped wholesale",
          [r.path for r in recs if not r.skipped] == ["ok.py"], str([r.path for r in recs]))

    # --- tree-sitter unavailable -> fallback flagged, not 'no calls found'
    root = workdir / "gap_fb"
    write_files(root, {"ok.py": "def f():\n    return g()\n\n\ndef g():\n    return 1\n"})
    real_get = mod._get_ts_parser
    mod._get_ts_parser = lambda grammar: None
    try:
        g4 = []
        recs = mod.parse_repo(root, ctx=Ctx(), gaps=g4)
    finally:
        mod._get_ts_parser = real_get
    check("gaps: fallback record flagged parse_mode=fallback with no calls",
          recs[0].parse_mode == "fallback" and not recs[0].calls)
    fb = [g for g in g4 if g.kind == "parse_fallback"]
    check("gaps: parse_fallback gap says NOT 'analysed, no calls found'",
          len(fb) == 1 and fb[0].path == "ok.py" and "NOT" in fb[0].impact, str(g4))
    return results


def write_files(root: Path, files: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ------------------------------------------------------------------ running ----

def clone_repo(url: str, dest: Path, log: logging.Logger):
    log.info("Shallow-cloning %s (depth 1) ...", url)
    t = time.perf_counter()
    cmd = ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)]
    env = {**__import__("os").environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {proc.stderr.strip()}")
    log.info("Cloned in %.1fs", time.perf_counter() - t)


def check_grammars(mod, log):
    """Load every grammar the parser can request and smoke-parse a snippet."""
    status = {}
    for g in GRAMMARS:
        parser = mod._get_ts_parser(g)
        if parser is None:
            err = mod._ts_load_errors.get(g) or mod._ts_load_errors.get("_module") or "unknown"
            status[g] = (False, err)
            log.error("grammar %-10s FAILED: %s", g, err)
            continue
        try:
            tree = parser.parse(b"x")
            status[g] = (True, f"root={tree.root_node.type}")
            log.info("grammar %-10s OK (%s)", g, status[g][1])
        except Exception as exc:  # noqa: BLE001
            status[g] = (False, f"{type(exc).__name__}: {exc}")
            log.error("grammar %-10s parse failed: %s", g, status[g][1])
    return status


def instrument(mod, ctx, per_file):
    """Wrap parse_file so we can see per-file engine, timing and syntax
    errors, without changing what parse_repo does."""
    orig_parse_file = mod.parse_file

    def parse_file(repo_root, path, ctx=None):
        keys = ("parse.treesitter.used", "parse.treesitter.fallback_used",
                "parse.treesitter.syntax_error", "parse.treesitter.parse_error")
        before = {k: ctx.count(k) for k in keys}
        t = time.perf_counter()
        rec = orig_parse_file(repo_root, path, ctx=ctx)
        d = {k: ctx.count(k) - before[k] for k in keys}
        if rec.skipped:
            engine = "skipped"
        elif rec.is_config:
            engine = "config"
        elif d["parse.treesitter.used"]:
            engine = "tree-sitter"
        elif d["parse.treesitter.fallback_used"]:
            engine = "FALLBACK"
        else:
            engine = "none"
        per_file[rec.path] = dict(
            engine=engine, seconds=time.perf_counter() - t, ext=Path(rec.path).suffix.lower(),
            syntax_error=bool(d["parse.treesitter.syntax_error"]),
            parse_error=bool(d["parse.treesitter.parse_error"]),
        )
        return rec

    mod.parse_file = parse_file


def summarize(records, per_file):
    """Per-extension statistics."""
    stats = defaultdict(lambda: Counter())
    for r in records:
        info = per_file.get(r.path, {})
        ext = info.get("ext") or Path(r.path).suffix.lower() or "(none)"
        s = stats[ext]
        s["files"] += 1
        s["defs"] += len(r.definitions)
        s["calls"] += len(r.calls)
        s["imports"] += len(r.imports)
        s["seconds_ms"] += int(info.get("seconds", 0) * 1000)
        engine = "skipped" if r.skipped else info.get("engine", "none")
        s[f"engine:{engine}"] += 1
        if info.get("syntax_error"):
            s["syntax_errors"] += 1
        if engine == "tree-sitter" and not (r.definitions or r.calls or r.imports):
            s["empty"] += 1
    return stats


def print_report(records, per_file, ctx, stats, log):
    line = "-" * 96
    print("\n" + line)
    print(f"{'ext':8} {'files':>6} {'tree-sitter':>12} {'fallback':>9} {'skipped':>8} {'config/other':>13} "
          f"{'defs':>7} {'calls':>7} {'imports':>8} {'syntax-err':>10} {'empty':>6} {'ms':>7}")
    print(line)
    order = SUPPORTED_EXTS + sorted(e for e in stats if e not in SUPPORTED_EXTS)
    for ext in order:
        s = stats.get(ext)
        if not s:
            if ext in SUPPORTED_EXTS:
                print(f"{ext:8} {'0':>6}   (no files with this extension in the repo)")
            continue
        other = s["engine:config"] + s["engine:none"]
        print(f"{ext:8} {s['files']:>6} {s['engine:tree-sitter']:>12} {s['engine:FALLBACK']:>9} "
              f"{s['engine:skipped']:>8} {other:>13} {s['defs']:>7} {s['calls']:>7} {s['imports']:>8} "
              f"{s['syntax_errors']:>10} {s['empty']:>6} {s['seconds_ms']:>7}")
    print(line)

    skipped = [r for r in records if r.skipped]
    if skipped:
        print(f"\nSKIPPED files ({len(skipped)}) - passed ahead as FileRecord(skipped=True):")
        for r in skipped[:30]:
            print(f"    {r.path}  [{r.skip_reason}]  {r.skip_detail}")
        if len(skipped) > 30:
            print(f"    ... and {len(skipped) - 30} more")

    fallbacks = [p for p, i in per_file.items() if i["engine"] == "FALLBACK"]
    if fallbacks:
        print("\nFiles parsed by the FALLBACK (tree-sitter did not handle them):")
        for p in fallbacks[:20]:
            print("   ", p)
    empties = [r.path for r in records
               if not r.skipped and per_file.get(r.path, {}).get("engine") == "tree-sitter"
               and not (r.definitions or r.calls or r.imports)]
    if empties:
        print(f"\n{len(empties)} tree-sitter file(s) yielded nothing (may be legitimately empty/constants-only):")
        for p in empties[:10]:
            print("   ", p)


def print_samples(records, per_file, n):
    """For each extension, show what was extracted from its richest file."""
    best = {}
    for r in records:
        info = per_file.get(r.path, {})
        if info.get("engine") != "tree-sitter":
            continue
        score = len(r.definitions) + len(r.calls) + len(r.imports)
        ext = info["ext"]
        if ext not in best or score > best[ext][0]:
            best[ext] = (score, r)
    for ext in SUPPORTED_EXTS:
        if ext not in best:
            continue
        r = best[ext][1]
        print(f"\n=== sample {ext}: {r.path}  (language={r.language}, "
              f"{len(r.definitions)} defs, {len(r.calls)} calls, {len(r.imports)} imports)")
        for d in r.definitions[:n]:
            print(f"   def   {d.kind:8} {d.qualified_name}  L{d.start_line}-{d.end_line}")
        for c in r.calls[:n]:
            arg = c.arg_text.replace("\n", " ")
            print(f"   call  L{c.line:<5} {c.caller.split('::', 1)[-1]} -> {c.callee_expr}({arg[:50]})")
        for i in r.imports[:n]:
            print(f"   import L{i.line:<4} {i.raw}")


def main():
    ap = argparse.ArgumentParser(description="Standalone test for parser.py (Tree-sitter stage)")
    ap.add_argument("repo", nargs="?", help="git URL or local folder; omit to run the built-in self-test")
    ap.add_argument("--parser", default=str(Path(__file__).with_name("parser.py")))
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--max-source-bytes", type=int)
    ap.add_argument("--max-other-bytes", type=int)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--log-file", default="test_parser.log")
    ap.add_argument("--json")
    args = ap.parse_args()

    log = setup_logging(args.log_file, args.trace).getChild("harness")
    parser_path = Path(args.parser).resolve()
    if not parser_path.is_file():
        log.error("parser file not found: %s", parser_path)
        return 2
    log.info("Testing %s", parser_path)

    mod = load_parser(parser_path, args.trace)
    grammar_status = check_grammars(mod, log)

    workdir = Path(tempfile.mkdtemp(prefix="parser_test_"))
    selftest = args.repo is None
    try:
        if selftest:
            repo_root = workdir / "fixture_repo"
            write_fixtures(repo_root)
            log.info("Self-test: wrote %d fixture files to %s", len(FIXTURES), repo_root)
        elif Path(args.repo).is_dir():
            repo_root = Path(args.repo).resolve()
            log.info("Using local folder %s", repo_root)
        else:
            repo_root = workdir / "repo"
            clone_repo(args.repo, repo_root, log)

        ctx = Ctx()
        per_file = {}
        instrument(mod, ctx, per_file)

        log.info("Running parse_repo ...")
        t = time.perf_counter()
        limits = {"max_files": args.max_files}
        if args.max_source_bytes:
            limits["max_source_bytes"] = args.max_source_bytes
        if args.max_other_bytes:
            limits["max_other_bytes"] = args.max_other_bytes
        records = mod.parse_repo(repo_root, ctx=ctx, **limits)
        elapsed = time.perf_counter() - t
        log.info("parse_repo finished: %d records in %.2fs", len(records), elapsed)

        stats = summarize(records, per_file)
        print_report(records, per_file, ctx, stats, log)
        print_samples(records, per_file, args.samples)

        print("\nCounters:")
        for k in sorted(ctx.counters):
            print(f"   {k:45} {ctx.counters[k]}")
        if ctx.notes:
            print("\nNotes raised by the parser:")
            for level, stage, msg in ctx.notes:
                print(f"   [{level}/{stage}] {msg}")

        # ---- verdict
        failures = []
        for g, (ok, detail) in grammar_status.items():
            if not ok:
                failures.append(f"grammar '{g}' did not load: {detail}")
        n_fallback = ctx.count("parse.treesitter.fallback_used")
        if n_fallback:
            failures.append(f"{n_fallback} file(s) used the fallback instead of tree-sitter")
        if ctx.count("parse.file_exception"):
            failures.append(f"{ctx.count('parse.file_exception')} file(s) raised inside the parser")

        if selftest:
            print("\nSelf-test assertions:")
            results = run_assertions(records, per_file)
            results += run_gap_assertions(mod, records, ctx, workdir)
            for label, ok, detail in results:
                print(f"   [{'PASS' if ok else 'FAIL'}] {label}" + (f"   -> {detail}" if not ok else ""))
            failed = [r for r in results if not r[1]]
            print(f"\n   {len(results) - len(failed)}/{len(results)} assertions passed")
            failures += [f"assertion failed: {r[0]}" for r in failed]

        if args.json:
            Path(args.json).write_text(json.dumps({
                "grammars": {g: {"ok": ok, "detail": d} for g, (ok, d) in grammar_status.items()},
                "counters": dict(ctx.counters),
                "per_file": per_file,
                "records": [asdict(r) for r in records],
            }, indent=2, default=str), encoding="utf-8")
            log.info("JSON written to %s", args.json)

        print("\n" + "=" * 96)
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print("   -", f)
        else:
            print("RESULT: PASS  - tree-sitter parsed every source file; no fallback used")
        print(f"Full debug log: {args.log_file}")
        return 1 if failures else 0
    finally:
        if args.keep:
            log.info("Kept working folder: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())