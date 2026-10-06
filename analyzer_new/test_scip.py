#!/usr/bin/env python3
"""
Standalone test for Stage 2 (parser positions + scip_check.py + resolver.py).

Runs OUTSIDE the pipeline: loads gaps.py, parser.py, scip_check.py and
resolver.py from one folder under a synthetic package with a stub `obs`. Needs the real SCIP
tools for the end-to-end part:

    npm i -g @sourcegraph/scip-python @sourcegraph/scip-typescript
    scip CLI  (https://github.com/sourcegraph/scip/releases)  on PATH or $SCIP_HOME
    pip install tree-sitter tree-sitter-language-pack requests

Usage:
    python test_scip_stage.py [--dir FOLDER_WITH_THE_THREE_FILES] [--keep]

Sections
  1. unit tests (no external tools): range/symbol handling, same-line trap,
     fallback rules
  2. end-to-end with the REAL indexers (skipped, and reported, if not installed)
  3. failure injection: every way SCIP can fail must degrade, never crash
  4. gap report: everything skipped / not covered / unmappable is reported as
     a structured Gap, and a call into an unanalysed file is never mistaken
     for an external call

Exit code 0 only if every executed check passed.
"""
import argparse
import importlib.util
import logging
import shutil
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

PKG = "_peach_stage2"
LOGGER = logging.getLogger("scip_test")
RESULTS = []


# ------------------------------------------------------------------------- plumbing ----

class Ctx:
    def __init__(self):
        self.counters, self.notes = Counter(), []

    def bump(self, key, n=1):
        self.counters[key] += n

    def count(self, key):
        return self.counters[key]

    def note(self, level, stage, msg):
        self.notes.append((level, stage, msg))


def load_modules(folder: Path):
    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    sys.modules[PKG] = pkg
    obs = types.ModuleType(f"{PKG}.obs")
    obs.TRACE_EDGES = False
    obs.get_logger = lambda name: logging.getLogger(f"scip_test.{name}")
    obs.log = lambda lg, lvl, msg, **kw: lg.log(
        getattr(logging, lvl.upper(), logging.INFO), "%s %s", msg,
        " ".join(f"{k}={v}" for k, v in kw.items()))
    sys.modules[obs.__name__] = obs
    mods = {}
    for name in ("gaps", "parser", "scip_check", "resolver"):
        spec = importlib.util.spec_from_file_location(f"{PKG}.{name}", folder / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok), detail))
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}" + (f"   -> {detail}" if not ok else ""))


def write_repo(root: Path, files: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def edge_at(edges, file, line, callee):
    hits = [e for e in edges if e.file == file and e.line == line and e.callee_expr == callee]
    return hits[0] if hits else None


def expect_edge(edges, file, line, callee, targets, via=None):
    """targets=None means 'must NOT be internal'."""
    e = edge_at(edges, file, line, callee)
    label = f"{file}:{line} {callee} -> " + ("unresolved/external" if targets is None else targets[0])
    if e is None:
        return check(label, False, "no such edge")
    if targets is None:
        check(label, e.status != "internal", f"status={e.status} targets={e.resolved_targets}")
    else:
        ok = e.status == "internal" and e.resolved_targets == targets and (via is None or e.resolved_via == via)
        check(label + (f" [{via}]" if via else ""), ok,
              f"status={e.status} targets={e.resolved_targets} via={e.resolved_via}")


# -------------------------------------------------------------------------- fixtures ----

PY = {
    "pkg/__init__.py": "",
    "pkg/util.py": "def helper(x):\n    return x + 1\n\n\ndef post(x):\n    return x\n",
    "app.py": '''import requests
from pkg.util import helper, post


class Client:
    def fetch(self, url):
        return self.build(url)

    def build(self, url):
        return requests.get(url)


def run():
    c = Client()
    c.fetch("https://api.example.com")
    requests.post(helper(1))
    post(2)
''',
}
WEB = {
    "web/tsconfig.json": '{"compilerOptions":{"allowJs":true,"target":"es2020","module":"commonjs"},"include":["src"]}',
    "web/src/lib.ts": '''export function helper(x: number): number { return x + 1; }
export class Service {
  load(id: string) { return this.build(id); }
  build(id: string) { return fetch(`/x/${id}`); }
}
''',
    "web/src/main.ts": '''import { helper, Service } from "./lib";
const s = new Service();
s.load("a");
helper(1);
axios.post(helper(2));
''',
}
MIXED = {**PY, **WEB}

JS_ONLY = {   # no tsconfig anywhere -> needs --infer-tsconfig, which must be cleaned up
    "src/a.js": "function helper(x) { return x + 1; }\nmodule.exports = { helper };\n",
    "src/b.js": 'const { helper } = require("./a");\nclass K { go() { return this.run(); } run() { return helper(1); } }\nnew K().go();\naxios.get(helper(2));\n',
}

TS_ROOT = {   # root tsconfig covers src/ only; scripts/tool.ts is NOT covered by SCIP
    "tsconfig.json": '{"compilerOptions":{"target":"es2020","module":"commonjs"},"include":["src"]}',
    "src/lib.ts": "export function helper(x: number): number { return x + 1; }\nhelper(1);\n",
    "scripts/tool.ts": "function a() { return b(); }\nfunction b() { return 1; }\nclass T { m() { return this.n(); } n() { return 2; } }\n",
}


def parse_and_resolve(mods, root, ctx, with_root=True):
    records = mods["parser"].parse_repo(root, ctx=ctx)
    edges = mods["resolver"].resolve_calls(records, ctx=ctx, repo_root=root if with_root else None)
    return records, edges


# --------------------------------------------------------------------- 1. unit tests ----

def unit_tests(mods, tmp: Path):
    print("\n== 1. unit tests (no external tools)")
    sc, rs = mods["scip_check"], mods["resolver"]

    check("range: 3-element -> (line+1, col)", sc._parse_range([4, 7, 12]) == (5, 7))
    check("range: 4-element -> (line+1, col)", sc._parse_range([4, 7, 6, 2]) == (5, 7))
    check("range: zero-width module range ignored", sc._parse_range([0, 0, 0]) is None)
    check("doc path: backslashes normalised", sc._doc_path({"relative_path": "a\\b\\c.py"}) == "a/b/c.py")
    check("doc path: camelCase accepted", sc._doc_path({"relativePath": "x.py"}) == "x.py")
    check("symbol key: locals are per-file", sc._symbol_key("a.py", "local 0") != sc._symbol_key("b.py", "local 0"))
    check("symbol key: globals are shared", sc._symbol_key("a.py", "pkg/f().") == "pkg/f().")

    # same-line trap: two calls on one line; only `helper` is defined in the repo
    index = {"documents": [
        {"relative_path": "m.py", "occurrences": [
            {"range": [0, 4, 10], "symbol": "pkg/helper().", "symbol_roles": 1},
            {"range": [5, 13, 17], "symbol": "requests/post()."},      # external: no definition anywhere
            {"range": [5, 18, 24], "symbol": "pkg/helper()."},
            {"range": [6, 4, 5], "symbol": "local 0", "symbol_roles": 1},
        ]},
        {"relative_path": "n.py", "occurrences": [
            {"range": [2, 0, 1], "symbol": "local 0"},                  # a DIFFERENT local 0
        ]},
    ]}
    wanted = {("m.py", 6, 13), ("m.py", 6, 18), ("n.py", 3, 0)}
    refs, covered = sc._merge_index(index, wanted)
    check("same-line trap: external call on the line is NOT resolved", ("m.py", 6, 13) not in refs)
    check("same-line trap: internal call on the line IS resolved", refs.get(("m.py", 6, 18)) == (("m.py", 1, 4),))
    check("local symbols do not leak across files", ("n.py", 3, 0) not in refs)
    check("covered files collected", covered == {"m.py", "n.py"})

    # lexical fallback rules
    defined = {"f.py::foo", "f.py::Klass", "f.py::Klass.meth", "f.py::Klass.other"}
    mk = lambda callee, caller: types.SimpleNamespace(callee_expr=callee, caller=caller, file="f.py")
    check("fallback: bare same-file function", rs._lexical_targets(mk("foo", "f.py::<module>"), defined) == ["f.py::foo"])
    check("fallback: bare same-file class", rs._lexical_targets(mk("Klass", "f.py::<module>"), defined) == ["f.py::Klass"])
    check("fallback: self.method in a method", rs._lexical_targets(mk("self.other", "f.py::Klass.meth"), defined) == ["f.py::Klass.other"])
    check("fallback: this.method in a method", rs._lexical_targets(mk("this.other", "f.py::Klass.meth"), defined) == ["f.py::Klass.other"])
    check("fallback: self.x outside a class is unresolved", rs._lexical_targets(mk("self.other", "f.py::foo"), defined) == [])
    check("fallback: requests.post is NOT captured by a local `post`",
          rs._lexical_targets(mk("requests.post", "f.py::<module>"), defined | {"f.py::post"}) == [])
    check("fallback: other receivers unresolved", rs._lexical_targets(mk("obj.meth", "f.py::Klass.meth"), defined) == [])
    check("fallback: unknown bare name unresolved", rs._lexical_targets(mk("nothere", "f.py::<module>"), defined) == [])


# ------------------------------------------------------------------ 2. real indexers ----

def tools_present(mods):
    sc = mods["scip_check"]
    sc._bin_cache.clear()
    return {n: sc._find_binary(n) for n in ("scip", "scip-python", "scip-typescript")}


def end_to_end(mods, tmp: Path):
    print("\n== 2. end-to-end with the REAL indexers")
    tools = tools_present(mods)
    missing = [n for n, p in tools.items() if not p]
    if missing:
        print(f"   SKIPPED: not on PATH/$SCIP_HOME: {', '.join(missing)}")
        return
    for n, p in tools.items():
        print(f"   using {n}: {p}")

    # --- mixed Python + nested TypeScript project
    root = tmp / "mixed"
    write_repo(root, MIXED)
    ctx = Ctx()
    records, edges = parse_and_resolve(mods, root, ctx)
    print(f"   [mixed] edges={len(edges)} via_scip={ctx.count('resolve.internal.via_scip')} "
          f"fallback_calls={ctx.count('resolve.fallback.calls')} notes={[n[2][:90] for n in ctx.notes]}")
    check("mixed: no file needed the fallback", ctx.count("resolve.fallback.calls") == 0,
          f"fallback_calls={ctx.count('resolve.fallback.calls')} notes={ctx.notes}")
    S = "scip"
    expect_edge(edges, "app.py", 7, "self.build", ["app.py::Client.build"], S)
    expect_edge(edges, "app.py", 10, "requests.get", None)
    expect_edge(edges, "app.py", 14, "Client", ["app.py::Client"], S)
    expect_edge(edges, "app.py", 15, "c.fetch", ["app.py::Client.fetch"], S)
    expect_edge(edges, "app.py", 16, "requests.post", None)          # same line as an internal call
    expect_edge(edges, "app.py", 16, "helper", ["pkg/util.py::helper"], S)
    expect_edge(edges, "app.py", 17, "post", ["pkg/util.py::post"], S)
    expect_edge(edges, "web/src/lib.ts", 3, "this.build", ["web/src/lib.ts::Service.build"], S)
    expect_edge(edges, "web/src/lib.ts", 4, "fetch", None)
    expect_edge(edges, "web/src/main.ts", 2, "Service", ["web/src/lib.ts::Service"], S)
    expect_edge(edges, "web/src/main.ts", 3, "s.load", ["web/src/lib.ts::Service.load"], S)
    expect_edge(edges, "web/src/main.ts", 4, "helper", ["web/src/lib.ts::helper"], S)
    expect_edge(edges, "web/src/main.ts", 5, "axios.post", None)     # same line as an internal call
    expect_edge(edges, "web/src/main.ts", 5, "helper", ["web/src/lib.ts::helper"], S)

    # --- JS-only repo: --infer-tsconfig, and the generated tsconfig.json must be removed
    root = tmp / "jsonly"
    write_repo(root, JS_ONLY)
    ctx = Ctx()
    records, edges = parse_and_resolve(mods, root, ctx)
    check("js-only: indexed via --infer-tsconfig (no fallback)", ctx.count("resolve.fallback.calls") == 0,
          f"notes={ctx.notes}")
    check("js-only: generated tsconfig.json removed from the repo", not (root / "tsconfig.json").exists())
    expect_edge(edges, "src/b.js", 2, "this.run", ["src/b.js::K.run"], S)
    expect_edge(edges, "src/b.js", 2, "helper", ["src/a.js::helper"], S)
    expect_edge(edges, "src/b.js", 4, "axios.get", None)

    # --- root tsconfig that does not include one file: only that file uses the fallback
    root = tmp / "tsroot"
    write_repo(root, TS_ROOT)
    ctx = Ctx()
    records, edges = parse_and_resolve(mods, root, ctx)
    expect_edge(edges, "src/lib.ts", 2, "helper", ["src/lib.ts::helper"], "scip")
    expect_edge(edges, "scripts/tool.ts", 1, "b", ["scripts/tool.ts::b"], "lexical_fallback")
    expect_edge(edges, "scripts/tool.ts", 3, "this.n", ["scripts/tool.ts::T.n"], "lexical_fallback")
    check("tsroot: fallback used only for the uncovered file",
          {e.file for e in edges if e.resolved_via == "lexical_fallback"} == {"scripts/tool.ts"})


# ------------------------------------------------------------- 3. failure injection ----

def failure_tests(mods, tmp: Path):
    print("\n== 3. failure injection: every failure must degrade, never raise")
    sc = mods["scip_check"]
    root = tmp / "fail"
    write_repo(root, MIXED)
    real_tools = tools_present(mods)
    saved = dict(sc._INDEXERS), sc.INDEX_TIMEOUT, sc.PRINT_TIMEOUT, sc._merge_index, sc._run_indexers

    def run_case(name, expect_skipped=None, expect_fallback=True):
        ctx = Ctx()
        try:
            records, edges = parse_and_resolve(mods, root, ctx)
        except Exception as exc:  # noqa: BLE001
            return check(f"{name}: pipeline did not crash", False, f"{type(exc).__name__}: {exc}")
        check(f"{name}: pipeline did not crash", True)
        if expect_fallback:
            check(f"{name}: fallback engaged", ctx.count("resolve.fallback.calls") > 0)
        # never fabricate: requests.post must stay unresolved whatever happened
        e = edge_at(edges, "app.py", 16, "requests.post")
        check(f"{name}: requests.post never internal", e is not None and e.status != "internal")
        e = edge_at(edges, "app.py", 7, "self.build")
        if expect_skipped and "python" in expect_skipped:
            check(f"{name}: self.build still resolved by same-file fallback",
                  e is not None and e.resolved_via == "lexical_fallback")
        return ctx, edges

    try:
        # a) SCIP CLI missing entirely
        import os
        old_path, old_home = os.environ.get("PATH", ""), os.environ.pop("SCIP_HOME", None)
        os.environ["PATH"] = str(tmp)           # nothing here
        sc._bin_cache.clear()
        ctx, edges = run_case("a) no tools installed", {"python", "typescript"})
        check("a) every edge came from fallback or is unresolved",
              all(e.resolved_via in (None, "lexical_fallback") for e in edges))
        check("a) warning note names the reason", any("not found" in n[2] for n in ctx.notes), str(ctx.notes))
        os.environ["PATH"] = old_path
        if old_home:
            os.environ["SCIP_HOME"] = old_home
        sc._bin_cache.clear()

        if not all(real_tools.values()):
            print("   (cases b-d need the real tools; skipped)")
        else:
            py_exe = sys.executable
            # b) python indexer crashes (exit 1): python files fall back, TS still SCIP
            sc._INDEXERS["scip-python"] = (("python",), lambda b, r, o: ([py_exe, "-c", "import sys; sys.exit(1)"], sc._noop))
            ctx, edges = run_case("b) python indexer crashes", {"python"})
            e = edge_at(edges, "web/src/main.ts", 3, "s.load")
            check("b) TypeScript still resolved by SCIP", e is not None and e.resolved_via == "scip")
            sc._INDEXERS.update(saved[0])

            # c) indexer hangs -> timeout
            sc.INDEX_TIMEOUT = 1
            sc._INDEXERS["scip-python"] = (("python",), lambda b, r, o: ([py_exe, "-c", "import time; time.sleep(30)"], sc._noop))
            ctx, edges = run_case("c) python indexer times out", {"python"})
            check("c) timeout reason reported", any("timed out" in n[2] for n in ctx.notes), str(ctx.notes))
            sc.INDEX_TIMEOUT = saved[1]
            sc._INDEXERS.update(saved[0])

            # d) indexer exits 0 but writes junk
            def junk(binary, r, out):
                return [py_exe, "-c", f"open(r'{out}','wb').write(b'not a scip index')"], sc._noop
            sc._INDEXERS["scip-python"] = (("python",), junk)
            ctx, edges = run_case("d) indexer writes a corrupt index", {"python"})
            sc._INDEXERS.update(saved[0])

        # e) exception while merging an index
        if all(real_tools.values()):
            def boom(*a, **k):
                raise RuntimeError("synthetic merge failure")
            sc._merge_index = boom
            run_case("e) exception while merging", {"python", "typescript"})
            sc._merge_index = saved[3]

        # f) exception escaping the whole indexing layer
        def boom2(*a, **k):
            raise RuntimeError("synthetic stage failure")
        sc._run_indexers = boom2
        run_case("f) unexpected exception in stage", {"python", "typescript"})
        sc._run_indexers = saved[4]

        # g) repo_root not passed
        ctx = Ctx()
        records = mods["parser"].parse_repo(root, ctx=ctx)
        edges = mods["resolver"].resolve_calls(records, ctx=ctx)
        check("g) no repo_root: no crash, fallback used", ctx.count("resolve.fallback.calls") == len(edges))
    finally:
        sc._INDEXERS.clear()
        sc._INDEXERS.update(saved[0])
        sc.INDEX_TIMEOUT, sc.PRINT_TIMEOUT, sc._merge_index, sc._run_indexers = saved[1:]
        sc._bin_cache.clear()



# ------------------------------------------------------------------- 4. gap report ----

def gap_tests(mods, tmp: Path):
    print("\n== 4. gap report: nothing skipped or unresolved may disappear silently")
    P, R, G = mods["parser"], mods["resolver"], mods["gaps"]
    kinds = lambda gs: Counter(g.kind for g in gs)
    of_kind = lambda gs, k: [g for g in gs if g.kind == k]
    ts_ok = P._get_ts_parser("python") is not None

    # ---- A: minified, vendored dirs, parser exception, syntax error (default limits)
    root = tmp / "gaps_a"
    files = {
        "ok.py": "def f():\n    return g()\n\n\ndef g():\n    return 1\n",
        "min.js": "a(b(c(d(1))));" * 6000,
        "node_modules/dep/index.js": "function dep() {}\n",
        "venv/lib/x.py": "def y():\n    pass\n",
        "boom.py": "def z():\n    return 1\n",
    }
    if ts_ok:
        files["bad.ts"] = "function ok() { return 1; }\nfunction broken( {\n"
    write_repo(root, files)
    orig = P.parse_file

    def exploding(repo_root, path, ctx=None):
        if path.name == "boom.py":
            raise RuntimeError("synthetic parser failure")
        return orig(repo_root, path, ctx=ctx)

    P.parse_file = exploding
    try:
        gaps = []
        records = P.parse_repo(root, ctx=Ctx(), gaps=gaps)
    finally:
        P.parse_file = orig
    by_path = {r.path: r for r in records}
    k = kinds(gaps)
    check("gaps A: minified file reported", [g.path for g in of_kind(gaps, "minified")] == ["min.js"], str(gaps))
    v = of_kind(gaps, "vendored_dir")
    check("gaps A: vendored dirs summarised in ONE repo-level gap with the file count",
          len(v) == 1 and v[0].scope == "repo" and v[0].count == 2 and set(v[0].samples) == {"node_modules", "venv"}, str(v))
    boom = by_path.get("boom.py")
    check("gaps A: a file whose parse raised is NOT dropped - it is a skipped record",
          boom is not None and boom.skipped and boom.skip_reason == "parse_exception" and boom.language == "python",
          str(boom))
    check("gaps A: ...and it has a parse_exception gap", [g.path for g in of_kind(gaps, "parse_exception")] == ["boom.py"], str(k))
    if ts_ok:
        check("gaps A: healthy file has no gap", not [g for g in gaps if g.path == "ok.py"])
    check("gaps A: every gap names a stage, scope, detail and impact",
          all(g.stage and g.scope and g.detail and g.impact for g in gaps), str(gaps))
    if ts_ok:
        check("gaps A: syntax-error file reported (partial extraction)",
              [g.path for g in of_kind(gaps, "syntax_errors")] == ["bad.ts"], str(k))
        check("gaps A: tree-sitter files carry parse_mode=treesitter", by_path["ok.py"].parse_mode == "treesitter")

    # ---- B: size limits -> too_large; config impact must say UNKNOWN
    root = tmp / "gaps_b"
    write_repo(root, {"big.py": "x = 1\n" * 400, "data.json": '{"a": [' + "1," * 500 + "1]}", "small.py": "y = 1\n"})
    gaps = []
    records = P.parse_repo(root, ctx=Ctx(), gaps=gaps, max_source_bytes=1000, max_other_bytes=500)
    tl = {g.path: g for g in of_kind(gaps, "too_large")}
    check("gaps B: both oversized files reported", set(tl) == {"big.py", "data.json"}, str(gaps))
    check("gaps B: skipped config says 'UNKNOWN', not 'nothing found'",
          "UNKNOWN" in tl["data.json"].impact, tl["data.json"].impact)
    check("gaps B: skipped source says calls are missing from the graph",
          "missing from the graph" in tl["big.py"].impact, tl["big.py"].impact)
    check("gaps B: gaps_to_dicts is JSON-ready",
          __import__("json").dumps(G.gaps_to_dicts(gaps)) and G.summarize_gaps(gaps).get("too_large") == 2)

    # ---- C: file cap is reported with a count; ctx.gaps works without gaps=
    root = tmp / "gaps_c"
    write_repo(root, {f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(5)})
    ctx = Ctx()
    records = P.parse_repo(root, ctx=ctx, max_files=2)            # no gaps= -> ctx.gaps
    cap = of_kind(getattr(ctx, "gaps", []), "file_cap")
    check("gaps C: cap hit -> one repo gap counting the files NOT examined",
          len(cap) == 1 and cap[0].count == 3 and len(cap[0].samples) == 3, str(cap))
    check("gaps C: exactly max_files files were parsed", len([r for r in records if not r.skipped]) == 2)
    ctx2 = Ctx()
    P.parse_repo(root, ctx=ctx2, max_files=5)
    check("gaps C: no cap gap when everything fit", not of_kind(getattr(ctx2, "gaps", []), "file_cap"))

    # ---- C2: a repo living under a folder named like a vendored dir is still parsed
    root = tmp / "build" / "proj"
    write_repo(root, {"ok.py": "def f():\n    return 1\n"})
    records = P.parse_repo(root, ctx=Ctx(), gaps=[])
    check("gaps C2: repo under a folder named 'build' is not skipped wholesale",
          [r.path for r in records if not r.skipped] == ["ok.py"], str([r.path for r in records]))

    # ---- D: tree-sitter unavailable -> fallback files are flagged, not "no calls found"
    root = tmp / "gaps_d"
    write_repo(root, {"ok.py": "def f():\n    return g()\n\n\ndef g():\n    return 1\n"})
    real_get = P._get_ts_parser
    P._get_ts_parser = lambda grammar: None
    try:
        gaps = []
        records = P.parse_repo(root, ctx=Ctx(), gaps=gaps)
    finally:
        P._get_ts_parser = real_get
    rec = records[0]
    check("gaps D: fallback record is flagged parse_mode=fallback", rec.parse_mode == "fallback" and not rec.calls, str(rec))
    fb = of_kind(gaps, "parse_fallback")
    check("gaps D: parse_fallback gap says this is NOT 'analysed, no calls found'",
          len(fb) == 1 and fb[0].path == "ok.py" and "NOT" in fb[0].impact, str(gaps))

    # ---- E: resolver gaps, with SCIP output faked so no tools are needed
    PR = P
    def rec_(path, lang="python", **kw):
        return PR.FileRecord(path=path, language=lang, parse_mode="treesitter", **kw)
    def call_(file, line, col, expr="helper"):
        return PR.CallSite(caller=f"{file}::<module>", callee_expr=expr, line=line, arg_text="", file=file,
                           callee_line=line, callee_col=col)
    a = rec_("a.py")
    a.definitions = [PR.Definition(qualified_name="a.py::local_fn", name="local_fn", kind="function", file="a.py",
                                   start_line=1, end_line=2, name_line=1, name_col=4)]
    a.calls = [call_("a.py", 3, 4, "big_helper"),                 # defined in a parser-SKIPPED file
               call_("a.py", 4, 0, "dep"),                         # defined in a vendored dir (third party)
               call_("a.py", 5, 0, "local_fn"),                    # defined in a.py, mapped
               call_("a.py", 6, 0, "requests.get")]                # SCIP knows nothing about it
    big = PR._skipped_record("big.py", "too_large", 9999, "9,999 bytes exceeds the limit")
    c = rec_("c.py")                                               # parsed, but not in the SCIP index
    c.calls = [call_("c.py", 1, 0, "whatever")]
    d = rec_("d.ts", "typescript")                                 # its language's indexer failed
    records = [a, big, c, d]
    refs = {("a.py", 3, 4): (("big.py", 1, 4),),
            ("a.py", 4, 0): (("node_modules/x/lib.js", 1, 0),),
            ("a.py", 5, 0): (("a.py", 1, 4),)}
    status = {"indexed": ["python"], "skipped": {"typescript": "tsc exploded"},
              "covered_files": {"a.py"}, "scip_print_available": True}
    real_brm = R.build_reference_map
    R.build_reference_map = lambda recs, root, ctx=None: (refs, status)
    try:
        ctx, gaps = Ctx(), []
        edges = R.resolve_calls(records, ctx=ctx, repo_root=tmp, gaps=gaps)
        edges_norootless = R.resolve_calls(records, ctx=Ctx(), repo_root=None, gaps=(g2 := []))
    finally:
        R.build_reference_map = real_brm
    st = {e.callee_expr: e for e in edges if e.file == "a.py"}
    check("gaps E: call into a parser-skipped file -> internal_unmapped (NOT unresolved/external)",
          st["big_helper"].status == "internal_unmapped" and st["big_helper"].unmapped_def_files == ["big.py"],
          f"{st['big_helper'].status} {st['big_helper'].unmapped_def_files}")
    check("gaps E: call into a vendored dir stays an ordinary unresolved/external candidate",
          st["dep"].status == "unresolved" and not st["dep"].unmapped_def_files, st["dep"].status)
    check("gaps E: mapped call is still internal via scip",
          st["local_fn"].status == "internal" and st["local_fn"].resolved_targets == ["a.py::local_fn"])
    check("gaps E: unknown callee stays unresolved", st["requests.get"].status == "unresolved")
    um = of_kind(gaps, "internal_unmapped")
    check("gaps E: internal_unmapped gap names the file, the count and why (parser skipped it)",
          len(um) == 1 and um[0].path == "big.py" and um[0].count == 1 and "too_large" in um[0].detail, str(um))
    check("gaps E: indexer-failed language -> ONE language-scope gap (no per-file flood)",
          len(of_kind(gaps, "scip_language_failed")) == 1
          and of_kind(gaps, "scip_language_failed")[0].scope == "language"
          and "tsc exploded" in of_kind(gaps, "scip_language_failed")[0].detail
          and not [g for g in gaps if g.path == "d.ts"], str(gaps))
    nc = of_kind(gaps, "scip_not_covered")
    check("gaps E: indexed language but file missing from the index -> per-file gap",
          [g.path for g in nc] == ["c.py"], str(nc))
    check("gaps E: no gap for the fully covered file", not [g for g in gaps if g.path == "a.py"])
    check("gaps E: counters - unmapped is not counted as unresolved",
          ctx.count("resolve.internal_unmapped") == 1
          and ctx.count("resolve.unresolved") == len(edges) - ctx.count("resolve.internal") - 1)
    check("gaps E: no repo_root -> one 'scip_not_attempted' gap, nothing else from scip",
          [g.kind for g in g2] == ["scip_not_attempted"], str(g2))
    check("gaps E: without repo_root no call is ever internal_unmapped",
          all(e.status != "internal_unmapped" for e in edges_norootless))

    # ---- F: the real hazard, end to end with the real indexer
    tools = tools_present(mods)
    if not (tools.get("scip") and tools.get("scip-python")):
        print("   (gaps F needs scip + scip-python; skipped)")
        return
    root = tmp / "gaps_f"
    write_repo(root, {
        "big_mod.py": "def big_helper(x):\n    return x + 1\n" + "# padding\n" * 300,
        "app.py": 'import requests\nfrom big_mod import big_helper\n\n\ndef run():\n    requests.get("u")\n    return big_helper(1)\n',
    })
    ctx, gaps = Ctx(), []
    records = P.parse_repo(root, ctx=ctx, gaps=gaps, max_source_bytes=1000)
    edges = R.resolve_calls(records, ctx=ctx, repo_root=root, gaps=gaps)
    bh = edge_at(edges, "app.py", 7, "big_helper")
    check("gaps F: [real SCIP] call into a skipped file is internal_unmapped, not external",
          bh is not None and bh.status == "internal_unmapped" and bh.unmapped_def_files == ["big_mod.py"],
          f"{bh and bh.status} {bh and bh.unmapped_def_files}")
    rg = edge_at(edges, "app.py", 6, "requests.get")
    check("gaps F: [real SCIP] requests.get is still an unresolved/external candidate",
          rg is not None and rg.status == "unresolved", rg and rg.status)
    check("gaps F: [real SCIP] both gaps present: parser too_large + scip internal_unmapped",
          [g.path for g in of_kind(gaps, "too_large")] == ["big_mod.py"]
          and [g.path for g in of_kind(gaps, "internal_unmapped")] == ["big_mod.py"], str(kinds(gaps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent),
                    help="folder containing gaps.py, parser.py, scip_check.py, resolver.py")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

    mods = load_modules(Path(args.dir))
    tmp = Path(tempfile.mkdtemp(prefix="scip_stage_test_"))
    try:
        unit_tests(mods, tmp)
        end_to_end(mods, tmp)
        failure_tests(mods, tmp)
        gap_tests(mods, tmp)
    finally:
        if args.keep:
            print(f"\nkept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 80)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label, _, detail in failed:
        print(f"   FAILED: {label} -> {detail}")
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())