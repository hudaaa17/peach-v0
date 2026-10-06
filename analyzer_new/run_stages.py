#!/usr/bin/env python3
"""
Run Stage 1 (tree-sitter parse) and Stage 2 (SCIP resolution) on a real
codebase and write reports you can read.

    python run_stages.py /path/to/codebase
    python run_stages.py https://github.com/pallets/click
    python run_stages.py /path/to/codebase --out reports/mycode --max-files 2000

Writes into the output folder (default: ./peach_report_<name>_<timestamp>):

    summary.txt     the console report (same text), the thing to read first
    unresolved.csv  every call that did NOT resolve to repo code, with a
                    "possible repo match" column for the ones worth a look
    edges.csv       every call and its status (internal / internal_unmapped / unresolved)
    gaps.csv        everything that was NOT analysed, and why
    report.json     all of the above, machine-readable (what a UI would consume)
    run.log         full DEBUG log of both stages (--trace adds one line per internal edge)

Needs the same tools as test_scip.py: tree-sitter, tree-sitter-language-pack,
scip, scip-python, scip-typescript (all present in the peach-env Docker image).
Use --no-scip to run Stage 1 only, or to see the same-file fallback behaviour.

The target folder is analysed IN PLACE. One thing may touch it: for a plain
JavaScript repo with no tsconfig.json, scip-typescript is run with
--infer-tsconfig, which writes a tsconfig.json and removes it afterwards.
Mount the folder read-write, or analyse a copy.
"""
import argparse
import csv
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
from datetime import datetime
from pathlib import Path

PKG = "_peach_run"
LOGGER = "peach_run"


# ------------------------------------------------------------------ plumbing ----

class Ctx:
    """Stand-in for the pipeline's run context: counters, notes, gaps."""

    def __init__(self):
        self.counters, self.notes, self.gaps = Counter(), [], []

    def bump(self, key, n=1):
        self.counters[key] += n

    def count(self, key):
        return self.counters[key]

    def note(self, level, stage, msg):
        self.notes.append((level, stage, msg))
        logging.getLogger(f"{LOGGER}.ctx").log(
            getattr(logging, level.upper(), logging.INFO), "[%s] %s", stage, msg)


def setup_logging(log_file: Path, verbose: bool):
    root = logging.getLogger(LOGGER)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    fileh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    root.addHandler(fileh)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING if verbose else logging.CRITICAL)
    console.setFormatter(fmt)
    root.addHandler(console)


def load_modules(folder: Path, trace: bool):
    """Import gaps/parser/scip_check/resolver under a synthetic package with a
    stub `obs` (the real obs.py is not needed for this runner)."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    sys.modules[PKG] = pkg
    obs = types.ModuleType(f"{PKG}.obs")
    obs.TRACE_EDGES = trace
    obs.get_logger = lambda name: logging.getLogger(f"{LOGGER}.{name}")
    obs.log = lambda lg, lvl, msg, **kw: lg.log(
        getattr(logging, lvl.upper(), logging.INFO), "%s %s", msg,
        " ".join(f"{k}={v}" for k, v in kw.items()))
    sys.modules[obs.__name__] = obs
    mods = {}
    for name in ("gaps", "parser", "scip_check", "resolver"):
        path = folder / f"{name}.py"
        if not path.is_file():
            raise SystemExit(f"cannot find {path} - use --dir to point at the folder with the stage files")
        spec = importlib.util.spec_from_file_location(f"{PKG}.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def resolve_target(target: str, workdir: Path):
    """A local folder, or a git URL (shallow clone). Returns (path, display name)."""
    if Path(target).is_dir():
        p = Path(target).resolve()
        return p, p.name
    if target.startswith(("http://", "https://", "git@")) or target.endswith(".git"):
        dest = workdir / "repo"
        proc = subprocess.run(["git", "clone", "--depth", "1", target, str(dest)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"git clone failed: {proc.stderr.strip()[-300:]}")
        return dest, target.rstrip("/").split("/")[-1].removesuffix(".git")
    raise SystemExit(f"not a folder or a git URL: {target}\n"
                     "(inside Docker, a Windows path must be mounted first - see the instructions)")


# ------------------------------------------------------------------ the run ----

def analyze(mods, repo_root: Path, ctx: Ctx, use_scip: bool, limits: dict):
    gaps = ctx.gaps
    t0 = time.perf_counter()
    records = mods["parser"].parse_repo(repo_root, ctx=ctx, gaps=gaps, **limits)
    t1 = time.perf_counter()
    edges = mods["resolver"].resolve_calls(records, ctx=ctx, gaps=gaps,
                                           repo_root=repo_root if use_scip else None)
    t2 = time.perf_counter()
    return records, edges, gaps, {"parse_seconds": t1 - t0, "resolve_seconds": t2 - t1}


# --------------------------------------------------------------- the summary ----

def _head(callee_expr: str) -> str:
    return callee_expr.split(".")[0]


def _tail(callee_expr: str) -> str:
    return callee_expr.rsplit(".", 1)[-1]


def build_summary(records, edges, gaps, ctx, timings):
    defs_by_name = defaultdict(list)
    for r in records:
        for d in r.definitions:
            defs_by_name[d.name].append(d.qualified_name)

    parsed = [r for r in records if not r.skipped]
    status = Counter(e.status for e in edges)
    via = Counter(e.resolved_via for e in edges if e.status == "internal")
    unresolved = [e for e in edges if e.status == "unresolved"]
    unmapped = [e for e in edges if e.status == "internal_unmapped"]

    roots = Counter(_head(e.callee_expr) for e in unresolved)
    root_samples = {}
    for e in unresolved:
        root_samples.setdefault(_head(e.callee_expr), []).append(e)

    # unresolved calls whose last name equals a repo definition's name: the
    # review list for "did resolution miss something?" (NOT used to resolve).
    misses = defaultdict(list)
    for e in unresolved:
        name = _tail(e.callee_expr)
        if name and not name.startswith("<") and name in defs_by_name:
            misses[name].append(e)

    unmapped_by_file = Counter()
    for e in unmapped:
        for p in e.unmapped_def_files:
            unmapped_by_file[p] += 1

    gap_groups = defaultdict(list)
    for g in gaps:
        gap_groups[g.kind].append(g)

    return {
        "records": len(records),
        "parsed_files": len(parsed),
        "skipped_files": sum(1 for r in records if r.skipped),
        "config_files": sum(1 for r in parsed if r.is_config),
        "by_language": dict(Counter(r.language for r in parsed if not r.is_config)),
        "parse_mode": dict(Counter(r.parse_mode or "-" for r in parsed
                                   if not r.is_config and r.language in ("python", "javascript", "typescript"))),
        "definitions": sum(len(r.definitions) for r in records),
        "calls": sum(len(r.calls) for r in records),
        "imports": sum(len(r.imports) for r in records),
        "edge_status": dict(status),
        "internal_via": dict(via),
        "calls_in_scip_covered_files": ctx.count("resolve.scip.calls_covered"),
        "calls_in_fallback_files": ctx.count("resolve.fallback.calls"),
        "scip_files_covered": ctx.count("resolve.scip.files_covered"),
        "scip_languages_indexed": ctx.count("resolve.scip.languages_indexed"),
        "scip_languages_failed": ctx.count("resolve.scip.languages_skipped"),
        "top_unresolved_roots": roots.most_common(),
        "root_samples": root_samples,
        "possible_misses": sorted(misses.items(), key=lambda kv: -len(kv[1])),
        "defs_by_name": defs_by_name,
        "unmapped_by_file": unmapped_by_file.most_common(),
        "gap_groups": gap_groups,
        "timings": timings,
    }


def _loc(e) -> str:
    return f"{e.file}:{e.line}"


def _args(e, n=70) -> str:
    text = " ".join(e.arg_text.split())
    return text if len(text) <= n else text[:n - 3] + "..."


def render_report(s, target_name, show=8) -> str:
    L = []
    add = L.append

    def head(title):
        add("")
        add("=" * 100)
        add(title)
        add("=" * 100)

    head(f"PEACH stage report: {target_name}")
    add(f"parse {s['timings']['parse_seconds']:.1f}s | resolve {s['timings']['resolve_seconds']:.1f}s")

    # ---- Stage 1
    head("STAGE 1: tree-sitter parse")
    add(f"files returned: {s['records']}   parsed: {s['parsed_files']}   "
        f"skipped: {s['skipped_files']}   (config among parsed: {s['config_files']})")
    add(f"source files by language: {s['by_language'] or 'none'}")
    add(f"how source files were parsed: {s['parse_mode'] or 'none'}   "
        "(\"fallback\" = no call sites extracted; see gaps)")
    add(f"extracted: {s['definitions']:,} definitions, {s['calls']:,} calls, {s['imports']:,} imports")

    # ---- Stage 2
    head("STAGE 2: SCIP symbol resolution")
    st = s["edge_status"]
    total = sum(st.values())
    add(f"calls: {total:,}")
    for key, label in (("internal", "resolved to repo code"),
                       ("internal_unmapped", "internal, but target has no graph node (see gaps)"),
                       ("unresolved", "NOT repo code -> external candidates for ast-grep")):
        n = st.get(key, 0)
        pct = f"{100 * n / total:.1f}%" if total else "-"
        add(f"   {key:18} {n:>8,}  {pct:>6}   {label}")
    add(f"internal edges came from: {s['internal_via'] or 'none'}")
    add(f"SCIP: {s['scip_languages_indexed']} language(s) indexed, {s['scip_languages_failed']} failed, "
        f"{s['scip_files_covered']} file(s) in the index")
    add(f"calls in files SCIP covered: {s['calls_in_scip_covered_files']:,}   "
        f"calls handled by the same-file fallback instead: {s['calls_in_fallback_files']:,}")
    if s["calls_in_fallback_files"]:
        add("   !! the fallback only resolves same-file and self/this calls; see the SCIP gaps below")

    # ---- unresolved
    head("UNRESOLVED CALLS (what ast-grep will receive as external candidates)")
    if not s["top_unresolved_roots"]:
        add("none")
    else:
        add(f"{sum(n for _, n in s['top_unresolved_roots']):,} calls, grouped by their first name "
            "(the object or function the call starts from). Top 25:")
        add("")
        for root, n in s["top_unresolved_roots"][:25]:
            add(f"  {n:>6,}  {root}")
            for e in s["root_samples"][root][:min(show, 2)]:
                add(f"          {_loc(e):45} {e.callee_expr}({_args(e)})")
        add("")
        add("Reading this: libraries (requests, axios, fetch, os, json, logger...) at the top are expected.")
        add("Names that look like YOUR code at the top mean resolution missed something.")

    head("POSSIBLE MISSES: unresolved calls whose last name matches a function/class defined in the repo")
    add("These are NOT resolved. They are listed so you can judge SCIP's quality. Most are probably")
    add("external or dynamic calls that share a name (e.g. requests.post vs your own post()); the")
    add("rest show where resolution fails (dynamic dispatch, unusual project layout, untyped receivers).")
    add("")
    if not s["possible_misses"]:
        add("none")
    for name, es in s["possible_misses"][:15]:
        add(f"  {len(es):>5,} call(s) ending in '{name}'   repo defines: "
            + ", ".join(s["defs_by_name"][name][:2]) + (" ..." if len(s["defs_by_name"][name]) > 2 else ""))
        for e in es[:min(show, 3)]:
            add(f"          {_loc(e):45} {e.callee_expr}({_args(e, 50)})")

    head("INTERNAL BUT UNMAPPED (SCIP says repo code; the parser has no node for the target)")
    if not s["unmapped_by_file"]:
        add("none")
    for path, n in s["unmapped_by_file"][:show]:
        add(f"  {n:>5,} call(s) into {path}")

    # ---- gaps
    head("GAPS: what was NOT analysed, and why (this is what the UI should show the user)")
    if not s["gap_groups"]:
        add("none - everything in scope was analysed")
    for kind, gs in sorted(s["gap_groups"].items(), key=lambda kv: -sum(g.count for g in kv[1])):
        total_count = sum(g.count for g in gs)
        add(f"\n[{kind}]  stage={gs[0].stage}  entries={len(gs)}  total count={total_count:,}")
        add(f"   impact: {gs[0].impact}")
        for g in gs[:show]:
            where = g.path or f"({g.scope})"
            add(f"   - {where}: {g.detail[:150]}")
        if len(gs) > show:
            add(f"   ... and {len(gs) - show} more (see gaps.csv)")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------- the files ----

def write_outputs(out: Path, s, records, edges, gaps, ctx, target_name, text):
    (out / "summary.txt").write_text(text, encoding="utf-8")

    with open(out / "edges.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "line", "caller", "callee_expr", "status", "resolved_via",
                    "resolved_targets", "unmapped_def_files", "arg_text"])
        for e in edges:
            w.writerow([e.file, e.line, e.caller, e.callee_expr, e.status, e.resolved_via or "",
                        " | ".join(e.resolved_targets), " | ".join(e.unmapped_def_files), e.arg_text])

    roots = dict(s["top_unresolved_roots"])
    unresolved = sorted((e for e in edges if e.status == "unresolved"),
                        key=lambda e: (-roots.get(_head(e.callee_expr), 0), e.file, e.line))
    with open(out / "unresolved.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "line", "caller", "callee_expr", "arg_text", "possible_repo_match"])
        for e in unresolved:
            match = s["defs_by_name"].get(_tail(e.callee_expr), []) if not _tail(e.callee_expr).startswith("<") else []
            w.writerow([e.file, e.line, e.caller, e.callee_expr, e.arg_text, " | ".join(match[:3])])

    with open(out / "gaps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "scope", "kind", "path", "count", "detail", "impact"])
        for g in gaps:
            w.writerow([g.stage, g.scope, g.kind, g.path, g.count, g.detail, g.impact])

    report = {
        "target": target_name,
        "summary": {k: v for k, v in s.items()
                    if k not in ("root_samples", "defs_by_name", "gap_groups", "possible_misses")},
        "counters": dict(ctx.counters),
        "notes": [{"level": a, "stage": b, "message": c} for a, b, c in ctx.notes],
        "gaps": [asdict(g) for g in gaps],
        "files": [{"path": r.path, "language": r.language, "skipped": r.skipped,
                   "skip_reason": r.skip_reason, "parse_mode": r.parse_mode,
                   "definitions": len(r.definitions), "calls": len(r.calls)} for r in records],
        "edges": [asdict(e) for e in edges],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description="Run Stage 1 (tree-sitter) + Stage 2 (SCIP) on a codebase")
    ap.add_argument("target", help="local folder or git URL")
    ap.add_argument("--out", help="output folder (default: ./peach_report_<name>_<timestamp>)")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent),
                    help="folder with gaps.py, parser.py, scip_check.py, resolver.py")
    ap.add_argument("--max-files", type=int, default=400, help="parser file cap (default 400; raise it)")
    ap.add_argument("--max-source-bytes", type=int)
    ap.add_argument("--max-other-bytes", type=int)
    ap.add_argument("--no-scip", action="store_true", help="skip SCIP (Stage 1 + same-file fallback only)")
    ap.add_argument("--trace", action="store_true", help="log one line per internal edge to run.log")
    ap.add_argument("--show", type=int, default=8, help="examples shown per section (default 8)")
    ap.add_argument("--verbose", action="store_true", help="print warnings to the console as they happen")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="peach_run_"))
    try:
        repo_root, name = resolve_target(args.target, workdir)
        out = Path(args.out) if args.out else Path(f"peach_report_{name}_{datetime.now():%Y%m%d_%H%M%S}")
        out.mkdir(parents=True, exist_ok=True)
        setup_logging(out / "run.log", args.verbose)
        mods = load_modules(Path(args.dir), args.trace)

        limits = {"max_files": args.max_files}
        if args.max_source_bytes:
            limits["max_source_bytes"] = args.max_source_bytes
        if args.max_other_bytes:
            limits["max_other_bytes"] = args.max_other_bytes

        print(f"analysing {repo_root} ...")
        ctx = Ctx()
        records, edges, gaps, timings = analyze(mods, repo_root, ctx, not args.no_scip, limits)
        summary = build_summary(records, edges, gaps, ctx, timings)
        text = render_report(summary, name, show=args.show)
        write_outputs(out, summary, records, edges, gaps, ctx, name, text)

        print(text)
        print(f"Files written to {out.resolve()}:")
        for f in ("summary.txt", "unresolved.csv", "edges.csv", "gaps.csv", "report.json", "run.log"):
            print(f"   {f}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())