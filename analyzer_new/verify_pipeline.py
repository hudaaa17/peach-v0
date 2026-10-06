"""Verify Stage 0 (clone), Stage 1 (tree-sitter parse) and SCIP on one real repo.

    python -m analyzer_new.verify_pipeline owner/repo [--max-files 400] [--dump DIR] [--keep]

Run it where the indexers are installed (scip-python, scip-typescript and the `scip`
CLI, e.g. inside your Docker container). Every check compares a stage's output with
something independent of that stage, so "it ran" is not enough to pass:

  Stage 0  the commit SHA equals what `git rev-parse` says; the handle file round-trips;
           no symlink in the checkout points outside it.
  Stage 1  every Definition / CallSite position really holds that identifier in the file
           on disk (checked in bytes, as tree-sitter columns are bytes).
  SCIP     the status says what was indexed; SCIP and the parser are cross-checked
           (same definitions found? same start/end lines?); resolve_calls is run on the
           prebuilt result and the edge and gap totals are printed.

Exit code 1 if a hard check fails. Soft findings are printed as WARN and are for you to
judge (they are expected in small numbers, for example decorators or non-ASCII lines).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .clone import RepoHandle, cleanup_workspace, clone_repo_to_workspace
from .gaps import gaps_to_dicts, summarize_gaps
from .parser import parse_repo
from .resolver import resolve_calls
from .scip_check import run_scip

DEF_KINDS = ("function", "method", "class")
FAILURES = []


def hard(ok, message):
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        FAILURES.append(message)


def warn(message):
    print("WARN  " + message)


def pct(part, whole):
    return f"{part}/{whole} ({100.0 * part / whole:.1f}%)" if whole else "0/0"


def _lines(root: Path, rel: str, cache: dict):
    if rel not in cache:
        try:
            cache[rel] = (root / rel).read_bytes().split(b"\n")
        except OSError:
            cache[rel] = None
    return cache[rel]


def text_at(root, rel, line1, col, text, cache):
    """True if `text` sits at (1-based line, 0-based byte column) in the file."""
    lines = _lines(root, rel, cache)
    if lines is None or not (1 <= line1 <= len(lines)):
        return False
    want = text.encode("utf-8")
    return lines[line1 - 1][col:col + len(want)] == want


# ------------------------------------------------------------------ stage 0 ----------
def check_clone(url, workspaces, job_id):
    print("\n== Stage 0: clone ==")
    handle = clone_repo_to_workspace(url, job_id, workspaces)
    print(f"cloned {handle.url} -> {handle.root}")
    print(f"commit={handle.commit_sha} branch={handle.default_branch or '?'} shallow={handle.is_shallow}")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=handle.root, capture_output=True,
                          text=True).stdout.strip()
    hard(handle.commit_sha == head and len(head) in (40, 64), "handle SHA equals `git rev-parse HEAD`")

    saved = workspaces / job_id / "artifacts" / "repo_handle.json"
    hard(saved.is_file() and RepoHandle.from_json(saved.read_text()) == handle,
         "artifacts/repo_handle.json exists and round-trips")

    files = [p for p in handle.root.rglob("*") if p.is_file() and not p.is_symlink()
             and ".git" not in p.relative_to(handle.root).parts]
    hard(len(files) > 0, f"checkout has files ({len(files)} outside .git)")
    root_real = os.path.realpath(handle.root)
    escaping = [p for p in handle.root.rglob("*") if p.is_symlink()
                and os.path.commonpath([root_real, os.path.realpath(p)]) != root_real]
    hard(not escaping, "no symlink points outside the checkout")
    return handle


# ------------------------------------------------------------------ stage 1 ----------
def check_parse(handle, max_files, gaps):
    print("\n== Stage 1: tree-sitter parse ==")
    records = parse_repo(handle.root, gaps=gaps, max_files=max_files)
    parsed = [r for r in records if not r.skipped and not r.is_config]
    print(f"records={len(records)} parsed_source={len(parsed)} "
          f"skipped={sum(1 for r in records if r.skipped)} config={sum(1 for r in records if r.is_config)}")
    print("languages:", dict(Counter(r.language for r in parsed)))
    print("parse_mode:", dict(Counter(r.parse_mode for r in parsed)))
    print("skip reasons:", dict(Counter(r.skip_reason for r in records if r.skipped)))
    hard(len(parsed) > 0, "at least one source file was parsed")

    n_defs = sum(len(r.definitions) for r in parsed)
    n_calls = sum(len(r.calls) for r in parsed)
    print(f"definitions={n_defs} calls={n_calls}")
    empty = [r.path for r in parsed if not r.definitions and not r.calls and not r.has_syntax_errors]
    if empty:
        warn(f"{len(empty)} parsed file(s) have no definitions and no calls, e.g. {empty[:3]}")
    fallback = [r.path for r in parsed if r.parse_mode == "fallback"]
    if fallback:
        warn(f"{len(fallback)} file(s) used the fallback parser, e.g. {fallback[:3]}")

    cache, bad_defs, bad_calls = {}, [], []
    for r in parsed:
        for d in r.definitions:
            if d.name_col >= 0 and not text_at(handle.root, r.path, d.name_line, d.name_col, d.name, cache):
                bad_defs.append((r.path, d.qualified_name, d.name_line, d.name_col))
        for c in r.calls:
            if c.callee_col >= 0:
                last = c.callee_expr.split(".")[-1]
                if not text_at(handle.root, r.path, c.callee_line, c.callee_col, last, cache):
                    bad_calls.append((r.path, c.callee_expr, c.callee_line, c.callee_col))
    hard(not bad_defs, f"every Definition name is at its recorded position "
                       f"({len(bad_defs)} wrong of {n_defs}; first: {bad_defs[:2]})")
    ratio = len(bad_calls) / n_calls if n_calls else 0
    hard(ratio < 0.01, f"CallSite callee positions hold the callee name "
                       f"({len(bad_calls)} wrong of {n_calls}; first: {bad_calls[:2]})")
    return records


# --------------------------------------------------------------------- SCIP ----------
def check_scip(handle, records, gaps, dump):
    print("\n== SCIP ==")
    scip = run_scip(records, handle.root)
    st = scip.status
    print(f"scip CLI found: {st['scip_print_available']}  indexed: {st['indexed']}")
    for lang, reason in st["skipped"].items():
        warn(f"SCIP skipped {lang}: {reason}")

    indexable_langs = {"python", "javascript", "typescript"}
    indexable = [r for r in records if not r.skipped and not r.is_config and r.language in indexable_langs]
    covered = [r for r in indexable if r.path in st["covered_files"]]
    print(f"files covered by an index: {pct(len(covered), len(indexable))}")
    print(f"call references resolved to repo code: {len(scip.ref_map)}")
    print(f"definition spans: {sum(len(v) for v in scip.spans.values())} in {len(scip.spans)} files")
    hard(bool(st["indexed"]), "SCIP indexed at least one language")
    if not st["indexed"]:
        return scip

    by_pos = {(s.file, s.name_line, s.name_col): s
              for spans in scip.spans.values() for s in spans}
    parser_defs = [(r, d) for r in covered for d in r.definitions
                   if d.kind in DEF_KINDS and d.name_col >= 0]
    found = [(d, by_pos[(d.file, d.name_line, d.name_col)]) for _, d in parser_defs
             if (d.file, d.name_line, d.name_col) in by_pos]
    print(f"parser definitions that SCIP also defines at the same spot: {pct(len(found), len(parser_defs))}")
    if parser_defs and len(found) / len(parser_defs) < 0.9:
        warn("under 90% agreement between parser definitions and SCIP definitions")
    exact = [(d, s) for d, s in found if s.has_enclosing_range
             and (d.start_line, d.end_line) == (s.start_line, s.end_line)]
    with_range = [(d, s) for d, s in found if s.has_enclosing_range]
    print(f"...of those with an enclosing range, same start/end lines: {pct(len(exact), len(with_range))}")
    off = [(d.qualified_name, (d.start_line, d.end_line), (s.start_line, s.end_line))
           for d, s in with_range if (d.start_line, d.end_line) != (s.start_line, s.end_line)]
    if off:
        warn(f"{len(off)} span(s) differ (decorators / trailing lines are common causes); first: {off[:3]}")

    known = {(d.file, d.name_line, d.name_col) for r in covered for d in r.definitions}
    missed = [s for spans in scip.spans.values() for s in spans
              if s.kind in DEF_KINDS and not s.is_local and s.file in {r.path for r in covered}
              and (s.file, s.name_line, s.name_col) not in known]
    if missed:
        warn(f"SCIP defines {len(missed)} function/method/class symbol(s) the parser did not record, "
             f"e.g. {[(s.file, s.display_name, s.name_line) for s in missed[:3]]}")

    edges = resolve_calls(records, repo_root=handle.root, gaps=gaps, scip=scip)
    print("resolve_calls edge statuses:", dict(Counter(e.status for e in edges)))
    print("gap totals by kind:", summarize_gaps(gaps))
    hard(any(e.status == "internal" for e in edges) or not scip.ref_map,
         "calls resolved to internal targets whenever SCIP found references")

    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        (dump / "scip_result.json").write_text(json.dumps(scip.to_json_safe()), encoding="utf-8")
        (dump / "gaps.json").write_text(json.dumps(gaps_to_dicts(gaps), indent=2), encoding="utf-8")
        print(f"wrote {dump / 'scip_result.json'} and gaps.json")
    return scip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", help="owner/repo or https://github.com/owner/repo")
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--dump", type=Path, help="directory to write scip_result.json and gaps.json")
    ap.add_argument("--keep", action="store_true", help="keep the cloned workspace")
    args = ap.parse_args(argv)

    workspaces = Path(tempfile.mkdtemp(prefix="peach_verify_"))
    job_id = "verify_job_001"
    gaps = []
    try:
        handle = check_clone(args.repo, workspaces, job_id)
        records = check_parse(handle, args.max_files, gaps)
        check_scip(handle, records, gaps, args.dump)
    finally:
        if args.keep:
            print(f"\nworkspace kept at {workspaces}")
        else:
            cleanup_workspace(workspaces, job_id, keep_artifacts=False)
            try:
                workspaces.rmdir()
            except OSError:
                pass
    print("\n" + ("ALL HARD CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} HARD CHECK(S) FAILED"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())