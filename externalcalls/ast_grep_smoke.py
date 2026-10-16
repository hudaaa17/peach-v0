"""
Smoke run of the ast-grep stage on a directory. Not a test: it prints what the
runner finds so you can try it on a real repo.

    python -m <package>.ast_grep_smoke DIR [--examples N]

Walks DIR the way parser.py does (same extensions, same skipped directories),
runs `run_patterns` and prints: the ast-grep binary and version, files and hits
per language, hits per catalogue id (every id, including zero) with a few
examples, alias/pass counters, client instances, and the run_report reasons.
Exit status is 1 if there are reasons
(something was not fully checked), else 0.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from . import ast_grep_check
from .ast_grep_check import CATALOGUE, run_patterns
from analyzer_new.parser import LANG_BY_EXT, SKIP_DIRS


def collect(root: Path):
    """{parser language: [repo-relative POSIX paths]} for the source files under root."""
    files = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            lang = LANG_BY_EXT.get(Path(name).suffix.lower())
            if lang:
                files[lang].append((Path(dirpath) / name).relative_to(root).as_posix())
    return dict(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ast-grep external-call stage on a directory.")
    ap.add_argument("directory")
    ap.add_argument("--examples", type=int, default=3, help="example hits shown per catalogue id")
    args = ap.parse_args(argv)

    root = Path(args.directory)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    files = collect(root)
    # Aggregated client_instances truncate samples and combine factory IDs.
    # Observe factory hits during this same run to print exact per-ID counts.
    factory_hits = []
    record_factory = ast_grep_check._record_factory
    def capture_factory(hit, report):
        factory_hits.append(hit)
        record_factory(hit, report)
    with patch.object(ast_grep_check, "_record_factory", side_effect=capture_factory):
        hits, report = run_patterns(root, files)

    print(f"directory : {root.resolve()}")
    print(f"ast-grep  : {report['binary'] or 'NOT FOUND'} {report['version']}".rstrip())
    print("files     : " + (", ".join(f"{lang}={len(v)}" for lang, v in sorted(files.items())) or "none"))
    print("\nper language")
    for lang, st in sorted(report["languages"].items()):
        print(f"  {lang:<11} files={st['files']} runs={st['runs']} hits={st['hits']} "
              f"dropped={st['dropped']} seconds={st['seconds']}")
        print(f"    alias_files={st.get('alias_files', 0)} alias_names={st.get('alias_names', 0)} "
              f"raw={st.get('alias_hits_raw', 0)} kept={st.get('alias_hits_kept', 0)} "
              f"dropped={st.get('alias_hits_dropped', 0)}")
        print(f"    discovery launches={st.get('discovery_runs', 0)} seconds={st.get('alias_discovery_seconds', 0)}; "
              f"plain launches={st['runs']} seconds={st['seconds']}; "
              f"alias launches={st.get('alias_pass_launches', 0)} seconds={st.get('alias_pass_seconds', 0)}")

    by_id = defaultdict(list)
    for h in hits + factory_hits:
        by_id[h.pattern_id].append(h)
    print(f"\nhits per catalogue id (total {len(hits)})")
    for entry in CATALOGUE:
        found = by_id.get(entry.id, [])
        print(f"  {entry.id:<22} total={len(found)} via_alias={sum(h.via_alias for h in found)}")
        for h in found[:args.examples]:
            print(f"      {h.file}:{h.start_line}:{h.start_col}  {h.callee_text}")

    print(f"\nclient instances ({len(report['client_instances'])})")
    for instance in report["client_instances"]:
        print(f"  {instance['file']} language={instance['language']} count={instance['count']} "
              f"via_alias_count={instance['via_alias_count']} factories={','.join(instance['factories'])}")
        print(f"    samples={'; '.join(instance['samples'])}")
        print(f"    {instance['kind']}: {instance['impact']}")

    print(f"\nreasons ({len(report['reasons'])})")
    for r in report["reasons"]:
        where = f" [{r['language']}]" if r["language"] else ""
        print(f"  {r['kind']} / {r['cause']}{where} x{r['count']}: {r['detail']}")
        if r["samples"]:
            print(f"      e.g. {', '.join(r['samples'])}")
    return 1 if report["reasons"] else 0


if __name__ == "__main__":
    sys.exit(main())
