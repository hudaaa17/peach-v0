"""Tests for symbol_map.py. No network, no real indexers: SCIP output is built by hand.

Assumes the package is importable as `peach`; adjust the imports below if it is not.
"""
import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from analyzer_new.parser import Definition, FileRecord
from analyzer_new.scip_check import ScipResult, ScipSymbolSpan
from analyzer_new.symbol_map import (
    NOT_LISTED_NOTE, SymbolMapError, build_symbol_map, load_symbol_map, write_symbol_map,
)


# ------------------------------------------------------------------ helpers ----

def defn(qname, kind, start, end, name_col=4, file="a.py", name_line=None):
    return Definition(qualified_name=qname, name=qname.split(".")[-1], kind=kind, file=file,
                      start_line=start, end_line=end,
                      name_line=start if name_line is None else name_line, name_col=name_col)


def span(symbol, display, kind, start, end, name_line=None, name_col=4, file="a.py",
         enclosing=True, local=False):
    return ScipSymbolSpan(symbol=symbol, display_name=display, kind=kind, file=file,
                          name_line=start if name_line is None else name_line, name_col=name_col,
                          start_line=start, end_line=end if enclosing else start,
                          has_enclosing_range=enclosing, is_local=local)


def record(path="a.py", defs=(), parse_mode="treesitter", **kw):
    return FileRecord(path=path, language=kw.pop("language", "python"), definitions=list(defs),
                      parse_mode=parse_mode, **kw)


def scip_result(spans_by_file, covered=None):
    covered = set(spans_by_file) if covered is None else set(covered)
    return ScipResult(status={"indexed": ["python"], "skipped": {}, "covered_files": covered,
                              "scip_print_available": True},
                      spans=spans_by_file)


def build(records, scip, **kw):
    """build_symbol_map with a gap list. Without repo_root every call would carry the
    repo-level `file_stats_unavailable` gap; it is dropped here so tests can assert on the
    gap they care about (test_missing_repo_root_... checks it directly)."""
    gaps = []
    sm = build_symbol_map(records, scip, "abc123", gaps=gaps, **kw)
    if "repo_root" not in kw:
        gaps = [g for g in gaps if g.kind != "file_stats_unavailable"]
    return sm, gaps


def by_name(sm, path="a.py"):
    return {s["qualified_name"]: s for s in sm["files"][path]["symbols"]}


# -------------------------------------------------------------------- tests ----

def test_nested_class_method_function():
    defs = [defn("Foo", "class", 1, 20), defn("Foo.bar", "method", 3, 10),
            defn("Foo.bar.inner", "function", 5, 8), defn("Foo.baz", "method", 12, 18)]
    spans = [span("s Foo#", "Foo", "class", 1, 20), span("s Foo#bar().", "bar", "method", 3, 10),
             span("s Foo#bar().inner().", "inner", "function", 5, 8),
             span("s Foo#baz().", "baz", "method", 12, 18)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": spans}))
    syms = by_name(sm)
    assert [s["qualified_name"] for s in sm["files"]["a.py"]["symbols"]] == \
        ["Foo", "Foo.bar", "Foo.bar.inner", "Foo.baz"]
    assert syms["Foo"]["parent_id"] is None
    assert syms["Foo.bar"]["parent_id"] == syms["Foo"]["symbol_id"]
    assert syms["Foo.bar.inner"]["parent_id"] == syms["Foo.bar"]["symbol_id"]
    assert syms["Foo.baz"]["parent_id"] == syms["Foo"]["symbol_id"]
    assert all(s["boundary_source"] == "scip" for s in syms.values())
    assert syms["Foo.bar"]["symbol_id"] == "s Foo#bar()."
    assert sm["files"]["a.py"]["coverage"] == "scip"
    assert gaps == []


def test_scip_span_replaces_treesitter_boundaries():
    defs = [defn("f", "function", 3, 4)]                      # tree-sitter missed the decorator
    spans = [span("s f().", "f", "function", 1, 6, name_line=3)]
    sm, _ = build([record(defs=defs)], scip_result({"a.py": spans}))
    f = by_name(sm)["f"]
    assert (f["start_line"], f["end_line"], f["name_line"]) == (1, 6, 3)
    assert f["boundary_source"] == "scip"


def test_partial_overlap_falls_back_to_treesitter_and_emits_gap():
    defs = [defn("A", "function", 1, 10), defn("B", "function", 5, 9)]
    spans = [span("s A().", "A", "function", 1, 10), span("s B().", "B", "function", 5, 15)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": spans}))
    syms = by_name(sm)
    assert syms["A"]["boundary_source"] == "scip"
    assert (syms["B"]["start_line"], syms["B"]["end_line"]) == (5, 9)
    assert syms["B"]["boundary_source"] == "treesitter"
    assert syms["B"]["symbol_id"] == "ts:a.py:B"
    assert syms["B"]["parent_id"] == syms["A"]["symbol_id"]       # the tree-sitter span nests
    assert [g.kind for g in gaps] == ["boundary_conflict"]        # the conflict is still reported


def test_partial_overlap_gap_fields_when_nothing_to_fall_back_to():
    defs = [defn("A", "function", 1, 10), defn("B", "function", 5, 15)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": []}))
    (gap,) = gaps
    assert (gap.stage, gap.scope, gap.kind, gap.path) == ("vector", "file", "boundary_conflict", "a.py")
    assert gap.count == 1 and "B" in gap.detail
    # the tree-sitter spans overlap too: B keeps its span, and is not nested in A
    assert by_name(sm)["B"]["parent_id"] is None
    assert by_name(sm)["B"]["boundary_source"] == "treesitter"


def test_fallback_resolves_later_conflicts_without_over_flipping():
    defs = [defn("A", "function", 1, 10), defn("B", "function", 5, 9), defn("C", "function", 11, 20)]
    spans = [span("s A().", "A", "function", 1, 10), span("s B().", "B", "function", 5, 15),
             span("s C().", "C", "function", 11, 20)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": spans}))
    syms = by_name(sm)
    assert syms["B"]["boundary_source"] == "treesitter"
    assert syms["C"]["boundary_source"] == "scip"          # its conflict vanished once B was swapped
    assert syms["B"]["parent_id"] == syms["A"]["symbol_id"]
    assert syms["C"]["parent_id"] is None
    assert gaps[0].count == 1


def test_no_gap_for_cleanly_nested_or_adjacent_spans():
    defs = [defn("A", "function", 1, 10), defn("B", "function", 11, 20), defn("B.c", "function", 12, 14)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": []}))
    assert gaps == []


def test_span_without_enclosing_range_uses_treesitter():
    defs = [defn("f", "function", 2, 9)]
    spans = [span("s f().", "f", "function", 2, 2, enclosing=False)]
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": spans}))
    f = by_name(sm)["f"]
    assert (f["start_line"], f["end_line"]) == (2, 9)
    assert f["boundary_source"] == "treesitter"
    assert f["symbol_id"] == "ts:a.py:f"
    assert len(sm["files"]["a.py"]["symbols"]) == 1       # the SCIP span was consumed, not re-added
    assert gaps == []


def test_scip_only_symbol_included_and_unusable_ones_skipped():
    defs = [defn("f", "function", 1, 3)]
    spans = [span("s f().", "f", "function", 1, 3),
             span("s Gen#", "Gen", "class", 10, 30, name_col=6),            # SCIP-only: kept
             span("local 1", "tmp", "function", 12, 14, name_col=8, local=True),
             span("s v.", "v", "variable", 40, 40),
             span("s mod/", "mod", "module", 50, 50),
             span("s Flat#", "Flat", "class", 60, 60, enclosing=False),     # no range: dropped
             span("s Anon().", "", "function", 70, 75)]                     # no name: dropped
    sm, gaps = build([record(defs=defs)], scip_result({"a.py": spans}))
    syms = by_name(sm)
    assert sorted(syms) == ["Gen", "f"]
    assert syms["Gen"]["symbol_id"] == "s Gen#" and syms["Gen"]["name"] == "Gen"
    assert syms["Gen"]["kind"] == "class" and syms["Gen"]["boundary_source"] == "scip"
    (gap,) = gaps
    assert gap.kind == "scip_symbol_dropped" and gap.count == 2 and gap.path == "a.py"


def test_skipped_file():
    rec = record(path="big.js", language="javascript", parse_mode="", skipped=True,
                 skip_reason="too_large", defs=[defn("ghost", "function", 1, 2, file="big.js")])
    sm, _ = build([rec], scip_result({}))
    entry = sm["files"]["big.js"]
    assert entry["coverage"] == "skipped" and entry["skip_reason"] == "too_large"
    assert entry["symbols"] == [] and entry["language"] == "javascript"


def test_fallback_treesitter_only_and_unparsed_coverage():
    recs = [record("fb.py", [defn("f", "function", 1, 3, file="fb.py")], parse_mode="fallback"),
            record("ts.py", [defn("g", "function", 1, 3, file="ts.py")], parse_mode="treesitter"),
            record("cfg.json", language="config", parse_mode="", is_config=True),
            record("note.txt", language="other", parse_mode="")]
    # SCIP knows nothing about these files; stray spans for an uncovered file must be ignored
    stray = {"ts.py": [span("s g().", "g", "function", 1, 9, file="ts.py")]}
    sm, _ = build(recs, scip_result(stray, covered=[]))
    files = sm["files"]
    assert files["fb.py"]["coverage"] == "fallback_lines"
    assert files["ts.py"]["coverage"] == "treesitter_only"
    assert files["cfg.json"]["coverage"] == "unparsed" and files["cfg.json"]["symbols"] == []
    assert files["note.txt"]["coverage"] == "unparsed"
    g = files["ts.py"]["symbols"][0]
    assert (g["end_line"], g["boundary_source"], g["symbol_id"]) == (3, "treesitter", "ts:ts.py:g")
    assert files["fb.py"]["symbols"][0]["symbol_id"] == "ts:fb.py:f"


def test_scip_none_is_tolerated():
    sm, _ = build([record(defs=[defn("f", "function", 1, 2)])], None)
    assert sm["files"]["a.py"]["coverage"] == "treesitter_only"


def test_symbols_sorted_by_start_then_longer_first():
    defs = [defn("z", "function", 30, 31), defn("Outer", "class", 5, 20),
            defn("Outer.m", "method", 5, 8, name_col=8)]
    sm, _ = build([record(defs=defs)], scip_result({"a.py": []}))
    assert [s["qualified_name"] for s in sm["files"]["a.py"]["symbols"]] == ["Outer", "Outer.m", "z"]


def test_symbol_ids_unique_per_file():
    defs = [defn("f", "function", 1, 3), defn("f", "function", 5, 7), defn("f", "function", 9, 11)]
    spans = [span("s f().", "f", "function", 1, 3), span("s f().", "f", "function", 5, 7)]
    sm, _ = build([record(defs=defs)], scip_result({"a.py": spans}))
    ids = [s["symbol_id"] for s in sm["files"]["a.py"]["symbols"]]
    assert len(ids) == len(set(ids)) == 3
    assert ids[:2] == ["s f().", "s f().~2"]
    for s in sm["files"]["a.py"]["symbols"]:
        assert s["parent_id"] is None or s["parent_id"] in ids


def test_name_col_unknown_never_matches_scip():
    defs = [defn("f", "function", 1, 3, name_col=-1)]
    spans = [span("s g().", "g", "function", 1, 9, name_col=0)]
    sm, _ = build([record(defs=defs)], scip_result({"a.py": spans}))
    syms = by_name(sm)
    assert sorted(syms) == ["f", "g"]                        # f was not joined to g's span
    assert syms["f"]["boundary_source"] == "treesitter" and syms["f"]["end_line"] == 3
    assert syms["g"]["boundary_source"] == "scip"


def test_line_count_and_sha_from_repo_root(tmp_path):
    (tmp_path / "a.py").write_bytes(b"a\nb\nc")
    (tmp_path / "b.py").write_bytes(b"a\nb\n")
    (tmp_path / "empty.py").write_bytes(b"")
    recs = [record("a.py"), record("b.py"), record("empty.py")]
    sm, gaps = build(recs, scip_result({}), repo_root=tmp_path)
    assert sm["files"]["a.py"]["line_count"] == 3
    assert sm["files"]["b.py"]["line_count"] == 3            # "\n"-split, like the parser
    assert sm["files"]["empty.py"]["line_count"] == 0
    assert sm["files"]["a.py"]["file_sha"] == hashlib.sha256(b"a\nb\nc").hexdigest()
    assert sm["file_sha_algorithm"] == "sha256"
    assert gaps == []


def test_missing_repo_root_leaves_zero_and_emits_gap():
    gaps = []
    sm = build_symbol_map([record("a.py"), record("b.py")], scip_result({}), "c", gaps=gaps)
    assert sm["files"]["a.py"]["line_count"] == 0 and sm["files"]["a.py"]["file_sha"] == ""
    (gap,) = gaps
    assert (gap.scope, gap.kind, gap.count) == ("repo", "file_stats_unavailable", 2)


def test_unreadable_and_escaping_paths_emit_gaps(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside.py").write_bytes(b"secret\n")
    recs = [record("gone.py"), record("../outside.py")]
    sm, gaps = build(recs, scip_result({}), repo_root=root)
    assert {g.path for g in gaps if g.kind == "file_unreadable"} == {"gone.py", "../outside.py"}
    assert sm["files"]["../outside.py"]["file_sha"] == ""
    assert sm["files"]["gone.py"]["line_count"] == 0


def test_only_parsed_files_listed_and_note_present():
    spans = {"a.py": [], "node_modules/x.js": [span("s x().", "x", "function", 1, 2,
                                                    file="node_modules/x.js")]}
    sm, _ = build([record("a.py")], scip_result(spans))
    assert list(sm["files"]) == ["a.py"]
    assert sm["not_listed_note"] == NOT_LISTED_NOTE
    assert "max_files" in sm["not_listed_note"] and "SKIP_DIRS" in sm["not_listed_note"]


def test_top_level_fields():
    sm, _ = build([], scip_result({}), pipeline_version="9.9")
    assert sm["schema_version"] == 1 and sm["commit_sha"] == "abc123"
    assert sm["pipeline_version"] == "9.9" and sm["files"] == {}


def test_gaps_go_to_ctx_when_no_list_given_and_nowhere_is_fine():
    defs = [defn("A", "function", 1, 10), defn("B", "function", 5, 15)]
    ctx = SimpleNamespace()
    build_symbol_map([record(defs=defs)], scip_result({"a.py": []}), "c", ctx=ctx)
    assert "boundary_conflict" in [g.kind for g in ctx.gaps]
    build_symbol_map([record(defs=defs)], scip_result({"a.py": []}), "c")      # no sink: no crash


def _mixed_input(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "b.py").write_bytes(b"def f(): pass\n")
    defs_a = [defn("A", "function", 1, 10), defn("B", "function", 5, 12), defn("C", "class", 20, 40),
              defn("C.m", "method", 22, 30, name_col=8)]
    spans_a = [span("s A().", "A", "function", 1, 10), span("s B().", "B", "function", 5, 15),
               span("s C#", "C", "class", 20, 40), span("s C#m().", "m", "method", 22, 30, name_col=8),
               span("s Extra().", "Extra", "function", 50, 60)]
    recs = [record("a.py", defs_a), record("b.py", [defn("f", "function", 1, 1, file="b.py")],
                                           parse_mode="fallback")]
    return recs, {"a.py": spans_a}


def test_determinism_byte_identical(tmp_path):
    recs, spans = _mixed_input(tmp_path)
    out = []
    for i in range(4):
        r, s = list(recs), {"a.py": list(spans["a.py"])}
        random.Random(i).shuffle(r)
        random.Random(i).shuffle(s["a.py"])
        sm, _ = build(r, scip_result(s), repo_root=tmp_path)
        path = write_symbol_map(sm, tmp_path / f"run{i}")
        out.append(path.read_bytes())
    assert len(set(out)) == 1
    assert json.loads(out[0])["files"]["a.py"]["symbols"]          # non-trivial content


def test_round_trip_and_atomic_write(tmp_path):
    recs, spans = _mixed_input(tmp_path)
    sm, _ = build(recs, scip_result(spans), repo_root=tmp_path)
    art = tmp_path / "workspaces" / "job1" / "artifacts"
    path = write_symbol_map(sm, art)
    assert path == art / "symbol_map.json"
    assert load_symbol_map(path) == sm
    assert [p.name for p in art.iterdir()] == ["symbol_map.json"]   # no temp file left behind
    sm["commit_sha"] = "new"
    write_symbol_map(sm, art)                                       # overwrite in place
    assert load_symbol_map(path)["commit_sha"] == "new"
    assert [p.name for p in art.iterdir()] == ["symbol_map.json"]
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"


def test_load_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SymbolMapError):
        load_symbol_map(bad)
    bad.write_text(json.dumps({"schema_version": 2, "files": {}}), encoding="utf-8")
    with pytest.raises(SymbolMapError):
        load_symbol_map(bad)
    sm, _ = build([record(defs=[defn("f", "function", 1, 2)])], scip_result({}))
    sm["files"]["a.py"]["coverage"] = "bogus"
    bad.write_text(json.dumps(sm), encoding="utf-8")
    with pytest.raises(SymbolMapError):
        load_symbol_map(bad)
    with pytest.raises(FileNotFoundError):
        load_symbol_map(tmp_path / "missing.json")


def test_unicode_and_odd_paths_round_trip(tmp_path):
    rec = record("src/héllo wörld.py", [defn("é", "function", 1, 2, file="src/héllo wörld.py")])
    sm, _ = build([rec], scip_result({}))
    assert load_symbol_map(write_symbol_map(sm, tmp_path)) == sm