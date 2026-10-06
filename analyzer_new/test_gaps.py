"""Tests for the merge / serialisation helpers in peach.gaps (no network, no indexers)."""
import json
import os

import pytest

from analyzer_new import gaps as g
from analyzer_new.gaps import (
    Gap,
    gaps_from_dicts,
    gaps_to_dicts,
    merge_gaps,
    read_gaps,
    summarize_gaps,
    write_gaps,
)


def _gap(**kw):
    base = dict(stage="vector", scope="file", kind="chunk_skipped", path="a.py", detail="d")
    base.update(kw)
    return Gap(**base)


# --- vocabulary -------------------------------------------------------------

@pytest.mark.parametrize("kind", [
    "boundary_conflict", "chunk_fallback_boundaries", "chunk_oversize_split",
    "chunk_skipped", "chunk_size_split_only", "embed_failed", "embed_partial",
    "store_failed", "branch_failed",
])
def test_docstring_lists_vector_kinds(kind):
    assert kind in g.__doc__


def test_docstring_keeps_unknown_not_nothing_found_rule():
    assert "A gap never means" in g.__doc__
    assert "It means \"unknown\"" in g.__doc__


# --- round trip -------------------------------------------------------------

def test_dict_round_trip():
    gaps = [
        _gap(),
        _gap(scope="repo", path="", kind="branch_failed", count=3, samples=["x.py", "y.py"],
             impact="no vector search"),
    ]
    assert gaps_from_dicts(gaps_to_dicts(gaps)) == gaps


def test_file_round_trip_utf8(tmp_path):
    gaps = [_gap(path="dir/файл.py", detail="détail – ünïcode"), _gap(kind="embed_failed")]
    p = tmp_path / "artifacts" / "gaps.json"   # parent dir is created
    write_gaps(gaps, p)
    assert read_gaps(p) == gaps
    raw = p.read_text(encoding="utf-8")
    assert "файл" in raw                      # ensure_ascii=False
    assert isinstance(json.loads(raw), list)


def test_empty_round_trip(tmp_path):
    p = tmp_path / "g.json"
    write_gaps([], p)
    assert read_gaps(p) == []
    assert gaps_from_dicts(None) == []


def test_summarize_unchanged_after_round_trip(tmp_path):
    gaps = [_gap(count=2), _gap(path="b.py", count=5), _gap(kind="embed_failed")]
    p = tmp_path / "g.json"
    write_gaps(gaps, p)
    assert summarize_gaps(read_gaps(p)) == summarize_gaps(gaps) == {
        "chunk_skipped": 7, "embed_failed": 1}


# --- unknown-key tolerance / malformed input --------------------------------

def test_unknown_keys_ignored():
    d = gaps_to_dicts([_gap()])[0]
    d["future_field"] = 123
    d["another"] = {"x": 1}
    assert gaps_from_dicts([d]) == [_gap()]


def test_defaults_applied_for_missing_optional_keys():
    got = gaps_from_dicts([{"stage": "vector", "scope": "repo", "kind": "store_failed"}])
    assert got == [Gap(stage="vector", scope="repo", kind="store_failed")]
    assert got[0].count == 1 and got[0].samples is None


@pytest.mark.parametrize("bad", [
    [{"stage": "vector", "scope": "repo"}],      # missing kind
    ["not a dict"],
    [None],
])
def test_malformed_entries_raise_value_error(bad):
    with pytest.raises(ValueError):
        gaps_from_dicts(bad)


def test_read_gaps_rejects_corrupt_and_wrong_shape(tmp_path):
    p = tmp_path / "g.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_gaps(p)
    p.write_text(json.dumps({"stage": "vector"}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_gaps(p)


def test_read_gaps_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_gaps(tmp_path / "nope.json")


# --- merge_gaps -------------------------------------------------------------

def test_merge_preserves_order_and_concatenates():
    a, b, c = _gap(path="a.py"), _gap(path="b.py"), _gap(path="c.py")
    assert merge_gaps([a, b], [c]) == [a, b, c]


def test_merge_drops_exact_duplicates_first_wins():
    first = _gap(count=1, impact="first")
    dup = _gap(count=9, impact="second")           # same key fields, different payload
    other = _gap(detail="different detail")
    out = merge_gaps([first], [dup, other], [first])
    assert out == [first, other]
    assert out[0].impact == "first"


@pytest.mark.parametrize("field,value", [
    ("stage", "parse"), ("scope", "repo"), ("kind", "embed_failed"),
    ("path", "z.py"), ("detail", "other"),
])
def test_merge_keeps_gaps_differing_in_any_key_field(field, value):
    assert len(merge_gaps([_gap()], [_gap(**{field: value})])) == 2


def test_merge_handles_none_and_empty_and_does_not_mutate():
    a = [_gap()]
    assert merge_gaps() == []
    assert merge_gaps(None, [], a) == a
    out = merge_gaps(a)
    out.append(_gap(path="new.py"))
    assert len(a) == 1


# --- atomic write -----------------------------------------------------------

def _files(d):
    return sorted(p.name for p in d.iterdir())


def test_successful_write_leaves_only_target(tmp_path):
    p = tmp_path / "g.json"
    write_gaps([_gap()], p)
    assert _files(tmp_path) == ["g.json"]


def test_failure_at_rename_leaves_no_partial_file(tmp_path, monkeypatch):
    p = tmp_path / "g.json"

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(g.os, "replace", boom)
    with pytest.raises(OSError):
        write_gaps([_gap()], p)
    assert _files(tmp_path) == []                  # no target, no temp


def test_failure_keeps_previous_file_intact(tmp_path, monkeypatch):
    p = tmp_path / "g.json"
    old = [_gap(kind="embed_failed")]
    write_gaps(old, p)

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(g.os, "replace", boom)
    with pytest.raises(OSError):
        write_gaps([_gap(kind="store_failed")], p)
    assert read_gaps(p) == old
    assert _files(tmp_path) == ["g.json"]


def test_unserialisable_payload_leaves_nothing_behind(tmp_path):
    p = tmp_path / "g.json"
    with pytest.raises(TypeError):
        write_gaps([_gap(samples=[{1, 2}])], p)    # a set is not JSON-serialisable
    assert _files(tmp_path) == []


def test_overwrite_replaces_existing(tmp_path):
    p = tmp_path / "g.json"
    write_gaps([_gap(kind="embed_failed")], p)
    write_gaps([_gap(kind="store_failed")], p)
    assert [x.kind for x in read_gaps(p)] == ["store_failed"]
    assert _files(tmp_path) == ["g.json"]