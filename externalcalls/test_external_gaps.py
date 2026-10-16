"""Report-to-gap mapping, isolation, and real external-call pipeline coverage."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from analyzer_new.gaps import gaps_from_dicts, gaps_to_dicts, merge_gaps
from analyzer_new.parser import parse_repo
from analyzer_new.resolver import resolve_calls
from externalcalls.ast_grep_check import run_patterns
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls.constprop_check import run_constprop_check
from externalcalls.external_gaps import build_external_call_gaps
from externalcalls.literal_check import run_literal_check


LITERAL = {"file_unreadable": "literal_file_unreadable", "grammar_unavailable": "literal_grammar_unavailable",
           "node_not_found": "literal_call_not_found", "ambiguous_node": "literal_call_ambiguous",
           "edge_failed": "literal_edge_failed", "no_position": "literal_no_position"}
CONSTPROP = {"file_unreadable": "constprop_file_unreadable", "grammar_unavailable": "constprop_grammar_unavailable",
             "argument_not_found": "constprop_argument_not_found", "edge_failed": "constprop_edge_failed"}
JOIN = {"join_aborted": "join_aborted", "stage_failed": "join_stage_failed", "hit_ambiguous": "join_hit_ambiguous",
        "hit_without_edge": "join_hit_without_edge", "hit_ignored": "join_hit_ignored",
        "regex_fallback_used": "regex_fallback_used"}


def reason(kind, cause="timeout", language="python", count=2):
    return dict(kind=kind, cause=cause, language=language, count=count, detail="unchecked work", samples=["client.py:3"])


def convert_at(index, report):
    reports = [None] * 4
    reports[index] = report
    return build_external_call_gaps(*reports)


@pytest.mark.parametrize("kind,language,expected", [
    ("ast_grep_language_failed", "python", "ast_grep_language_failed"),
    ("ast_grep_unavailable", "", "ast_grep_unavailable"),
    ("stage_failed", "", "ast_grep_stage_failed"),
])
def test_run_reasons(kind, language, expected):
    item, = convert_at(0, {"reasons": [reason(kind, language=language)]})
    assert (item.stage, item.scope, item.kind, item.path, item.count, item.samples) == (
        "ast_grep", "language" if language else "repo", expected, "", 2, ["client.py:3"])
    assert f"cause=timeout language={language}: unchecked work" == item.detail
    assert "unknown" in item.impact


def test_client_instances():
    record = dict(file="client.js", count=3, factories=["js.axios.create"], samples=["2:0 axios.create"],
                  impact="calls made through this client object are not detected")
    item, = convert_at(0, {"client_instances": [record]})
    assert (item.stage, item.scope, item.kind, item.path, item.count) == (
        "ast_grep", "file", "client_instance_untracked", "client.js", 3)
    assert "js.axios.create" in item.detail and "2:0 axios.create" in item.detail
    assert item.samples == record["samples"] and item.impact == record["impact"]


@pytest.mark.parametrize("kind,expected", JOIN.items())
@pytest.mark.parametrize("language", ["", "javascript"])
def test_join_reasons(kind, expected, language):
    item, = convert_at(1, {"reasons": [reason(kind, language=language)]})
    assert (item.stage, item.scope, item.kind, item.count, item.samples) == (
        "ast_grep", "language" if language else "repo", expected, 2, ["client.py:3"])
    assert "cause=timeout" in item.detail and item.impact


def test_edges_without_position():
    item, = convert_at(1, {"edges_without_position": 7})
    assert (item.stage, item.scope, item.kind, item.path, item.count) == (
        "ast_grep", "repo", "edges_not_checkable", "", 7)
    assert item.detail == "unresolved calls whose callee position or spelling is unusable (opaque expression or position -1) were not checked for external calls"
    assert item.impact == "Some external calls may be missed (for example chained calls such as <expr>.then)."


@pytest.mark.parametrize("index,stage,mapping,impact", [
    (2, "literal", LITERAL, "The URL of these external calls was not examined; host unknown."),
    (3, "constprop", CONSTPROP, "Variable URLs of these calls were not examined; host unknown."),
])
def test_url_failure_outcomes(index, stage, mapping, impact):
    samples = [f"client.py:{i}" for i in range(7)]
    report = {"outcomes": dict.fromkeys(mapping, 7), "samples": dict.fromkeys(mapping, samples)}
    gaps = convert_at(index, report)
    assert {g.kind for g in gaps} == set(mapping.values())
    assert len(gaps) == len(mapping)
    for gap in gaps:
        assert (gap.stage, gap.scope, gap.path, gap.count, gap.samples, gap.impact) == (
            stage, "repo", "", 7, samples[:5], impact)


@pytest.mark.parametrize("index,stage", [(2, "literal"), (3, "constprop")])
def test_url_aborted_flags(index, stage):
    item, = convert_at(index, {"aborted": True, "reasons": [{"cause": "pair_mismatch", "detail": "bad pairing"}]})
    assert (item.stage, item.scope, item.kind, item.count) == (stage, "repo", stage + "_aborted", 1)
    assert "pair_mismatch" in item.detail and "bad pairing" in item.detail


def test_join_aborted():
    gaps = convert_at(1, {"aborted": True, "reasons": [reason("join_aborted", cause="pair_mismatch")]})
    item, = gaps
    assert item.kind == "join_aborted" and "NO edge was marked external" in item.impact
    gaps = convert_at(1, {"aborted": True, "edges_without_position": 2,
                          "reasons": [reason("stage_failed", cause="exception")]})
    assert {g.kind for g in gaps} == {"join_stage_failed", "edges_not_checkable"}
    assert all("NO edge was marked external" in gap.impact for gap in gaps)


def test_none_and_normal_outcomes():
    assert build_external_call_gaps(None, None, None, None) == []
    normal = ["identifier", "relative_url", "env_var", "imported_constant", "multiple_assignments", "parameter",
              "no_url_argument", "spread_before_url", "member_access", "full_url", "partial_url", "port_boundary_host",
              "host_incomplete", "unsupported_authority", "cycle", "chain_too_deep", "value_not_static", "not_found"]
    assert build_external_call_gaps({}, {}, {"outcomes": dict.fromkeys(normal, 3)},
                                    {"outcomes": dict.fromkeys(normal, 4)}) == []
    gaps = build_external_call_gaps({"reasons": [reason("ast_grep_unavailable")]}, None,
                                    {"outcomes": {"node_not_found": 1}}, {})
    assert [g.kind for g in gaps] == ["ast_grep_unavailable", "literal_call_not_found"]
    assert convert_at(1, {"reasons": [reason("regex_fallback_used", count=0)], "edges_without_position": 0}) == []


def test_distinct_causes_survive_merge():
    gaps = convert_at(0, {"reasons": [reason("ast_grep_language_failed", cause=c) for c in ("timeout", "missing_file")]})
    assert len(gaps) == len(merge_gaps(gaps)) == 2


def test_conversion_failure_preserves_prior_gaps():
    class Broken:
        def get(self, *args):
            raise RuntimeError("broken report")
    gaps = build_external_call_gaps({"reasons": [reason("ast_grep_unavailable")]}, Broken(), None, None)
    assert [g.kind for g in gaps] == ["ast_grep_unavailable", "gap_conversion_failed"]
    assert (gaps[-1].stage, gaps[-1].scope, gaps[-1].count, gaps[-1].detail, gaps[-1].impact) == (
        "ast_grep", "repo", 1, "RuntimeError: broken report", "External-call gaps may be incomplete.")


def test_roundtrip_and_no_mutation():
    reports = [{"reasons": [reason("ast_grep_unavailable")]}, {"reasons": [reason("hit_ignored")]},
               {"outcomes": dict.fromkeys(LITERAL, 2), "samples": {"edge_failed": ["a.py:1"]}},
               {"outcomes": dict.fromkeys(CONSTPROP, 1)}]
    before = deepcopy(reports)
    gaps = build_external_call_gaps(*reports)
    assert gaps_from_dicts(gaps_to_dicts(gaps)) == gaps
    assert reports == before
    assert all(g.stage and g.scope and g.kind and g.detail and g.impact and g.count >= 1 for g in gaps)
    gaps[0].samples.append("changed")
    assert reports == before


def real_reports(root):
    records = parse_repo(root, max_files=10000)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for rec in records:
        if rec.language in ("python", "javascript", "typescript", "tsx"):
            groups.setdefault(rec.language, []).append(rec.path)
    hits, run = run_patterns(root, groups)
    joined = join_ast_grep_hits(edges, records, hits, run)
    literal = run_literal_check(edges, records, root)
    constprop = run_constprop_check(edges, records, root, literal)
    assert not joined["aborted"] and not literal["aborted"] and not constprop["aborted"]
    return run, joined, literal, constprop


@pytest.mark.parametrize("folder", ["ast_grep", "literal", "constprop"])
@pytest.mark.parametrize("missing_binary", [False, True])
def test_real_fixture_accounting(folder, missing_binary, monkeypatch, tmp_path):
    if missing_binary:
        monkeypatch.setenv("PEACH_AST_GREP_BIN", str(tmp_path / "no-such-binary"))
    reports = real_reports(Path(__file__).parent / "fixtures" / folder)
    before = deepcopy(reports)
    gaps = build_external_call_gaps(*reports)
    assert before == reports
    assert gaps_from_dicts(gaps_to_dicts(gaps)) == gaps
    assert all(g.stage and g.scope and g.kind and g.detail and g.impact and g.count >= 1 for g in gaps)
    expected = Counter()
    run, joined, literal, constprop = reports
    for r in run["reasons"]:
        expected[("ast_grep", "ast_grep_stage_failed" if r["kind"] == "stage_failed" else r["kind"])] += r["count"]
    expected[("ast_grep", "client_instance_untracked")] += sum(r["count"] for r in run["client_instances"])
    expected[("ast_grep", "edges_not_checkable")] += joined["edges_without_position"]
    for r in joined["reasons"]:
        expected[("ast_grep", JOIN[r["kind"]])] += r["count"]
    for report in (run, joined):
        for r in report["reasons"]:
            if r["count"] > 0:
                detail = f"cause={r['cause']} language={r['language']}: {r['detail']}"
                matches = [g for g in gaps if g.stage == "ast_grep" and g.detail == detail]
                assert sum(g.count for g in matches) == r["count"]
    for stage, report, mapping in (("literal", literal, LITERAL), ("constprop", constprop, CONSTPROP)):
        for outcome, kind in mapping.items():
            expected[(stage, kind)] += report["outcomes"].get(outcome, 0)
    actual = Counter()
    for gap in gaps:
        actual[(gap.stage, gap.kind)] += gap.count
    assert dict(actual) == {key: count for key, count in expected.items() if count > 0}
    if missing_binary:
        assert "ast_grep_unavailable" in {g.kind for g in gaps}
        assert "regex_fallback_used" in {g.kind for g in gaps}
    else:
        assert run["binary"] is not None and run["reasons"] == []


def test_config_file_unparsed_gap():
    config = {"unparsed_files": ["charts/template.yaml"],
              "unparsed_errors": {"charts/template.yaml": "ConstructorError: unhashable key"},
              "outcomes": {"no_match": 2, "multiple_values": 3, "value_not_url": 1}}
    before = deepcopy(config)
    item, = build_external_call_gaps(None, None, None, None, config)
    assert (item.stage, item.scope, item.kind, item.path, item.count) == (
        "config", "file", "config_file_unparsed", "charts/template.yaml", 1)
    assert item.detail == "YAML could not be parsed (for example a Helm template): ConstructorError: unhashable key"
    assert item.impact == "Config values defined in this file are unknown; env-based URLs of external calls may stay unresolved."
    assert gaps_from_dicts(gaps_to_dicts([item])) == [item]
    assert config == before


def test_config_none_and_normal_outcomes_are_not_gaps():
    assert build_external_call_gaps(None, None, None, None, config_report=None) == []
    assert build_external_call_gaps(None, None, None, None, {"outcomes": {
        "no_match": 1, "multiple_values": 1, "value_not_url": 1, "defined_value_unknown": 1}}) == []


def test_config_conversion_failure_preserves_existing_gaps():
    class Broken:
        def get(self, *args):
            raise RuntimeError("bad config report")
    gaps = build_external_call_gaps({"reasons": [reason("ast_grep_unavailable")]}, None, None, None, Broken())
    assert [g.kind for g in gaps] == ["ast_grep_unavailable", "gap_conversion_failed"]
