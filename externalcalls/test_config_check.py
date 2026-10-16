"""Handwritten config fixtures with real parser/resolver/external-call edges."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analyzer_new.parser import FileRecord, parse_repo
from analyzer_new.resolver import resolve_calls
from externalcalls.ast_grep_check import run_patterns
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls.config_check import _config_table, run_config_check
from externalcalls.constprop_check import run_constprop_check
from externalcalls.literal_check import run_literal_check


FIXTURES = Path(__file__).parent / "fixtures" / "config"
EXPECTED = {
    "DIRECT": ("api.stripe.com", "config_host_set", ["01-deployment.yaml"]),
    "CM_URL": ("configmap.example.com", "config_host_set", ["02-configmap.yaml"]),
    "FROM_URL": ("envfrom.example.com", "config_host_set", ["03-envfrom.yaml"]),
    "PRE_FROM_URL": ("envfrom.example.com", "config_host_set", ["03-envfrom.yaml"]),
    "MULTIDOC": ("multidoc.example.com", "config_host_set", ["04-multidoc.yaml"]),
    "STATEFUL": ("stateful.example.com", "config_host_set", ["05-stateful.yaml"]),
    "CRON": ("cron.example.com", "config_host_set", ["05-cron.yaml"]),
    "INIT": ("init.example.com", "config_host_set", ["05-init.yaml"]),
    "SECRET_URL": (None, "defined_value_unknown", ["06-secret.yaml"]),
    "HELM": (None, "no_match", []),
    "COMPOSE_MAP": ("compose-map.example.com", "config_host_set", ["08-compose-map.yml"]),
    "COMPOSE_LIST": ("compose-list.example.com", "config_host_set", ["08-compose-list.yml"]),
    "COMPOSE_BARE": (None, "no_match", []),
    "DOT_EXPORT": ("export.example.com", "config_host_set", [".env.example"]),
    "DOT_QUOTED": ("quoted.example.com", "config_host_set", [".env.example"]),
    "DOT_EMPTY": (None, "value_not_url", [".env.example"]),
    "SAME": ("same.example.com", "config_host_set", ["10-a.env", "10-b.env"]),
    "DIFFERENT": (None, "multiple_values", ["11-dev.env", "11-prod.env"]),
    "USERINFO": ("host", "config_host_set", ["12-userinfo.env"]),
    "DYNAMIC_PORT": ("a.example.com", "config_host_set", ["13-port.env"]),
    "DYNAMIC_HOST": (None, "value_not_url", ["14-host.env"]),
    "NO_SCHEME": (None, "value_not_url", ["15-relative.env"]),
    "UNREFERENCED": (None, "no_match", []),
    "MISSING_MAP": (None, "defined_value_unknown", ["17-missing.yaml"]),
    "PROPAGATED": ("propagated.example.com", "config_host_set", [".env.example"]),
    "SHARED": ("shared.example.com", "config_host_set", [".env.example"]),
    "LIST_TOP": (None, "no_match", []),
    None: (None, "no_var_name", []),
}


def pipeline(root):
    records = parse_repo(root, max_files=10000)
    assert all(r.is_config or (r.parse_mode == "treesitter" and not r.has_syntax_errors) for r in records)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for rec in records:
        if not rec.is_config:
            groups.setdefault(rec.language, []).append(rec.path)
    hits, run = run_patterns(root, groups)
    assert run["binary"] is not None and not run["reasons"]
    joined = join_ast_grep_hits(edges, records, hits, run)
    assert not joined["aborted"] and not joined["hits_unmatched"]
    literal = run_literal_check(edges, records, root)
    constprop = run_constprop_check(edges, records, root, literal)
    assert not literal["aborted"] and not constprop["aborted"]
    return records, edges, literal, constprop


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("config")
    for path in FIXTURES.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    records, edges, literal, constprop = pipeline(root)
    cases = {}
    for index, edge in enumerate(edges):
        if edge.status == "external_candidate":
            line = (root / edge.file).read_text().splitlines()[edge.line - 1]
            case = int(line.rsplit("#" if edge.file.endswith(".py") else "//", 1)[1])
            cases.setdefault(case, []).append(index)
    assert set(cases) == set(range(1, 26))
    assert sum(r.is_config for r in records) == 25
    assert any(ref["var"] == "PROPAGATED" for ref in constprop["env_refs"].values())
    return root, records, edges, literal, constprop, cases


class BrokenTelemetry:
    def bump(self, *args):
        raise RuntimeError("telemetry unavailable")


@pytest.mark.parametrize("case", range(1, 26))
def test_fixture_case(fixtures, case):
    root, records, original, literal, constprop, cases = fixtures
    edges = deepcopy(original)
    chosen = cases[case]
    lit = {"env_refs": {i: v for i, v in literal["env_refs"].items() if i in chosen}}
    prop = {"env_refs": {i: v for i, v in constprop["env_refs"].items() if i in chosen}}
    if case == 19:
        edges[chosen[0]].status = "unresolved"
        edges[chosen[1]].host = "already.example.com"
    if case == 25:
        edges[chosen[0]].config_matches = ["existing.yaml", "01-deployment.yaml"]
    before = deepcopy(edges)
    input_reports = deepcopy((lit, prop))
    report = run_config_check(edges, records, None if case == 20 else lit,
                              None if case == 20 else prop, BrokenTelemetry() if case == 24 else None)
    assert (lit, prop) == input_reports
    assert not report["reasons"] and not report["edge_errors"] and not report["file_errors"]
    assert sum(report["outcomes"].values()) == report["edges_considered"] == (0 if case == 20 else len(chosen))
    assert all(len(samples) <= 5 for samples in report["samples"].values())
    assert report["unparsed_files"] == ["07-helm.yaml"]
    assert "ConstructorError" in report["unparsed_errors"]["07-helm.yaml"]
    assert report["secret_documents"] == report["secret_or_field_ref_skipped"] == report["configmap_not_in_repo"] == 1
    assert report["config_files_total"] == 25
    assert "secret-must-never-be-stored" not in json.dumps(report)
    allowed = {"host", "needs_review", "review_reason", "config_matches"}
    for index, edge in enumerate(edges):
        assert {k: v for k, v in asdict(edge).items() if k not in allowed} == {
            k: v for k, v in asdict(before[index]).items() if k not in allowed}
        if index not in chosen or case in (19, 20):
            assert edge == before[index]
    rows = []
    for index in chosen:
        edge = edges[index]
        var = lit["env_refs"].get(index, prop["env_refs"].get(index, {})).get("var")
        if case in (19, 20):
            outcome = "skipped_not_candidate" if case == 19 else "not_considered"
            if case == 19:
                assert report["outcomes"] == {outcome: len(chosen)}
        else:
            host, outcome, paths = EXPECTED[var]
            assert edge.host == host
            if case == 25:
                assert edge.config_matches == ["existing.yaml", "01-deployment.yaml"]
            else:
                assert set(edge.config_matches) == set(paths) and len(edge.config_matches) == len(paths)
            assert f"{edge.file}:{edge.line}" in report["samples"][outcome]
            if host:
                assert edge.needs_review is True and edge.review_reason == "config_value"
                assert report["resolved_edges"][index]["host"] == host
            if case == 11:
                assert set(report["multi_value_edges"][index]["values"]) == {"https://dev.example.com", "https://prod.example.com"}
            if case == 12:
                assert report["resolved_edges"][index]["value"] == "https://host/x"
                assert "user:pw@" not in json.dumps(report)
            if case == 14:
                assert report["value_not_url"] == {"host_incomplete": 1}
            if case == 15:
                assert report["value_not_url"] == {"relative_url": 1}
        rows.append({"case": case, "file": edge.file, "line": edge.line, "host": edge.host,
                     "outcome": outcome, "config_matches": edge.config_matches})
    print("CONFIG_CASE " + json.dumps(rows))


def record(path, text):
    return FileRecord(path=path, language="config", is_config=True, source_lines=text.splitlines())


def candidate(fixtures):
    root, records, original, literal, constprop, cases = fixtures
    index = cases[1][0]
    return deepcopy(original[index])


def test_literal_wins_and_review_reason_preserved(fixtures):
    edge = candidate(fixtures)
    edge.review_reason = "keep_existing_reason"
    report = run_config_check([edge], [record(".env", "X=https://x.example.com\nY=https://y.example.com")],
                              {"env_refs": {0: {"var": "X"}}}, {"env_refs": {0: {"var": "Y"}}})
    assert report["edges_considered"] == 1 and edge.host == "x.example.com"
    assert edge.review_reason == "keep_existing_reason" and edge.needs_review


@pytest.mark.parametrize("marker", ["$(PORT)", "${PORT}", "{{port}}"])
def test_interpolation_cut(fixtures, marker):
    edge = candidate(fixtures)
    report = run_config_check([edge], [record(".env", "X=https://host.example.com:" + marker)],
                              {"env_refs": {0: {"var": "X"}}}, None)
    assert edge.host == "host.example.com" and edge.review_reason == "config_value"
    assert report["outcomes"] == {"config_host_set": 1}


def test_one_parse_per_yaml_and_invalid_later_document_is_atomic(fixtures):
    edge = candidate(fixtures)
    import externalcalls.config_check as module
    real_loader = module.yaml.safe_load_all
    records = [record("broken.yaml", 'services: {a: {environment: {X: "https://should-not-be-used.example"}}}\n---\nvalue: {{ invalid }}'),
               record("good.yaml", 'services: {a: {environment: {X: "https://good.example.com"}}}')]
    with patch.object(module.yaml, "safe_load_all", wraps=real_loader) as loader:
        report = run_config_check([edge], records, {"env_refs": {0: {"var": "X"}}}, None)
    assert loader.call_count == 2
    assert edge.host == "good.example.com" and edge.config_matches == ["good.yaml"]
    assert report["unparsed_files"] == ["broken.yaml"]


def test_non_string_scalars_and_unknowns(fixtures):
    text = '''services:
  a:
    environment: {NUMBER: 8080, BOOL: true, UNKNOWN: null}
'''
    edges = [candidate(fixtures) for _ in range(3)]
    refs = {i: {"var": name} for i, name in enumerate(("NUMBER", "BOOL", "UNKNOWN"))}
    report = run_config_check(edges, [record("compose.yaml", text)], {"env_refs": refs}, None)
    assert report["outcomes"] == {"value_not_url": 2, "defined_value_unknown": 1}
    assert report["value_not_url"] == {"relative_url": 2}
    assert all(e.host is None and e.config_matches == ["compose.yaml"] for e in edges)


def test_file_and_edge_errors_continue(fixtures):
    class BrokenFile:
        is_config = True
        path = "broken.env"
        @property
        def source_lines(self):
            raise RuntimeError("read failed")
    edge = candidate(fixtures)
    records = [BrokenFile(), record(".env", "X=https://good.example.com")]
    report = run_config_check([edge], records, {"env_refs": {99: {"var": "X"}, 0: {"var": "X"}}}, None)
    assert report["file_failed"] == 1 and len(report["file_errors"]) == 1
    assert report["outcomes"] == {"edge_failed": 1, "config_host_set": 1}
    assert sum(report["outcomes"].values()) == report["edges_considered"] == 2
    assert edge.host == "good.example.com"


def test_no_expansion_and_no_regexp_or_tree_sitter_in_stage():
    source = (Path(__file__).parent / "config_check.py").read_text()
    assert "import re\n" not in source and "tree_sitter" not in source
    report = run_config_check([], [record(".env", 'X="$UNEXPANDED"')], None, None)
    table = _config_table([record(".env", 'X="$UNEXPANDED"')], report)
    assert table["X"] == [{"value": "$UNEXPANDED", "path": ".env", "source": "dotenv"}]


def test_edge_mutation_failure_rolls_back_and_continues(fixtures):
    original = candidate(fixtures)
    class RejectReview(type(original)):
        def __setattr__(self, name, value):
            if name == "needs_review" and value is True:
                raise RuntimeError("review setter failed")
            super().__setattr__(name, value)
    broken = RejectReview(**asdict(original))
    good = candidate(fixtures)
    report = run_config_check([broken, good], [record(".env", "X=https://good.example.com")],
                              {"env_refs": {0: {"var": "X"}, 1: {"var": "X"}}}, None)
    assert asdict(broken) == asdict(original)
    assert good.host == "good.example.com"
    assert report["outcomes"] == {"edge_failed": 1, "config_host_set": 1}
    assert set(report["resolved_edges"]) == {1}


def test_field_refs_secret_envfrom_and_samples(fixtures):
    config = record("pod.yaml", '''apiVersion: v1
kind: Pod
spec:
  containers:
  - env:
    - name: FIELD
      valueFrom: {fieldRef: {fieldPath: metadata.name}}
    - name: RESOURCE
      valueFrom: {resourceFieldRef: {resource: limits.cpu}}
    envFrom: [{secretRef: {name: hidden}}]
''')
    edges = [candidate(fixtures) for _ in range(9)]
    for i, edge in enumerate(edges):
        edge.line = i + 1
    refs = {i: {"var": "FIELD" if i % 2 else "RESOURCE"} for i in range(9)}
    report = run_config_check(edges, [config], {"env_refs": refs}, None)
    assert report["outcomes"] == {"defined_value_unknown": 9}
    assert report["secret_or_field_ref_skipped"] == 3
    assert len(report["samples"]["defined_value_unknown"]) == 5
    assert all(edge.host is None and edge.config_matches == ["pod.yaml"] for edge in edges)


def test_redaction_precedes_value_deduplication(fixtures):
    edge = candidate(fixtures)
    report = run_config_check([edge], [record("a.env", "X=https://one:pw@host/x"),
                                      record("b.env", "X=https://two:pw@host/x")],
                              {"env_refs": {0: {"var": "X"}}}, None)
    assert report["outcomes"] == {"config_host_set": 1}
    assert edge.host == "host" and edge.config_matches == ["a.env", "b.env"]
    assert report["resolved_edges"][0]["value"] == "https://host/x"
    assert "pw@" not in json.dumps(report)
