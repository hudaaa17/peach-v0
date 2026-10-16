"""Literal inspection through the real parser, resolver, runner and join."""

from copy import deepcopy
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from analyzer_new.parser import parse_repo
from analyzer_new.resolver import resolve_calls
from externalcalls.ast_grep_check import run_patterns
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls.literal_check import run_literal_check


FIELDS = ('literal', 'host', 'needs_review', 'review_reason')
ALLOWED = set(FIELDS) | {'literal_method'}
UNCHANGED = (None, None, False, None)
FULL = lambda text, host='a.com': (text, host, False, None)
PARTIAL = ('https://a.com/', 'a.com', True, 'partial_url')
INCOMPLETE = ('https://a.', None, True, 'host_incomplete')
EXPECTED = {
    1: [FULL('https://api.stripe.com/v1/charges', 'api.stripe.com')],
    2: [FULL('https://a.com/x')], 3: [FULL('https://a.com/x')], 4: [FULL('https://a.com/x')],
    5: [PARTIAL], 6: [INCOMPLETE], 7: [PARTIAL], 8: [INCOMPLETE],
    9: [('/users', None, True, 'relative_url')], 10: [PARTIAL],
    11: [UNCHANGED], 12: [UNCHANGED], 13: [UNCHANGED], 14: [UNCHANGED], 15: [UNCHANGED],
    16: [UNCHANGED, UNCHANGED, UNCHANGED], 17: [UNCHANGED], 18: [UNCHANGED], 19: [UNCHANGED],
    20: [FULL('https://a.com/x')], 21: [FULL('https://a.com/y')], 22: [FULL('https://a.com/z')],
    23: [UNCHANGED], 24: [PARTIAL], 25: [PARTIAL], 26: [UNCHANGED],
    27: [FULL('https://a.com/x')], 28: [FULL('https://a.com/z')],
    29: [FULL('https://a.com:8080/x')],
    30: [FULL('https://one.com/x', 'one.com'), FULL('https://two.com/y', 'two.com')],
    31: [FULL('https://unicode.com/x', 'unicode.com')],
    32: [FULL('https://crlf.com/x', 'crlf.com')],
    33: [UNCHANGED], 34: [UNCHANGED], 35: [('preserved', None, False, None)],
}


def accounting(report):
    assert sum(report['outcomes'].values()) == report['edges_considered']
    assert sum(sum(counts.values()) for counts in report['per_pattern'].values()) == report['edges_considered']
    assert all(len(samples) <= 5 for samples in report['samples'].values())
    assert report['already_set'] == report['outcomes'].get('already_set', 0)


def unchanged_other_fields(before, edges):
    for old, edge in zip(before, edges):
        assert {k: v for k, v in asdict(edge).items() if k not in ALLOWED} == {
            k: v for k, v in asdict(old).items() if k not in ALLOWED}


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp('literal')
    fixtures = Path(__file__).parent / 'fixtures' / 'literal'
    for path in fixtures.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    crlf = root / 'crlf.py'
    crlf.write_bytes(crlf.read_bytes().replace(b'\n', b'\r\n'))
    assert b'\r\n' in crlf.read_bytes()
    gaps = []
    records = parse_repo(root, gaps=gaps)
    assert all(rec.parse_mode == 'treesitter' and not rec.has_syntax_errors for rec in records)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for rec in records:
        groups.setdefault(rec.language, []).append(rec.path)
    hits, runner_report = run_patterns(root, groups)
    assert runner_report['binary'] is not None
    assert runner_report['reasons'] == []
    joined = join_ast_grep_hits(edges, records, hits, runner_report)
    assert not joined['aborted']
    assert joined['hits_unmatched'] == 0
    assert joined['hits_marked'] == len(hits)
    cases = {}
    for index, edge in enumerate(edges):
        if edge.status != 'external_candidate':
            continue
        line = (root / edge.file).read_text(encoding='utf-8').splitlines()[edge.line - 1]
        number = int(line.rsplit('#' if edge.file.endswith('.py') else '//', 1)[1].strip())
        cases.setdefault(number, []).append(index)
    assert set(cases) == set(range(1, 33))
    return root, records, edges, cases


@pytest.mark.parametrize('case', range(1, 36))
def test_fixture_case(pipeline, case):
    root, records, original, cases = pipeline
    edges = deepcopy(original)
    if case <= 32:
        indices = cases[case]
    else:
        indices = cases[1]
        edge = edges[indices[0]]
        if case == 33:
            edge.callee_expr = 'altered'
        elif case == 34:
            edge.status = 'internal'
        else:
            edge.literal = 'preserved'
    before = deepcopy(edges)
    report = run_literal_check(edges, records, root)
    accounting(report)
    actual = [tuple(getattr(edges[i], field) for field in FIELDS) for i in indices]
    assert actual == EXPECTED[case]
    unchanged_other_fields(before, edges)
    if case == 33:
        assert report['aborted'] and report['reasons'][0]['cause'] == 'pair_mismatch'
        assert edges == before
    else:
        assert not report['aborted']
        assert report['reasons'] == []
    for index in indices:
        edge = edges[index]
        if case in (12, 13, 14, 23):
            pending = report['pending_constprop'][index]
            assert pending['name'] == ('BASE' if case in (13, 14) else 'url')
            from externalcalls.ast_access import find_call_node, get_url_argument, load_tree
            from externalcalls.ast_grep_check import CATALOGUE
            calls = [call for rec in records for call in rec.calls]
            call = calls[index]
            tree, source, error = load_tree(root, edge.file, {})
            assert error is None
            node = find_call_node(tree, call.callee_line, call.callee_col)
            entry = next(e for e in CATALOGUE if e.id == edge.external_pattern)
            argument, error = get_url_argument(node, entry, 'python' if edge.file.endswith('.py') else 'javascript')
            assert error is None
            assert (pending['arg_line'], pending['arg_col']) == (argument.start_point[0] + 1, argument.start_point[1])
        if case in (16, 26):
            assert report['env_refs'][index] == {'var': 'X'}
            assert edge == before[index]
        if case <= 32 and edge.literal is not None:
            assert edge.literal_method == 'literal'
            assert report['resolved_edges'][index] == {
                'literal': edge.literal, 'host': edge.host,
                'partial': case in (5, 6, 7, 8, 10, 24, 25)}
        if case == 34:
            assert edge == before[index]
            assert index not in report['resolved_edges']
        if case == 35:
            assert edge == before[index]
            assert report['already_set'] == 1
    for kind in ('bytes_literal', 'identifier', 'member_access', 'env_var', 'call_expression',
                 'no_url_argument', 'spread_before_url'):
        if case <= 32:
            assert report['outcomes'][kind] > 0
    print('FIXTURE', case, 'PASS', json.dumps(actual, ensure_ascii=False))


def test_deleted_file(pipeline, tmp_path):
    root, records, original, cases = pipeline
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    (tmp_path / 'client.py').unlink()
    edges = deepcopy(original)
    report = run_literal_check(edges, records, tmp_path)
    assert not report['aborted']
    assert report['outcomes']['file_unreadable'] == sum(e.file == 'client.py' and e.status == 'external_candidate' for e in edges)
    for old, edge in zip(original, edges):
        if edge.file == 'client.py':
            assert edge == old
    assert any(e.literal for e in edges if e.file != 'client.py')
    accounting(report)


def test_telemetry_failure(pipeline):
    root, records, original, _ = pipeline
    class Broken:
        def bump(self, *args):
            raise RuntimeError('telemetry failed')
    expected, actual = deepcopy(original), deepcopy(original)
    report = run_literal_check(expected, records, root)
    assert run_literal_check(actual, records, root, Broken()) == report
    assert actual == expected
    accounting(report)


@pytest.mark.parametrize('failure', ['grammar_unavailable', 'node_not_found', 'ambiguous_node', 'edge_failed', 'no_position'])
def test_edge_failures(pipeline, failure):
    root, original_records, original, cases = pipeline
    records, edges = deepcopy(original_records), deepcopy(original)
    first = cases[1][0]
    from externalcalls import literal_check
    if failure == 'no_position':
        calls = [c for rec in records for c in rec.calls]
        calls[first].callee_col = -1
        report = run_literal_check(edges, records, root)
    else:
        function = 'load_tree' if failure == 'grammar_unavailable' else 'find_call_node'
        real = getattr(literal_check, function)
        def fail(*args):
            if function == 'load_tree' and args[1] == 'client.py':
                return None, None, failure
            if function == 'find_call_node' and tuple(args[1:]) == (6, 9):
                if failure == 'edge_failed':
                    raise RuntimeError('injected edge failure')
                return failure
            return real(*args)
        with patch.object(literal_check, function, side_effect=fail):
            report = run_literal_check(edges, records, root)
    assert not report['aborted']
    assert report['outcomes'][failure] >= 1
    assert edges[first] == original[first]
    assert any(e.literal for e in edges if e.file == 'client.js')
    accounting(report)


def test_count_mismatch_and_malformed(pipeline):
    root, records, original, _ = pipeline
    edges = deepcopy(original[:-1])
    before = deepcopy(edges)
    report = run_literal_check(edges, records, root)
    assert report['aborted'] and report['reasons'][0]['cause'] == 'edge_count_mismatch'
    assert edges == before
    accounting(report)
    assert run_literal_check(None, None, root)['aborted']
    assert run_literal_check(original, [object()], root)['aborted']
    assert run_literal_check([], [], root)['outcomes'] == {}


def test_eligibility_and_sample_cap(pipeline):
    root, records, original, cases = pipeline
    edges = deepcopy(original)
    for index in cases[1] + cases[2]:
        edges[index].external_pattern = 'unknown' if index in cases[1] else 'py.requests.Session'
    before = deepcopy(edges)
    report = run_literal_check(edges, records, root)
    for index in cases[1] + cases[2]:
        assert edges[index] == before[index]
    assert report['edges_considered'] == sum(e.status == 'external_candidate' for e in original) - 2
    assert len(report['samples']['full_url']) == 5
    accounting(report)


def test_exact_outcomes_and_pending(pipeline):
    root, records, original, cases = pipeline
    edges = deepcopy(original)
    report = run_literal_check(edges, records, root)
    outcomes = {5: 'partial_url', 6: 'host_incomplete', 7: 'partial_url', 8: 'host_incomplete',
                9: 'relative_url', 10: 'partial_url', 11: 'bytes_literal', 12: 'identifier',
                13: 'identifier', 14: 'identifier', 15: 'member_access', 16: 'env_var',
                17: 'call_expression', 18: 'no_url_argument', 19: 'spread_before_url',
                23: 'identifier', 24: 'partial_url', 25: 'partial_url', 26: 'env_var'}
    expected = Counter()
    patterns = {}
    for case, indices in cases.items():
        outcome = outcomes.get(case, 'full_url')
        for index in indices:
            expected[outcome] += 1
            patterns.setdefault(edges[index].external_pattern, Counter())[outcome] += 1
    assert report['outcomes'] == dict(expected)
    assert report['per_pattern'] == {key: dict(value) for key, value in patterns.items()}
    pending_cases = {6: 'env', 8: 'env', 12: 'url', 13: 'BASE', 14: 'BASE', 23: 'url'}
    assert set(report['pending_constprop']) == {i for case in pending_cases for i in cases[case]}
    for case, name in pending_cases.items():
        for index in cases[case]:
            assert report['pending_constprop'][index]['name'] == name
    assert set(report['env_refs']) == set(cases[16] + cases[26])
    assert report['edges_considered'] == sum(len(indices) for indices in cases.values())
    accounting(report)


def test_userinfo_redaction_counter(pipeline):
    root, records, original, cases = pipeline
    edges = deepcopy(original)
    report = run_literal_check(edges, records, root)
    assert report['userinfo_redacted'] == 1
    assert edges[cases[29][0]].literal == 'https://a.com:8080/x'
    assert report['resolved_edges'][cases[29][0]]['literal'] == 'https://a.com:8080/x'
    accounting(report)
