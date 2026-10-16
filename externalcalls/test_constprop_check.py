"""Real-pipeline constant propagation, conservative bindings and failure isolation."""

from copy import deepcopy
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
from externalcalls.constprop_check import run_constprop_check


ALLOWED = {'literal', 'literal_method', 'constprop_via', 'host', 'needs_review', 'review_reason'}
FAILURES = {5: 'cycle', 6: 'chain_too_deep', 7: 'multiple_assignments', 8: 'multiple_assignments',
            10: 'parameter', 11: 'imported_constant', 12: 'conditional_assignment',
            13: 'assigned_after_use', 14: 'complex_binding', 15: 'complex_binding',
            16: 'complex_binding', 17: 'no_value', 21: 'value_not_static',
            22: 'no_host_after_constprop', 23: 'not_found', 24: 'not_found', 25: 'multiple_assignments'}


def pipeline(root):
    records = parse_repo(root)
    assert all(r.parse_mode == 'treesitter' and not r.has_syntax_errors for r in records)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for rec in records:
        groups.setdefault(rec.language, []).append(rec.path)
    hits, runner = run_patterns(root, groups)
    assert runner['binary'] is not None and runner['reasons'] == []
    joined = join_ast_grep_hits(edges, records, hits, runner)
    assert not joined['aborted'] and joined['hits_unmatched'] == 0
    literal = run_literal_check(edges, records, root)
    assert not literal['aborted']
    return records, edges, literal


def accounting(report):
    assert not report['aborted']
    assert sum(report['outcomes'].values()) == report['edges_considered']
    assert all(len(samples) <= 5 for samples in report['samples'].values())


def preserve(before, edges):
    for old, edge in zip(before, edges):
        assert {k: v for k, v in asdict(old).items() if k not in ALLOWED} == {
            k: v for k, v in asdict(edge).items() if k not in ALLOWED}


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp('constprop')
    for path in (Path(__file__).parent / 'fixtures' / 'constprop').iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    # Exercise all shared JS binding cases with the TS grammar as well.
    shutil.copyfile(root / 'client.js', root / 'expanded.ts')
    crlf = root / 'crlf.py'
    crlf.write_bytes(crlf.read_bytes().replace(b'\n', b'\r\n'))
    assert b'\r\n' in crlf.read_bytes()
    records, edges, literal = pipeline(root)
    cases = {}
    for index, edge in enumerate(edges):
        if edge.status == 'external_candidate':
            line = (root / edge.file).read_text(encoding='utf-8').splitlines()[edge.line - 1]
            case = int(line.rsplit('#' if edge.file.endswith('.py') else '//', 1)[1].strip())
            cases.setdefault(case, []).append(index)
    assert set(cases) == set(range(1, 28)) | {30}
    return root, records, edges, literal, cases


@pytest.mark.parametrize('case', list(range(1, 28)) + [30])
def test_fixture_case(fixtures, case):
    root, records, original, literal, cases = fixtures
    edges = deepcopy(original)
    report = run_constprop_check(edges, records, root, literal)
    accounting(report)
    preserve(original, edges)
    rows = []
    for index in cases[case]:
        edge = edges[index]
        if case in FAILURES:
            assert report['still_pending'][index] == FAILURES[case]
            assert edge == original[index]
            assert index not in report['resolved_edges']
            if case == 11:
                assert report['imported_constants'][index] in ('IMPORTED11', 'REQUIRE11')
            if case == 21:
                assert report['env_refs'][index] == {'var': 'X'}
                assert {'cause': 'value_not_static', 'detail': 'env_var',
                        'sample': f'{edge.file}:{edge.line}'} in report['reasons']
            if case == 22:
                assert {'cause': 'no_host_after_constprop', 'detail': 'host_incomplete',
                        'sample': f'{edge.file}:{edge.line}'} in report['reasons']
        else:
            expected = ('https://module.com/x' if case in (2, 3) else
                        'https://local.com/x' if case == 9 else
                        'https://a.com' if case == 4 else
                        'https://a.com/users' if case == 19 else
                        'https://a.com:8080/x' if case == 30 else
                        'https://crlf.com/x' if edge.file == 'crlf.py' else
                        'https://unicode.com/x' if case == 27 else 'https://a.com/x')
            host = ('module.com' if case in (2, 3) else 'local.com' if case == 9 else
                    'crlf.com' if edge.file == 'crlf.py' else 'unicode.com' if case == 27 else 'a.com')
            assert (edge.literal, edge.host, edge.literal_method, edge.constprop_via,
                    edge.needs_review, edge.review_reason) == (expected, host, 'const_prop', 'ast', False, None)
            assert report['resolved_edges'][index]['partial'] is False
            assert index not in report['still_pending']
            if case == 4:
                assert report['resolved_edges'][index]['chain'] == ['b', 'a']
        rows.append(dict(file=edge.file, literal=edge.literal, host=edge.host,
                         needs_review=edge.needs_review, review_reason=edge.review_reason,
                         outcome=report['still_pending'].get(index, 'full_url')))
    assert report['userinfo_redacted'] == len(cases[30])
    print('CONSTPROP_CASE', case, 'PASS', json.dumps(rows))


def test_case28_skipped(fixtures):
    root, records, original, literal, cases = fixtures
    edges = deepcopy(original)
    modified = deepcopy(literal)
    indices = [cases[n][0] for n in (1, 2, 3)]
    modified['pending_constprop'].pop(indices[0])
    edges[indices[1]].host = 'already.com'
    edges[indices[2]].status = 'internal'
    before = deepcopy(edges)
    report = run_constprop_check(edges, records, root, modified)
    accounting(report)
    for index in indices:
        assert edges[index] == before[index]
        assert index not in report['resolved_edges']
    assert report['still_pending'][indices[1]] == 'skipped_not_pending'
    assert report['still_pending'][indices[2]] == 'skipped_not_pending'
    assert report['outcomes']['skipped_not_pending'] >= 3


@pytest.mark.parametrize('mismatch', ['pair', 'count'])
def test_case29_pairing(fixtures, mismatch):
    root, records, original, literal, _ = fixtures
    edges = deepcopy(original)
    if mismatch == 'pair':
        edges[0].callee_expr = 'altered'
    else:
        edges.pop()
    before = deepcopy(edges)
    report = run_constprop_check(edges, records, root, literal)
    assert report['aborted']
    assert report['reasons'][0]['cause'] == ('pair_mismatch' if mismatch == 'pair' else 'edge_count_mismatch')
    assert report['edges_considered'] == 0
    assert edges == before


def test_deleted_file(fixtures, tmp_path):
    root, records, original, literal, _ = fixtures
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    (tmp_path / 'client.py').unlink()
    edges = deepcopy(original)
    report = run_constprop_check(edges, records, tmp_path, literal)
    accounting(report)
    assert report['outcomes']['file_unreadable'] > 0
    for old, edge in zip(original, edges):
        if edge.file == 'client.py':
            assert edge == old
    assert any(e.literal_method == 'const_prop' for e in edges if e.file == 'client.js')


def test_telemetry(fixtures):
    root, records, original, literal, _ = fixtures
    class Broken:
        def bump(self, *args):
            raise RuntimeError('telemetry')
    expected, actual = deepcopy(original), deepcopy(original)
    report = run_constprop_check(expected, records, root, literal)
    assert run_constprop_check(actual, records, root, literal, Broken()) == report
    assert actual == expected
    accounting(report)


@pytest.mark.parametrize('failure', ['grammar_unavailable', 'argument_not_found', 'edge_failed'])
def test_failure_isolation(fixtures, failure):
    root, records, original, literal, _ = fixtures
    edges = deepcopy(original)
    literal = deepcopy(literal)
    if failure == 'argument_not_found':
        for index, info in literal['pending_constprop'].items():
            if edges[index].file.endswith('.py'):
                info['arg_line'] = 99999
    from externalcalls import constprop_check
    function = 'load_tree' if failure == 'grammar_unavailable' else 'find_node_at'
    real = getattr(constprop_check, function)
    def injected(*args):
        if function == 'load_tree' and args[1] == 'client.py':
            return None, None, failure
        if function == 'find_node_at' and args[0].root_node.type == 'module':
            if failure == 'argument_not_found':
                return None
            raise RuntimeError('edge failure')
        return real(*args)
    with patch.object(constprop_check, function, side_effect=injected):
        report = run_constprop_check(edges, records, root, literal)
    accounting(report)
    assert report['outcomes'][failure] > 0
    for old, edge in zip(original, edges):
        if edge.file.endswith('.py') and (failure != 'grammar_unavailable' or edge.file == 'client.py'):
            assert edge == old
    assert any(e.literal_method == 'const_prop' for e in edges if e.file == 'client.js')


def test_malformed_and_empty():
    assert run_constprop_check(None, None, '.', {})['aborted']
    assert run_constprop_check([], [], '.', {})['outcomes'] == {}


@pytest.mark.parametrize('suffix,source,outcome,detail', [
    ('.py', 'import requests\ndef f():\n global url\n requests.get(url)\n', 'complex_binding', None),
    ('.py', 'import requests\ndef f():\n del url\n requests.get(url)\n', 'complex_binding', None),
    ('.py', 'import requests\ndef f():\n with ctx as url: pass\n requests.get(url)\n', 'complex_binding', None),
    ('.py', 'import requests\ndef f():\n (url := "https://a.com")\n requests.get(url)\n', 'complex_binding', None),
    ('.py', 'import requests\ndef f():\n url: str = "https://a.com/x"\n requests.get(url)\n', 'full_url', None),
    ('.py', 'import requests\ndef f(url: str):\n requests.get(url)\n', 'parameter', None),
    ('.py', 'import requests\ndef f():\n url = get_url()\n requests.get(url)\n', 'value_not_static', 'call_expression'),
    ('.py', 'import requests\ndef unrelated():\n OTHER = "https://a.com"\ndef f():\n requests.get(OTHER)\n', 'not_module_level', None),
    ('.js', 'function f(){ try {} catch(url) { fetch(url); } }', 'complex_binding', None),
    ('.js', 'function f(){ const [url] = values; fetch(url); }', 'complex_binding', None),
    ('.js', 'function f(){ const {url = "x"} = values; fetch(url); }', 'complex_binding', None),
    ('.js', 'const f = url => fetch(url);', 'parameter', None),
    ('.js', 'function f(){ url = "https://a.com/x"; fetch(url); }', 'full_url', None),
    ('.js', 'function f(){ const BASE = "https://a.com"; fetch(BASE + "/" + dynamic); }', 'partial_url', None),
    ('.py', 'import requests\ndef f():\n BASE = "https://a.com"\n requests.get(BASE + "/" + dynamic)\n', 'partial_url', None),
    ('.js', 'function f(){ const BASE = "/users"; fetch(BASE); }', 'no_host_after_constprop', 'relative_url'),
    ('.js', 'function f(){ const BASE = "https://[::1]/"; fetch(BASE); }', 'no_host_after_constprop', 'unsupported_authority'),
    ('.py', 'import requests\ndef f():\n url = "https://a." + env\n requests.get(url)\n', 'no_host_after_constprop', 'host_incomplete'),
    ('.js', 'function f(){ const url = "https://a." + env; fetch(url); }', 'no_host_after_constprop', 'host_incomplete'),
    ('.py', 'import requests\ndef f():\n requests.get(LATE)\nLATE = "https://a.com/x"\n', 'full_url', None),
    ('.js', 'function f(){ fetch(LATE); } const LATE = "https://a.com/x";', 'full_url', None),
    ('.py', 'import requests\ndef f():\n if condition:\n  url="https://a.com/x"\n  requests.get(url)\n', 'full_url', None),
    ('.js', 'function f(){ if(condition){ const url="https://a.com/x"; fetch(url); } }', 'full_url', None),
    ('.py', 'import requests\ndef f():\n url="https://a.com/x"\n def nested():\n  url="https://other.com"\n requests.get(url)\n', 'multiple_assignments', None),
    ('.js', 'function f(){ const url="https://a.com/x"; function nested(){const url="https://other.com";} fetch(url); }', 'multiple_assignments', None),
    ('.js', 'function f(port){ fetch("http://localhost:" + port); }', 'parameter', None),
    ('.py', 'import requests\ndef f(port):\n requests.get("http://localhost:" + port)\n', 'parameter', None),
    ('.js', 'function f(){ let port="80"; port="81"; fetch("http://localhost:" + port); }', 'multiple_assignments', None),
    ('.py', 'import requests\ndef f():\n port="80"\n port="81"\n requests.get("http://localhost:" + port)\n', 'multiple_assignments', None),
    ('.py', 'import requests\ndef f():\n env="com"\n requests.get("https://a." + env + "/x")\n', 'full_url', None),
    ('.js', 'function f(){ const env="com"; fetch("https://a." + env + "/x"); }', 'full_url', None),
])
def test_additional_bindings(tmp_path, suffix, source, outcome, detail):
    (tmp_path / ('a' + suffix)).write_text(source, encoding='utf-8')
    records, edges, literal = pipeline(tmp_path)
    before = deepcopy(edges)
    report = run_constprop_check(edges, records, tmp_path, literal)
    accounting(report)
    pending = list(literal['pending_constprop'])
    assert len(pending) == 1
    index = pending[0]
    if outcome in ('full_url', 'partial_url'):
        assert report['outcomes'][outcome] == 1
        assert edges[index].host == 'a.com'
        assert edges[index].needs_review is (outcome == 'partial_url')
        assert edges[index].review_reason == ('partial_url' if outcome == 'partial_url' else None)
    else:
        assert report['still_pending'][index] == outcome
        assert edges[index] == before[index]
    if detail:
        assert any(r['cause'] == outcome and r['detail'] == detail for r in report['reasons'])
    preserve(before, edges)
