"""Cached call lookup preserves positions without repeating tree traversal."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analyzer_new.parser import _get_ts_parser
from externalcalls.ast_access import find_call_node, load_tree


@pytest.mark.parametrize('suffix,source,spelling', [
    ('.py', '"héllo 😀"; requests.get("a"); requests.get("b")\r\n', b'get'),
    ('.js', '"héllo 😀"; axios.get("a"); axios.get("b");', b'get'),
    ('.ts', 'await axios.get<User>("a"); await axios.get<User>("b");', b'get'),
    ('.tsx', 'await axios.get<User>("a"); await axios.get<User>("b");', b'get'),
])
def test_cached_lookup_never_rewalks(tmp_path, suffix, source, spelling):
    path = tmp_path / ('client' + suffix)
    path.write_bytes(source.encode('utf-8'))
    cache = {}
    tree, source_bytes, error = load_tree(tmp_path, path.name, cache)
    assert error is None
    positions = []
    offset = 0
    while True:
        offset = source_bytes.find(spelling, offset)
        if offset < 0:
            break
        positions.append(offset)
        offset += len(spelling)
    assert len(positions) == 2
    with patch('externalcalls.ast_access._walk', side_effect=AssertionError('repeated traversal')):
        calls = [find_call_node(tree, 1, col) for col in positions]
        assert all(call.type in ('call', 'call_expression') for call in calls)
        assert calls[0] != calls[1]
        assert [call.child_by_field_name('arguments').text for call in calls] == [b'("a")', b'("b")']
        assert find_call_node(tree, 99, 0) == 'node_not_found'
        assert load_tree(tmp_path, path.name, cache)[0] is tree
        assert [find_call_node(tree, 1, col) for col in positions] == calls


def test_cached_ambiguity(tmp_path):
    source = b'await axios.get<User>("x")'
    raw = _get_ts_parser('typescript').parse(source)
    call = raw.root_node.named_children[0].named_children[0]
    duplicate = SimpleNamespace(root_node=SimpleNamespace(type='program', named_children=[call, call]))
    (tmp_path / 'a.ts').write_bytes(source)
    parser = SimpleNamespace(parse=lambda source: duplicate)
    with patch('externalcalls.ast_access._get_ts_parser', return_value=parser):
        tree, _, error = load_tree(tmp_path, 'a.ts', {})
    assert error is None
    with patch('externalcalls.ast_access._walk', side_effect=AssertionError('repeated traversal')):
        assert find_call_node(tree, 1, source.index(b'get')) == 'ambiguous_node'
