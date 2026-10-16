"""URL evaluation and AST access against real, in-memory grammar trees."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analyzer_new.parser import _get_ts_parser
from externalcalls.ast_access import (find_call_node, find_node_at, get_url_argument,
                                      load_tree, read_source)
from externalcalls.url_eval import evaluate, host_from_text


def expression(text, language):
    source = text.encode()
    tree = _get_ts_parser(language).parse(source)
    assert not tree.root_node.has_error
    node = tree.root_node.named_children[0]
    if node.type == "expression_statement":
        node = node.named_children[0]
    return node, source


@pytest.mark.parametrize("language", ["python", "javascript", "typescript", "tsx"])
@pytest.mark.parametrize("text,expected,kind", [
    ('"abc"', "abc", None), ("'abc'", "abc", None), ('""', "", None),
    (r'"a\nb"', "a", "escape_in_string"),
    ('"https://a.com/" + x + "ignored"', "https://a.com/", "identifier"),
    ('x + "/x"', "", "identifier"), ('"a" + "b"', "ab", None),
    ('config.url', "", "member_access"), ('get_url()', "", "call_expression"),
    ('1', "", "other_expression"), ('"a" * 2', "", "other_expression"),
])
def test_common(language, text, expected, kind):
    node, source = expression(text, language)
    result = evaluate(node, source, language)
    assert result["text"] == expected
    assert result["complete"] is (kind is None)
    if kind:
        assert result["blocked_by"]["kind"] == kind
        assert result["blocked_by"]["line"] == 1
        assert result["blocked_by"]["col"] >= 0
    else:
        assert result["blocked_by"] is None


@pytest.mark.parametrize("text,expected,kind", [
    (r'r"a\b"', "a", "escape_in_string"), ('b"abc"', "", "bytes_literal"),
    ('B"abc"', "", "bytes_literal"), ('"a" "b"', "ab", None),
    ('f"a" "b"', "ab", None), ('f"https://a.com/{x}"', "https://a.com/", "identifier"),
    ('f"{x!r}"', "", "other_expression"), ('f"{x:>10}"', "", "other_expression"),
    ('f"{{lit}}"', "{lit}", None), ('f"{x.y}"', "", "other_expression"),
    ('f"{{{{lit}}}}"', "{{lit}}", None),
    (r'f"{{lit}}\n"', "{lit}", "escape_in_string"),
    ('f"a{x}b"', "a", "identifier"),
])
def test_python(text, expected, kind):
    node, source = expression(text, "python")
    result = evaluate(node, source, "python")
    assert result["text"] == expected
    assert result["complete"] is (kind is None)
    assert (result["blocked_by"] or {}).get("kind") == kind


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
@pytest.mark.parametrize("text,expected,kind", [
    ('`abc`', 'abc', None), ('`a${x}b`', 'a', 'identifier'),
    (r'`a\nb`', 'a', 'escape_in_string'), ('`${x}`', '', 'identifier'),
    ('`${x.y}`', '', 'other_expression'),
])
def test_templates(language, text, expected, kind):
    node, source = expression(text, language)
    result = evaluate(node, source, language)
    assert result["text"] == expected
    assert result["complete"] is (kind is None)
    assert (result["blocked_by"] or {}).get("kind") == kind


@pytest.mark.parametrize("language,text", [
    ("python", 'os.environ["X"]'), ("python", 'os.environ.get("X")'),
    ("python", 'os.getenv("X")'), ("python", 'os.getenv(x)'),
    ("javascript", 'process.env.X'), ("javascript", 'process.env["X"]'),
    ("typescript", 'process.env[x]'), ("tsx", 'process.env["X"]'),
])
def test_env(language, text):
    node, source = expression(text, language)
    result = evaluate(node, source, language)
    assert result["text"] == ""
    assert result["blocked_by"]["kind"] == "env_var"
    assert result["blocked_by"]["name"] == (None if text.endswith('(x)') or text.endswith('[x]') else "X")


@pytest.mark.parametrize("language,text", [
    ("python", 'f"{BASE}/x"'), ("javascript", '`${BASE}/x`'),
    ("typescript", 'BASE + "/x"'), ("tsx", 'BASE + "/x"'),
])
def test_resolve_continues(language, text):
    node, source = expression(text, language)
    seen = []
    def resolve(name):
        seen.append(name)
        return "https://a.com"
    assert evaluate(node, source, language, resolve) == {
        "text": "https://a.com/x", "complete": True, "blocked_by": None}
    assert seen == ["BASE"]
    blocked = evaluate(node, source, language, lambda name: None)
    assert blocked["blocked_by"]["kind"] == "identifier_unresolved"
    assert blocked["blocked_by"]["name"] == "BASE"


@pytest.mark.parametrize("text,complete,host,reason", [
    ('https://A.COM/x', True, 'a.com', None), ('http://a.com', True, 'a.com', None),
    ('ws://a.com?x', False, 'a.com', None), ('wss://a.com#x', False, 'a.com', None),
    ('https://user:pw@A.com:8080/x', True, 'a.com', None),
    ('https://a.com/', False, 'a.com', None), ('https://a.', False, None, 'host_incomplete'),
    ('https://a.com', False, None, 'host_incomplete'), ('/users', True, None, 'relative_url'),
    ('ftp://a.com/', True, None, 'relative_url'), ('HTTPS://a.com/', True, None, 'relative_url'),
    ('https://[::1]/', True, None, 'unsupported_authority'),
    ('https://a.com:abc/', True, None, 'unsupported_authority'),
    ('https://a.com:/', True, None, 'unsupported_authority'),
    ('https://a..com/', True, None, 'unsupported_authority'),
    ('https://-a.com/', True, None, 'unsupported_authority'),
    ('https://a com/', True, None, 'unsupported_authority'),
    ('https://a@b@c/', True, None, 'unsupported_authority'),
    ('https:///', True, None, 'unsupported_authority'),
])
def test_hosts(text, complete, host, reason):
    assert host_from_text(text, complete) == (host, reason)


@pytest.mark.parametrize("language", ["typescript", "tsx"])
@pytest.mark.parametrize("wrapper", ['("https://a.com/" as string)',
                                     '("https://a.com/" satisfies string)', '"https://a.com/"!'])
def test_wrappers(language, wrapper):
    node, source = expression(wrapper, language)
    assert evaluate(node, source, language)["text"] == "https://a.com/"


@pytest.mark.parametrize("language,text,index,keyword,prop,expected,reason", [
    ('python', 'requests.request("GET", "url")', 1, 'url', None, '"url"', None),
    ('python', 'requests.request("GET", url="url")', 1, 'url', None, '"url"', None),
    ('python', 'f(*args)', 0, 'url', None, None, 'spread_before_url'),
    ('python', 'f(**args)', 0, 'url', None, None, 'spread_before_url'),
    ('python', 'f()', 0, 'url', None, None, 'no_url_argument'),
    ('javascript', 'axios({url: "url"})', 0, None, 'url', '"url"', None),
    ('javascript', 'axios({url})', 0, None, 'url', 'url', None),
    ('javascript', 'axios("url")', 0, None, 'url', '"url"', None),
    ('javascript', 'fetch(...args)', 0, None, None, None, 'spread_before_url'),
    ('typescript', 'fetch(("url" as string))', 0, None, None, '"url"', None),
])
def test_arguments(language, text, index, keyword, prop, expected, reason):
    call, source = expression(text, language)
    entry = SimpleNamespace(url_arg_index=index, url_arg_keyword=keyword, url_arg_property=prop)
    node, actual = get_url_argument(call, entry, language)
    assert actual == reason
    assert (source[node.start_byte:node.end_byte].decode() if node else None) == expected


def test_source_cache_and_positions(tmp_path):
    path = tmp_path / 'a.py'
    path.write_bytes('"héllo 😀"; requests.get("a"); requests.get("b")\r\n'.encode())
    text, source = read_source(tmp_path, 'a.py')
    assert '\r' not in text
    cache = {}
    tree, actual_source, error = load_tree(tmp_path, 'a.py', cache)
    assert actual_source == source and error is None
    with patch('externalcalls.ast_access._get_ts_parser', side_effect=AssertionError):
        assert load_tree(tmp_path, 'a.py', cache)[0] is tree
    for url in ('a', 'b'):
        offset = source.index(('get("' + url).encode())
        call = find_call_node(tree, 1, offset)
        assert call.type == 'call'
        argument = call.child_by_field_name('arguments').named_children[0]
        assert find_node_at(tree, 1, argument.start_point[1], 'string') == argument
    assert find_call_node(tree, 99, 0) == 'node_not_found'
    assert find_node_at(tree, 99, 0, 'string') is None
    assert load_tree(tmp_path, 'missing.py', cache)[2] == 'file_unreadable'
    path.with_suffix('.unknown').write_text('x')
    assert load_tree(tmp_path, 'a.unknown', cache)[2] == 'grammar_unavailable'


def test_identifier_position_and_stop_order():
    node, source = expression('"é" + missing + later', 'python')
    seen = []
    def resolve(name):
        seen.append(name)
        return None
    result = evaluate(node, source, 'python', resolve)
    assert result['text'] == 'é'
    assert seen == ['missing']
    assert result['blocked_by'] == {'kind': 'identifier_unresolved', 'name': 'missing',
                                    'line': 1, 'col': len('"é" + '.encode())}
    assert evaluate(*expression('x + "tail"', 'python'), 'python', lambda name: '') == {
        'text': 'tail', 'complete': True, 'blocked_by': None}


def test_awaited_generic_and_ambiguous():
    source = b'await axios.get<User>("x")'
    tree = _get_ts_parser('typescript').parse(source)
    call = find_call_node(tree, 1, source.index(b'get'))
    assert call.type == 'call_expression'
    fake_root = SimpleNamespace(type='program', named_children=[call, call])
    assert find_call_node(SimpleNamespace(root_node=fake_root), 1, source.index(b'get')) == 'ambiguous_node'


def test_parser_failure_cached(tmp_path):
    (tmp_path / 'a.py').write_text('x')
    cache = {}
    with patch('externalcalls.ast_access._get_ts_parser', side_effect=RuntimeError):
        assert load_tree(tmp_path, 'a.py', cache)[2] == 'grammar_unavailable'
    assert load_tree(tmp_path, 'a.py', cache)[2] == 'grammar_unavailable'


@pytest.mark.parametrize('text,expected', [
    ('https://user:pw@a.com:8080/x', 'https://a.com:8080/x'),
    ('http://u@v@host/x', 'http://host/x'),
    ('ws://u@host?x', 'ws://host?x'), ('wss://u@host#x', 'wss://host#x'),
    ('https://host/x', 'https://host/x'), ('https://host/x@y', 'https://host/x@y'),
    ('https://host?x@y', 'https://host?x@y'), ('https://host#x@y', 'https://host#x@y'),
    ('https://u@', 'https://'), ('', ''), ('u@host/x', 'u@host/x'),
])
def test_redact_userinfo(text, expected):
    from externalcalls.url_eval import redact_userinfo
    assert redact_userinfo(text) == expected
