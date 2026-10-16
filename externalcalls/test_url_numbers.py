"""Decimal integer contexts and incomplete port boundaries, using real grammars."""

import pytest

from externalcalls.test_url_eval import expression
from externalcalls.test_constprop_check import pipeline
from externalcalls.constprop_check import run_constprop_check
from externalcalls.url_eval import evaluate, host_from_text


@pytest.mark.parametrize("text,host,reason", [
    ("http://localhost:", "localhost", "port_boundary_host"),
    ("https://127.0.0.1:", "127.0.0.1", "port_boundary_host"),
    ("http://urluser:urlpass@127.0.0.1:", "127.0.0.1", "port_boundary_host"),
    ("http://[::1]:", None, "unsupported_authority"),
    ("https://a.", None, "host_incomplete"),
    ("http://", None, "host_incomplete"),
    ("http://one@two@LOCALHOST:", "localhost", "port_boundary_host"),
])
def test_port_boundary(text, host, reason):
    assert host_from_text(text, False) == (host, reason)


@pytest.mark.parametrize("language,text,expected,complete,kind", [
    ("python", '"http://localhost:" + 3000', "http://localhost:", False, "other_expression"),
    ("python", 'f"http://a.com:{PORT}"', "http://a.com:8080", True, None),
    ("python", '"http://localhost:" + str(PORT)', "http://localhost:8080", True, None),
    ("javascript", '"http://localhost:" + 3000', "http://localhost:3000", True, None),
    ("javascript", '`http://a.com:${PORT}`', "http://a.com:8080", True, None),
    ("javascript", '"http://localhost:" + PORT', "http://localhost:8080", True, None),
    ("python", 'f"http://a.com:{123}/x"', "http://a.com:123/x", True, None),
    ("python", '"http://a.com:" + str(123)', "http://a.com:123", True, None),
    ("typescript", '"http://a.com:" + 8080', "http://a.com:8080", True, None),
    ("typescript", '`http://a.com:${8080}/x`', "http://a.com:8080/x", True, None),
])
def test_numeric_context(language, text, expected, complete, kind):
    node, source = expression(text, language)
    result = evaluate(node, source, language, lambda name: 8080 if name == "PORT" else None)
    assert result["text"] == expected
    assert result["complete"] is complete
    assert (result["blocked_by"] or {}).get("kind") == kind


@pytest.mark.parametrize("language", ["python", "javascript", "typescript"])
@pytest.mark.parametrize("form", ["1.5", "0xff", "0o10", "0b10", "-1", "1_000", "1 + 2", "BOOL"])
def test_unsupported_numbers(language, form):
    form = form.replace("BOOL", "True" if language == "python" else "true")
    text = 'f"{'+form+'}"' if language == "python" else '`${'+form+'}`'
    node, source = expression(text, language)
    result = evaluate(node, source, language)
    assert result["text"] == ""
    assert result["complete"] is False
    assert result["blocked_by"]["kind"] == "other_expression"


@pytest.mark.parametrize("language,text,kind", [
    ("javascript", 'String(123)', "call_expression"),
    ("javascript", '(123).toString()', "call_expression"),
    ("javascript", '1 + 2', "other_expression"),
    ("javascript", '"a" + (1 + 2)', "other_expression"),
    ("python", 'str(123, 10)', "call_expression"),
    ("python", 'str(object=123)', "call_expression"),
    ("python", 'f"{PORT:04}"', "other_expression"),
    ("python", 'f"{PORT!r}"', "other_expression"),
    ("python", '"a" + PORT', "other_expression"),
])
def test_unsupported_contexts(language, text, kind):
    node, source = expression(text, language)
    result = evaluate(node, source, language, lambda name: 8080)
    assert result["complete"] is False
    assert result["blocked_by"]["kind"] == kind


@pytest.mark.parametrize("filename,text", [
    ("client.py", 'import requests\nPORT = 8080\nrequests.get(f"http://a.com:{PORT}/x")\n'),
    ("client.js", 'const PORT = 8080;\nfetch(`http://a.com:${PORT}/x`);\n'),
    ("client.ts", 'const PORT = 8080;\nfetch(`http://a.com:${PORT}/x`);\n'),
    ("client.py", 'import requests\nPORT = 8080\nrequests.get("http://a.com:" + str(PORT) + "/x")\n'),
    ("client.js", 'const PORT = 8080;\nfetch("http://a.com:" + PORT + "/x");\n'),
])
def test_numeric_constprop(tmp_path, filename, text):
    (tmp_path / filename).write_text(text, encoding="utf-8")
    records, edges, literal = pipeline(tmp_path)
    candidates = [(i, e) for i, e in enumerate(edges) if e.status == "external_candidate"]
    assert len(candidates) == 1
    index, edge = candidates[0]
    assert (edge.literal, edge.host, edge.needs_review, edge.review_reason) == (
        "http://a.com:", "a.com", True, "port_boundary_host")
    assert literal["pending_constprop"][index]["name"] == "PORT"
    report = run_constprop_check(edges, records, tmp_path, literal)
    assert not report["aborted"]
    assert report["outcomes"]["full_url"] == 1
    assert (edge.literal, edge.host, edge.needs_review, edge.review_reason, edge.constprop_via) == (
        "http://a.com:8080/x", "a.com", False, None, "ast")


def test_redacted_boundary(tmp_path):
    (tmp_path / "client.js").write_text('fetch("http://user:pass@127.0.0.1:" + port);\n')
    records, edges, literal = pipeline(tmp_path)
    edge = next(e for e in edges if e.status == "external_candidate")
    assert (edge.literal, edge.host, edge.needs_review, edge.review_reason) == (
        "http://127.0.0.1:", "127.0.0.1", True, "port_boundary_host")
    assert literal["userinfo_redacted"] == 1
    report = run_constprop_check(edges, records, tmp_path, literal)
    assert report["outcomes"]["not_found"] == 1
    assert (edge.literal, edge.host, edge.review_reason) == (
        "http://127.0.0.1:", "127.0.0.1", "port_boundary_host")


@pytest.mark.parametrize("language", ["python", "javascript", "typescript"])
@pytest.mark.parametrize("form", ["1.5", "0xff", "0o10", "0b10", "-1", "1_000", "1 + 2", "BOOL"])
def test_unsupported_constant(tmp_path, language, form):
    form = form.replace("BOOL", "True" if language == "python" else "true")
    if language == "python":
        text = 'import requests\nPORT = ' + form + '\nrequests.get(f"http://a.com:{PORT}/x")\n'
        filename = "client.py"
    else:
        text = 'const PORT = ' + form + ';\nfetch(`http://a.com:${PORT}/x`);\n'
        filename = "client.js" if language == "javascript" else "client.ts"
    (tmp_path / filename).write_text(text)
    records, edges, literal = pipeline(tmp_path)
    edge = next(e for e in edges if e.status == "external_candidate")
    report = run_constprop_check(edges, records, tmp_path, literal)
    assert not report["aborted"]
    assert report["outcomes"]["value_not_static"] == 1
    assert not report["resolved_edges"]
    assert (edge.literal, edge.host, edge.needs_review, edge.review_reason, edge.constprop_via) == (
        "http://a.com:", "a.com", True, "port_boundary_host", None)
