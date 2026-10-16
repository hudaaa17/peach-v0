import json
import logging
from dataclasses import asdict

from analyzer_new.parser import LOG, parse_repo


SOURCE_FILES = {
    "main.py": "def f():\n    return 1\n",
    "main.js": "function f() { return 1; }\n",
    "main.jsx": "const f = () => <div />;\n",
    "main.ts": "function f(): number { return 1; }\n",
    "main.tsx": "const f = () => <div />;\n",
    "module.mjs": "export function f() { return 1; }\n",
    "module.cjs": "module.exports = function f() { return 1; };\n",
}


def _write_fixture(root, sources):
    configs = {
        "package.json": '{}\n',
        "settings.yaml": "enabled: true\n",
        ".env": "PORT=8080\n",
        "Dockerfile": "FROM scratch\n",
    }
    others = {
        "README.md": "Read me.\n",
        "notes.txt": "Notes.\n",
        "style.css": "body {}\n",
        "Main.java": "class Main {}\n",
    }
    for name, content in {**sources, **configs, **others}.items():
        (root / name).write_text(content, encoding="utf-8")
    return configs, others


def _full_result(records, gaps, caplog):
    return json.dumps({
        "records": [asdict(record) for record in records],
        "gaps": [asdict(gap) for gap in gaps],
        "logs": [record.getMessage() for record in caplog.records],
    }, indent=2)


def test_source_cap_keeps_non_source_files_and_counts_unexamined_sources(tmp_path, caplog):
    sources = SOURCE_FILES
    max_files = len(sources) - 2
    configs, others = _write_fixture(tmp_path, sources)
    gaps = []
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        records = parse_repo(tmp_path, max_files=max_files, gaps=gaps)
    result = _full_result(records, gaps, caplog)

    parsed_sources = {r.path for r in records if r.parse_mode == "treesitter" and not r.skipped}
    assert len(parsed_sources) == max_files, result
    assert {r.path for r in records if r.path in sources} == parsed_sources, result
    assert {r.path for r in records if r.is_config} == set(configs), result
    assert {r.path: r.source_lines for r in records if not r.parse_mode and not r.skipped} == {
        name: content.splitlines() for name, content in {**configs, **others}.items()
    }, result

    not_examined = len(sources) - max_files
    cap_gaps = [gap for gap in gaps if gap.kind == "file_cap"]
    assert len(cap_gaps) == 1, result
    assert cap_gaps[0].scope == "repo", result
    assert cap_gaps[0].count == not_examined, result
    assert set(cap_gaps[0].samples) == set(sources) - parsed_sources, result
    cap_warnings = [record.args[1] for record in caplog.records
                    if record.getMessage().startswith("file cap reached; repo truncated ")]
    assert cap_warnings == [{
        "cap": max_files,
        "not_examined": not_examined,
        "note": "raise max_files if the graph looks incomplete",
    }], result


def test_non_source_files_do_not_reach_source_cap(tmp_path, caplog):
    sources = {name: SOURCE_FILES[name] for name in ("main.py", "main.js")}
    max_files = len(sources) + 1
    configs, others = _write_fixture(tmp_path, sources)
    gaps = []
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        records = parse_repo(tmp_path, max_files=max_files, gaps=gaps)
    result = _full_result(records, gaps, caplog)

    parsed_sources = {r.path for r in records if r.parse_mode == "treesitter" and not r.skipped}
    assert parsed_sources == set(sources), result
    assert {r.path for r in records} == set(sources) | set(configs) | set(others), result
    assert {r.path for r in records if r.is_config} == set(configs), result
    assert {r.path: r.source_lines for r in records if not r.parse_mode and not r.skipped} == {
        name: content.splitlines() for name, content in {**configs, **others}.items()
    }, result
    assert [gap for gap in gaps if gap.kind == "file_cap"] == [], result
