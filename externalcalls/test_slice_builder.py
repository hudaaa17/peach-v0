from pathlib import Path

from analyzer_new.parser import parse_repo
from analyzer_new.resolver import CallEdge, resolve_calls
from externalcalls.slice_builder import (
    JoernTask,
    Slice,
    SliceBudget,
    build_slice,
    group_tasks,
    materialize_slice,
)


def _write(root, relative, text="pass\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _records(root):
    return {record.path: record for record in parse_repo(root)}


def _task(path, reason="imported_constant", variable="URL", language="python", edge_index=0):
    return JoernTask("task", edge_index, path, 1, "requests.get", 0,
                     reason, variable, language)


def test_import_depth_two_is_incomplete_and_deterministic(tmp_path):
    root = tmp_path
    _write(root, "pkg/sink.py", "from .middle import URL\nrequests.get(URL)\n")
    _write(root, "pkg/middle.py", "from .third import URL\n")
    _write(root, "pkg/third.py", "from .fourth import URL\n")
    _write(root, "pkg/fourth.py", "URL = 'https://fourth.invalid'\n")
    records = _records(root)
    result = build_slice(_task("pkg/sink.py"), records, [], root,
                         SliceBudget(max_import_distance=2))
    print(f"distance-2 slice: {result!r}")
    assert result.files == ["pkg/sink.py", "pkg/middle.py", "pkg/third.py"]
    assert result.incomplete is True


def test_import_priority_max_files_max_lines_and_unresolved(tmp_path):
    root = tmp_path
    _write(root, "sink.py", "from .b import URL\nfrom .a import URL\nrequests.get(URL)\n")
    _write(root, "a.py", "URL = 'https://a.invalid'\n")
    _write(root, "b.py", "URL = 'https://b.invalid'\n")
    records = _records(root)
    by_files = build_slice(_task("sink.py"), records, [], root,
                           SliceBudget(max_files=2))
    assert by_files.files == ["sink.py", "a.py"]
    assert ("b.py", "budget_files") in by_files.omitted
    assert by_files.incomplete is True

    _write(root, "alias_sink.py", "from .z import OTHER as U\nfrom .a import URL\nrequests.get(U)\n")
    _write(root, "z.py", "OTHER = 'https://alias.invalid'\n")
    _write(root, "a.py", "URL = 'https://other.invalid'\n")
    alias_records = _records(root)
    alias_slice = build_slice(_task("alias_sink.py", variable="U"), alias_records, [], root,
                              SliceBudget(max_files=2))
    assert alias_slice.files == ["alias_sink.py", "z.py"]

    by_lines = build_slice(_task("sink.py"), records, [], root,
                           SliceBudget(max_lines=4))
    assert by_lines.files == ["sink.py"]
    assert ("a.py", "budget_lines") in by_lines.omitted
    assert ("b.py", "budget_lines") in by_lines.omitted

    _write(root, "broken.py", "from .absent import URL\nrequests.get(URL)\n")
    broken_records = _records(root)
    unresolved = build_slice(_task("broken.py"), broken_records, [], root)
    assert unresolved.incomplete is True
    assert ("absent.URL", "unresolved_import") in unresolved.omitted

    _write(root, "alias.ts", 'import URL from "@/constants";\nfetch(URL);\n')
    alias_records = _records(root)
    alias_result = build_slice(_task("alias.ts", "imported_constant", "URL", "typescript"),
                               alias_records, [], root)
    assert alias_result.incomplete is True
    assert ("@/constants", "alias_import_unresolved") in alias_result.omitted

    _write(root, "external.ts", 'import axios from "axios";\naxios.get(URL);\n')
    external_records = _records(root)
    external_result = build_slice(_task("external.ts", "imported_constant", "URL", "typescript"),
                                  external_records, [], root)
    assert external_result.incomplete is False
    assert external_result.files == ["external.ts"]


def test_sink_file_failures_and_non_joern_reasons(tmp_path):
    root = tmp_path
    _write(root, "large.py", "one\ntwo\n")
    records = _records(root)
    too_large = build_slice(_task("large.py"), records, [], root, SliceBudget(max_lines=1))
    assert too_large.files == ["large.py"]
    assert too_large.incomplete is True
    assert too_large.omitted == [("large.py", "sink_file_too_large")]

    missing = build_slice(_task("missing.py"), records, [], root)
    assert missing.incomplete is True
    assert missing.omitted == [("missing.py", "sink_file_unreadable")]

    for reason in ("env_var", "value_not_static", "unknown_reason"):
        result = build_slice(_task("large.py", reason), records, [], root)
        assert result.files == ["large.py"]
        assert result.incomplete is True
        assert result.omitted == [("large.py", "not_joern_task")]


def test_each_supported_reason_branch(tmp_path):
    root = tmp_path
    _write(root, "sink.py", "from .dep import URL\nrequests.get(URL)\n")
    _write(root, "dep.py", "URL = 'https://dep.invalid'\n")
    records = _records(root)

    for reason in ("imported_constant", "not_found"):
        result = build_slice(_task("sink.py", reason), records, [], root)
        assert result.files == ["sink.py", "dep.py"]
    for reason in ("member_access", "call_expression", "complex_binding"):
        result = build_slice(_task("sink.py", reason), records, [], root)
        assert result.files == ["sink.py", "dep.py"]
    multiple = build_slice(_task("sink.py", "multiple_assignments"), records, [], root)
    assert multiple.files == ["sink.py"]
    assert multiple.incomplete is False


def test_parameter_callers_choose_nearest_five_and_report_limit(tmp_path):
    root = tmp_path
    sink = "pkg/target.py"
    _write(root, sink, "def sink(url):\n    requests.get(url)\n")
    caller_paths = [
        "other/z.py", "pkg/c.py", "pkg/a.py", "pkg/deep/d.py",
        "pkg/b.py", "outside/e.py", "other/f.py", "pkg/deep/g.py",
    ]
    for path in caller_paths:
        _write(root, path)
    records = _records(root)
    target = "pkg/target.py::sink"
    edges = [CallEdge(target, "requests.get", sink, 2, "url")]
    edges.extend(CallEdge(f"{path}::caller", "sink", path, 1, "url",
                          resolved_targets=[target]) for path in caller_paths)
    task = _task(sink, "parameter", edge_index=0)
    result = build_slice(task, records, edges, root, SliceBudget(max_callers=5))
    print(f"nearest parameter callers: {result.files!r}; omitted={result.omitted!r}")
    assert result.files == [sink, "pkg/a.py", "pkg/b.py", "pkg/c.py",
                            "pkg/deep/d.py", "pkg/deep/g.py"]
    assert result.incomplete is True
    assert set(result.omitted) == {
        ("other/f.py", "caller_limit"), ("other/z.py", "caller_limit"),
        ("outside/e.py", "caller_limit"),
    }


def test_real_parse_and_resolve_calls_drive_parameter_slice(tmp_path):
    root = tmp_path
    path = _write(root, "app.py", "def sink(url):\n    requests.get(url)\n\ndef caller():\n    sink('https://x.invalid')\n")
    records_list = parse_repo(root)
    edges = resolve_calls(records_list)
    target_edge = next(edge for edge in edges if edge.callee_expr == "requests.get")
    caller_edge = next(edge for edge in edges if edge.callee_expr == "sink")
    print(f"real parameter edge: caller={caller_edge.caller!r}, "
          f"resolved_targets={caller_edge.resolved_targets!r}")
    assert caller_edge.resolved_targets == ["app.py::sink"]
    assert caller_edge.caller == "app.py::caller"
    records = {record.path: record for record in records_list}
    task = _task("app.py", "parameter", edge_index=edges.index(target_edge))
    result = build_slice(task, records, edges, root)
    assert result.files == ["app.py"]


def _slice(files, counts, incomplete=False, omitted=None):
    value = Slice(files, sum(counts.values()), incomplete, omitted or [])
    setattr(value, "_file_lines", counts)
    return value


def test_grouping_overlap_no_overlap_limits_language_and_exclusions():
    budget = SliceBudget(max_files=4, max_lines=9)
    first = (_task("a.py"), _slice(["a.py", "shared.py"], {"a.py": 2, "shared.py": 3}))
    second = (_task("b.py"), _slice(["shared.py", "b.py"], {"shared.py": 3, "b.py": 4}))
    merged = group_tasks([second, first], budget)
    assert merged == [("python", ["a.py", "shared.py", "b.py"], ["task", "task"])]

    separate = group_tasks([
        (_task("a.py"), _slice(["a.py"], {"a.py": 2})),
        (_task("z.py"), _slice(["z.py"], {"z.py": 2})),
    ], budget)
    assert len(separate) == 2

    over_budget = group_tasks([
        (_task("a.py"), _slice(["a.py", "shared.py"], {"a.py": 3, "shared.py": 3})),
        (_task("b.py"), _slice(["shared.py", "b.py"], {"shared.py": 3, "b.py": 3})),
    ], SliceBudget(max_lines=8))
    assert len(over_budget) == 2

    over_file_budget = group_tasks([
        (_task("a.py"), _slice(["a.py", "shared.py"], {"a.py": 1, "shared.py": 1})),
        (_task("b.py"), _slice(["shared.py", "b.py"], {"shared.py": 1, "b.py": 1})),
    ], SliceBudget(max_files=2))
    assert len(over_file_budget) == 2

    different_languages = group_tasks([
        (_task("a.py"), _slice(["shared.py", "a.py"], {"shared.py": 1, "a.py": 1})),
        (_task("b.ts", language="typescript"),
         _slice(["shared.py", "b.ts"], {"shared.py": 1, "b.ts": 1})),
    ], budget)
    assert len(different_languages) == 2

    excluded = group_tasks([
        (_task("a.py"), _slice(["a.py"], {"a.py": 1}, omitted=[("a.py", "not_joern_task")])),
        (_task("b.py"), _slice(["a.py", "b.py"], {"a.py": 1, "b.py": 1})),
    ], budget)
    assert len(excluded) == 2


def test_materialize_preserves_crlf_and_refuses_unsafe_paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "pkg" / "client.py"
    source.parent.mkdir()
    source.write_bytes(b"first\r\nsecond\r\n")
    destination = tmp_path / "slice"
    copied, error = materialize_slice(["pkg/client.py"], root, destination)
    print(f"materialized bytes: {(copied / 'pkg/client.py').read_bytes()!r}")
    assert error is None
    assert (copied / "pkg/client.py").read_bytes() == b"first\r\nsecond\r\n"
    assert materialize_slice(["pkg/client.py"], root, root / "inside") == (
        None, "destination_inside_repo")
    assert materialize_slice(["missing.py"], root, tmp_path / "missing-dest") == (
        None, "file_missing")
    assert materialize_slice(["../outside.py"], root, tmp_path / "escape-dest") == (
        None, "path_escapes_repo")
