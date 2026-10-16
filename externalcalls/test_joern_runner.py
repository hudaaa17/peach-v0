"""Real Joern integration tests; skipped only when the Joern CLI is absent."""

from pathlib import Path
import os

import pytest

from externalcalls.joern_runner import run_joern_tasks
from externalcalls.slice_builder import JoernTask, Slice


JOERN = Path("/opt/joern/joern-cli/joern")
pytestmark = pytest.mark.skipif(
    not JOERN.exists(), reason="real Joern 4.0.653 CLI is not installed")


def _write_slice(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
    return root


def _task(task_id, file, line_number, line, callee, language="python", *,
          reason="imported_constant", variable="URL", url_arg_index=0,
          url_arg_keyword=None):
    leaf = callee.rsplit(".", 1)[-1]
    callee_col = len(line[:line.index(leaf + "(")].encode("utf-8"))
    return JoernTask(
        task_id=task_id,
        edge_index=0,
        file=file,
        line=line_number,
        callee_expr=callee,
        callee_col=callee_col,
        reason=reason,
        variable=variable,
        language=language,
        url_arg_index=url_arg_index,
        url_arg_keyword=url_arg_keyword,
    )


def _make_task(task_id, file, source, callee, language="python", **kwargs):
    lines = source.splitlines()
    sink_line = next(i for i, line in enumerate(lines, 1)
                     if callee.rsplit(".", 1)[-1] + "(" in line)
    return _task(task_id, file, sink_line, lines[sink_line - 1], callee,
                 language, **kwargs)


def _metadata(task_id, files, incomplete=False):
    return {task_id: Slice(files=list(files), total_lines=0, incomplete=incomplete)}


def _run(root, tasks, *, slices=None, timeout_seconds=300):
    return run_joern_tasks(
        tasks,
        root,
        task_slices=slices,
        timeout_seconds=timeout_seconds,
    )


def _source_codes(result):
    return {source["code"] for source in result["sources"]}


def _assert_result(result, status, expected_codes, reason_contains=None):
    details = (
        "Full Joern result on assertion failure:\n"
        f"status={result.get('status')!r}\n"
        f"reason={result.get('reason')!r}\n"
        f"sources={result.get('sources')!r}"
    )
    assert result.get("status") == status, details
    assert _source_codes(result) == set(expected_codes), details
    if reason_contains is not None:
        assert reason_contains.lower() in str(result.get("reason", "")).lower(), details


def test_imported_constant_across_two_files_is_a_literal_source(tmp_path):
    files = {
        "client.py": 'import requests\nfrom consts import URL\nrequests.get(URL)\n',
        "consts.py": 'URL = "https://imported.invalid"\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("imported", "client.py", files["client.py"], "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://imported.invalid"'})
    assert result["sources"][0]["file"].endswith("consts.py"), result


def test_sink_file_alone_is_unknown_without_a_literal_source(tmp_path):
    source = "import requests\nrequests.get(URL)\n"
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("sink-only", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "no_sources_found", set(), "UNKNOWN")


def test_relative_import_and_as_alias_preserves_literal_source(tmp_path):
    files = {
        "pkg/__init__.py": "",
        "pkg/client.py": 'import requests\nfrom .consts import URL as TARGET\nrequests.get(TARGET)\n',
        "pkg/consts.py": 'URL = "https://relative-alias.invalid"\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("relative-alias", "pkg/client.py", files["pkg/client.py"],
                      "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://relative-alias.invalid"'})


def test_parameter_with_two_callers_returns_both_literals(tmp_path):
    files = {
        "service.py": 'import requests\ndef send(url):\n    requests.get(url)\n',
        "first.py": 'import service\nservice.send("https://a.invalid")\n',
        "second.py": 'import service\nservice.send("https://b.invalid")\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("parameter-two-callers", "service.py", files["service.py"],
                      "requests.get", reason="parameter", variable="url")

    result, = _run(root, [task], slices=_metadata(task.task_id, files))

    _assert_result(result, "complete", {'"https://a.invalid"', '"https://b.invalid"'})


def test_two_key_dict_returns_both_literal_sources(tmp_path):
    source = ('import requests\n'
              'config = {"url": "https://dict-url.invalid", '
              '"timeout": "https://dict-timeout.invalid"}\n'
              'requests.get(config)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("dict-two-keys", "client.py", source, "requests.get",
                      reason="multiple_assignments", variable="config")

    result, = _run(root, [task])

    _assert_result(result, "complete", {
        '"https://dict-url.invalid"', '"https://dict-timeout.invalid"'})


def test_self_base_url_returns_literal_source(tmp_path):
    source = ('import requests\n'
              'class Client:\n'
              '    def __init__(self):\n'
              '        self.base_url = "https://self.invalid"\n'
              '    def get(self):\n'
              '        requests.get(self.base_url)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("self-base-url", "client.py", source, "requests.get",
                      variable="self.base_url")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://self.invalid"'})


def test_keyword_url_argument_is_selected_by_name(tmp_path):
    source = 'import requests\nURL = "https://keyword.invalid"\nrequests.get(url=URL)\n'
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("keyword-url", "client.py", source, "requests.get",
                      url_arg_index=None, url_arg_keyword="url")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://keyword.invalid"'})


def test_emoji_before_call_matches_converted_python_column(tmp_path):
    source = 'import requests\nx = "ðŸ˜€"; requests.get("https://emoji.invalid")\n'
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("emoji-column", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://emoji.invalid"'})


def test_two_sinks_in_different_files_are_queried_in_one_cpg(tmp_path):
    files = {
        "first.py": 'import requests\nrequests.get("https://first.invalid")\n',
        "second.py": 'import requests\nrequests.get("https://second.invalid")\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    tasks = [
        _make_task("first-sink", "first.py", files["first.py"], "requests.get"),
        _make_task("second-sink", "second.py", files["second.py"], "requests.get"),
    ]

    first, second = _run(root, tasks)

    _assert_result(first, "complete", {'"https://first.invalid"'})
    _assert_result(second, "complete", {'"https://second.invalid"'})


@pytest.mark.parametrize("language,extension", [
    ("javascript", "js"), ("typescript", "ts"),
])
def test_js_ts_parameter_returns_literal(language, extension, tmp_path):
    files = {
        f"service.{extension}": "export function send(url) { fetch(url); }\n",
        f"caller.{extension}": 'import { send } from "./service";\nsend("https://parameter.invalid");\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    service = files[f"service.{extension}"]
    task = _make_task("parameter", f"service.{extension}", service, "fetch",
                      language, reason="parameter", variable="url")

    result, = _run(root, [task], slices=_metadata(task.task_id, files))

    _assert_result(result, "complete", {'"https://parameter.invalid"'})


@pytest.mark.parametrize("language,extension", [
    ("javascript", "js"), ("typescript", "ts"),
])
def test_js_ts_parameter_same_file_control(language, extension, tmp_path):
    service = ('function send(url) { fetch(url); }\n'
               'send("https://same-file.invalid");\n')
    root = _write_slice(tmp_path / "slice", {f"service.{extension}": service})
    task = _make_task("same-file-parameter", f"service.{extension}", service,
                      "fetch", language, reason="parameter", variable="url")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://same-file.invalid"'})


@pytest.mark.parametrize("language,extension", [
    ("javascript", "js"), ("typescript", "ts"),
])
def test_imported_constant_known_limitation(language, extension, tmp_path):
    files = {
        f"client.{extension}": ('import axios from "axios";\n'
                                 'import { URL } from "./consts";\n'
                                 'axios.get(URL);\n'),
        f"consts.{extension}": ('export const URL = '
                                 '"https://known-limitation.invalid";\n'),
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("known-limitation", f"client.{extension}",
                      files[f"client.{extension}"], "axios.get", language)

    result, = _run(root, [task])

    _assert_result(result, "no_sources_found", set(), "UNKNOWN")


def test_python_if_else_returns_both_literal_sources(tmp_path):
    source = ('import requests\n'
              'if flag:\n    URL = "https://if.invalid"\n'
              'else:\n    URL = "https://else.invalid"\n'
              'requests.get(URL)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("python-if-else", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://if.invalid"', '"https://else.invalid"'})


def test_python_if_elif_else_returns_three_literal_sources(tmp_path):
    source = ('import requests\n'
              'if first:\n    URL = "https://if.invalid"\n'
              'elif second:\n    URL = "https://elif.invalid"\n'
              'else:\n    URL = "https://else.invalid"\n'
              'requests.get(URL)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("python-if-elif-else", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {
        '"https://if.invalid"', '"https://elif.invalid"', '"https://else.invalid"'})


def test_python_ternary_returns_both_literal_sources(tmp_path):
    source = ('import requests\n'
              'URL = "https://yes.invalid" if flag else "https://no.invalid"\n'
              'requests.get(URL)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("python-ternary", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://yes.invalid"', '"https://no.invalid"'})


def test_python_try_except_returns_both_literal_sources(tmp_path):
    source = ('import requests\n'
              'try:\n    URL = "https://try.invalid"\n'
              'except Exception:\n    URL = "https://except.invalid"\n'
              'requests.get(URL)\n')
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("python-try-except", "client.py", source, "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://try.invalid"', '"https://except.invalid"'})


@pytest.mark.parametrize("language,extension", [
    ("javascript", "js"), ("typescript", "ts"),
])
def test_js_ts_if_else_returns_both_literal_sources(language, extension, tmp_path):
    source = ('import axios from "axios";\n'
              'let URL;\n'
              'if (flag) { URL = "https://if.invalid"; }\n'
              'else { URL = "https://else.invalid"; }\n'
              'axios.get(URL);\n')
    root = _write_slice(tmp_path / "slice", {f"client.{extension}": source})
    task = _make_task("js-if-else", f"client.{extension}", source,
                      "axios.get", language)

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://if.invalid"', '"https://else.invalid"'})


@pytest.mark.parametrize("language,extension", [
    ("javascript", "js"), ("typescript", "ts"),
])
def test_js_ts_ternary_returns_both_literal_sources(language, extension, tmp_path):
    source = ('import axios from "axios";\n'
              'const URL = flag ? "https://yes.invalid" : "https://no.invalid";\n'
              'axios.get(URL);\n')
    root = _write_slice(tmp_path / "slice", {f"client.{extension}": source})
    task = _make_task("js-ternary", f"client.{extension}", source,
                      "axios.get", language)

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://yes.invalid"', '"https://no.invalid"'})


def test_requests_alias_import_in_another_file_is_a_literal_source(tmp_path):
    files = {
        "consts.py": 'URL = "https://alias-import.invalid"\n',
        "client.py": 'import requests as r\nfrom consts import URL\nr.get(URL)\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("requests-as-alias", "client.py", files["client.py"], "r.get")

    result, = _run(root, [task])

    _assert_result(result, "complete", {'"https://alias-import.invalid"'})


def _other_joern_cmdlines():
    matches = []
    proc_root = Path("/proc")
    for cmdline_path in proc_root.glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline_path.parent.name)
            if pid == os.getpid():
                continue
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
        except (OSError, ValueError):
            continue
        if "joern" in command.lower():
            matches.append(f"pid={pid} cmdline={command}")
    return matches


def test_tiny_timeout_fails_and_leaves_no_joern_process(tmp_path):
    source = 'import requests\nrequests.get("https://timeout.invalid")\n'
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = _make_task("timeout", "client.py", source, "requests.get")

    result, = _run(root, [task], timeout_seconds=0.001)

    _assert_result(result, "failed", set(), "timeout")
    matches = _other_joern_cmdlines()
    assert not matches, "Other Joern processes remain after timeout:\n" + "\n".join(matches)


def test_bare_field_access_is_rejected_as_sink_not_found(tmp_path):
    source = "import requests\nrequests.get\n"
    root = _write_slice(tmp_path / "slice", {"client.py": source})
    task = JoernTask(
        task_id="bare-field",
        edge_index=0,
        file="client.py",
        line=2,
        callee_expr="requests.get",
        callee_col=len("requests.".encode("utf-8")),
        reason="imported_constant",
        variable="URL",
        language="python",
        url_arg_index=0,
    )

    result, = _run(root, [task])

    _assert_result(result, "failed", set(), "sink_not_found")


def test_module_attribute_import_has_no_literal_source(tmp_path):
    files = {
        "client.py": ('import requests\nimport pkg.consts\n'
                      'requests.get(pkg.consts.URL)\n'),
        "pkg/__init__.py": "",
        "pkg/consts.py": 'URL = "https://module-attribute.invalid"\n',
    }
    root = _write_slice(tmp_path / "slice", files)
    task = _make_task("module-attribute", "client.py", files["client.py"],
                      "requests.get")

    result, = _run(root, [task])

    _assert_result(result, "no_sources_found", set(), "UNKNOWN")


def test_parameter_task_without_caller_is_incomplete(tmp_path):
    source = "import requests\ndef send(url):\n    requests.get(url)\n"
    root = _write_slice(tmp_path / "slice", {"service.py": source})
    task = _make_task("no-caller", "service.py", source, "requests.get",
                      reason="parameter", variable="url")

    result, = _run(root, [task], slices=_metadata(task.task_id, ["service.py"]))

    _assert_result(result, "incomplete", set(), "no caller")
