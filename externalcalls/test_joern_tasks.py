from copy import deepcopy
from analyzer_new.parser import parse_repo
from analyzer_new.resolver import resolve_calls
from externalcalls.ast_grep_check import run_patterns
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls.config_check import run_config_check
from externalcalls.constprop_check import run_constprop_check
from externalcalls.literal_check import run_literal_check
from externalcalls.slice_builder import build_slice
from externalcalls.joern_tasks import build_joern_tasks


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pipeline(root):
    records = parse_repo(root)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for record in records:
        groups.setdefault(record.language, []).append(record.path)
    hits, runner_report = run_patterns(root, groups)
    joined = join_ast_grep_hits(edges, records, hits, runner_report)
    literal_report = run_literal_check(edges, records, root)
    constprop_report = run_constprop_check(edges, records, root, literal_report)
    config_report = run_config_check(edges, records, literal_report, constprop_report)
    assert not joined["aborted"]
    assert not literal_report["aborted"]
    assert not constprop_report["aborted"]
    return records, edges, literal_report, constprop_report, config_report


def _tasks(root):
    records, edges, literal_report, constprop_report, config_report = _pipeline(root)
    old_edges = deepcopy(edges)
    old_literal = deepcopy(literal_report)
    old_constprop = deepcopy(constprop_report)
    old_config = deepcopy(config_report)
    result = build_joern_tasks(edges, records, literal_report, constprop_report, config_report)
    print(f"{root.name} task result: tasks={[vars(task) for task in result['tasks']]!r}, "
          f"skipped={result['skipped']!r}")
    assert result["problem"] is None
    assert result["counts"]["considered"] == (
        result["counts"]["tasks"] + sum(result["counts"]["skipped"].values()))
    assert edges == old_edges
    assert literal_report == old_literal
    assert constprop_report == old_constprop
    assert config_report == old_config
    return records, edges, result


def test_imported_constant_builds_task_and_slice(tmp_path):
    root = tmp_path
    _write(root, "config.py", 'API_URL = "https://api.example.test"\n')
    _write(root, "app.py", "from config import API_URL\nrequests.get(API_URL)\n")
    records, edges, result = _tasks(root)
    assert result["counts"] == {"considered": 1, "tasks": 1, "skipped": {}}
    assert result["skipped"] == {}
    task = result["tasks"][0]
    assert (task.task_id, task.edge_index, task.file, task.line, task.callee_expr,
            task.reason, task.variable, task.language) == (
                "edge-0", 0, "app.py", 2, "requests.get", "imported_constant",
                "API_URL", "python")
    assert (task.url_arg_index, task.url_arg_keyword, task.url_arg_property) == (0, "url", None)
    built = build_slice(task, {record.path: record for record in records}, edges, root)
    print(f"imported-constant task slice: {built.files!r}")
    assert built.files == ["app.py", "config.py"]


def test_parameter_builds_task_and_slice(tmp_path):
    root = tmp_path
    _write(root, "app.py", "def f(url):\n    requests.get(url)\n")
    records, edges, result = _tasks(root)
    assert result["counts"]["tasks"] == 1
    task = result["tasks"][0]
    assert task.reason == "parameter"
    assert task.variable == "url"
    built = build_slice(task, {record.path: record for record in records}, edges, root)
    print(f"parameter task slice: {built.files!r}")
    assert built.files == ["app.py"]


def test_multiple_assignments_builds_task(tmp_path):
    root = tmp_path
    _write(root, "app.py", "def f(flag):\n    if flag:\n        url = 'https://a.invalid'\n"
                           "    else:\n        url = 'https://b.invalid'\n"
                           "    requests.get(url)\n")
    _, _, result = _tasks(root)
    assert result["counts"]["tasks"] == 1
    assert result["tasks"][0].reason == "multiple_assignments"
    assert result["tasks"][0].variable == "url"


def test_env_reference_is_skipped_as_config_owned(tmp_path):
    root = tmp_path
    _write(root, "app.py", "import os\nrequests.get(os.environ['X'])\n")
    _, _, result = _tasks(root)
    assert result["tasks"] == []
    assert result["skipped"] == {0: "env_var_config_owned"}
    assert result["counts"] == {
        "considered": 1, "tasks": 0, "skipped": {"env_var_config_owned": 1}}


def test_literal_host_is_skipped(tmp_path):
    root = tmp_path
    _write(root, "app.py", 'requests.get("https://literal.example.test")\n')
    _, _, result = _tasks(root)
    assert result["tasks"] == []
    assert result["skipped"] == {0: "has_host"}


def test_missing_constprop_outcome_is_skipped(tmp_path):
    root = tmp_path
    _write(root, "app.py", "requests.get(123)\n")
    _, _, result = _tasks(root)
    assert result["tasks"] == []
    assert result["skipped"] == {0: "no_reason_recorded"}


def test_out_of_scope_cycle_outcome_is_skipped(tmp_path):
    root = tmp_path
    _write(root, "app.py", "def f():\n    url = url\n    requests.get(url)\n")
    records, edges, literal, constprop, config = _pipeline(root)
    assert constprop["still_pending"] == {0: "cycle"}
    old_edges = deepcopy(edges)
    old_literal = deepcopy(literal)
    old_constprop = deepcopy(constprop)
    old_config = deepcopy(config)
    result = build_joern_tasks(edges, records, literal, constprop, config)
    print(f"out-of-scope real const-prop outcome: {constprop['still_pending'][0]!r}")
    print(f"{root.name} task result: tasks={[vars(task) for task in result['tasks']]!r}, "
          f"skipped={result['skipped']!r}")
    assert result["skipped"] == {0: "reason_not_in_scope:cycle"}
    assert result["counts"]["considered"] == (
        result["counts"]["tasks"] + sum(result["counts"]["skipped"].values()))
    assert edges == old_edges
    assert literal == old_literal
    assert constprop == old_constprop
    assert config == old_config


def test_javascript_parameter_builds_task(tmp_path):
    root = tmp_path
    _write(root, "app.js", "function f(url) { axios.get(url); }\n")
    _, _, result = _tasks(root)
    assert result["counts"]["tasks"] == 1
    task = result["tasks"][0]
    assert task.language == "javascript"
    assert task.reason == "parameter"
    assert task.variable == "url"
    assert (task.url_arg_index, task.url_arg_keyword, task.url_arg_property) == (0, None, None)



