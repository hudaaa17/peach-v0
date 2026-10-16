"""End-to-end J4 tests using the real parser, runner and Joern CLI."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from analyzer_new.gaps import Gap
from analyzer_new.parser import parse_repo
from analyzer_new.resolver import resolve_calls
from externalcalls.ast_grep_check import CATALOGUE, run_patterns
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls.constprop_check import run_constprop_check
from externalcalls.edge_pairing import pair_edges_with_calls
from externalcalls.joern_apply import apply_joern_results
from externalcalls.joern_runner import run_joern_tasks
from externalcalls.joern_tasks import build_joern_tasks
from externalcalls.literal_check import run_literal_check
from externalcalls.slice_builder import JoernTask, build_slice, materialize_slice


JOERN = Path("/opt/joern/joern-cli/joern")
pytestmark = pytest.mark.skipif(
    not JOERN.exists(), reason="real Joern 4.0.653 CLI is not installed")


def _write(root, files):
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
    return root


def _diagnostic(context):
    root = context.get("root")
    task_slices = context.get("task_slices", {})
    task_diagnostics = []
    for task in context.get("tasks", []):
        source_slice = task_slices.get(task.task_id)
        slice_files = list(getattr(source_slice, "files", ()) or ())
        slice_file_texts = {}
        for relative_path in slice_files:
            try:
                path_parts = str(relative_path).replace("\\", "/").split("/")
                slice_file_texts[str(relative_path)] = (
                    Path(root, *path_parts).read_text(
                        encoding="utf-8", errors="replace"))
            except (OSError, TypeError, ValueError) as error:
                slice_file_texts[str(relative_path)] = (
                    f"<read failed: {type(error).__name__}: {error}>")
        task_diagnostics.append({
            "task": asdict(task),
            "slice_files": slice_files,
            "slice_incomplete": getattr(source_slice, "incomplete", None),
            "slice_omitted": getattr(source_slice, "omitted", None),
            "slice_file_texts": slice_file_texts,
        })
    return "container: Full J4 test result on assertion failure:\n" + json.dumps(
        {
            "runner_results": context.get("runner_results", []),
            "task_diagnostics": task_diagnostics,
            "apply_report": context.get("apply_report"),
            "gaps": [asdict(gap) if isinstance(gap, Gap) else gap
                     for gap in context.get("gaps", [])],
            "edges": [vars(edge) for edge in context.get("edges", [])],
            "task_report": context.get("task_report"),
            "literal_report": context.get("literal_report"),
            "constprop_report": context.get("constprop_report"),
        },
        ensure_ascii=False,
        default=str,
        indent=2,
    )


def _expect(actual, expected, context):
    assert actual == expected, _diagnostic(context)


def _one_task(context, file=None):
    tasks = context["tasks"]
    if file is not None:
        tasks = [task for task in tasks if task.file == file]
    if len(tasks) != 1:
        raise AssertionError(_diagnostic(context))
    return tasks[0]


def _prepare(root, *, direct_tasks=None):
    records = parse_repo(root)
    edges = resolve_calls(records, repo_root=None)
    groups = {}
    for record in records:
        groups.setdefault(record.language, []).append(record.path)
    hits, pattern_report = run_patterns(root, groups)
    join_report = join_ast_grep_hits(edges, records, hits, pattern_report)
    literal_report = run_literal_check(edges, records, root)
    constprop_report = run_constprop_check(edges, records, root, literal_report)
    task_report = build_joern_tasks(edges, records, literal_report, constprop_report)
    records_by_path = {record.path: record for record in records}
    tasks = list(task_report["tasks"])
    if direct_tasks:
        # The current task builder only emits pending-name outcomes. For the
        # required config[key] cases, construct task fields from the paired
        # CallSite and catalogue, then use the real slice builder and runner.
        calls, problem = pair_edges_with_calls(edges, records)
        if problem:
            raise RuntimeError(f"call pairing failed: {problem}")
        catalogue = {entry.id: entry for entry in CATALOGUE}
        for file, variable in direct_tasks.items():
            edge_index = next(
                index for index, edge in enumerate(edges)
                if edge.file == file and edge.status == "external_candidate"
                and not any(task.edge_index == index for task in tasks)
            )
            edge = edges[edge_index]
            record = records_by_path[edge.file]
            entry = catalogue[edge.external_pattern]
            call = calls[edge_index]
            tasks.append(JoernTask(
                task_id=f"j4-direct-{edge_index}",
                edge_index=edge_index,
                file=edge.file,
                line=call.callee_line,
                callee_expr=edge.callee_expr,
                callee_col=call.callee_col,
                reason="complex_binding",
                variable=variable,
                language=record.language,
                url_arg_index=entry.url_arg_index,
                url_arg_keyword=entry.url_arg_keyword,
                url_arg_property=entry.url_arg_property,
            ))
    task_slices = {
        task.task_id: build_slice(task, records_by_path, edges, root)
        for task in tasks
    }
    context = {
        "root": root,
        "records": records,
        "edges": edges,
        "tasks": tasks,
        "direct_tasks": list(direct_tasks or {}),
        "task_slices": task_slices,
        "task_report": task_report,
        "pattern_report": pattern_report,
        "join_report": join_report,
        "literal_report": literal_report,
        "constprop_report": constprop_report,
        "runner_results": [],
        "apply_report": None,
        "gaps": [],
    }
    if task_report.get("problem") is not None:
        raise RuntimeError(_diagnostic(context))
    return context


def _run(context, tmp_path, *, timeout_seconds=300):
    files = list(dict.fromkeys(
        path for task in context["tasks"]
        for path in context["task_slices"][task.task_id].files
    ))
    input_dir = tmp_path / "joern-input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    materialized, error = materialize_slice(files, context["root"], input_dir)
    if error:
        raise RuntimeError(f"slice materialization failed: {error}\n{_diagnostic(context)}")
    context["runner_results"] = run_joern_tasks(
        context["tasks"], materialized,
        timeout_seconds=timeout_seconds,
        task_slices=context["task_slices"],
    )
    return context["runner_results"]


def _apply(context):
    context["apply_report"] = apply_joern_results(
        context["edges"], context["tasks"], context["runner_results"],
        context["root"], task_slices=context["task_slices"], gaps=context["gaps"])
    return context["apply_report"]


def _edge_for_task(context, task):
    return context["edges"][task.edge_index]


def _candidate(literal, host, file, line, column, key_filtered=False):
    return {
        "literal": literal,
        "host": host,
        "file": file,
        "line": line,
        "column": column,
        "key_filtered": key_filtered,
    }


def _gap_kinds(context):
    return [gap.kind for gap in context["gaps"]]


def test_a_imported_constant_sets_host_and_exact_source_location(tmp_path):
    root = _write(tmp_path / "repo", {
        "app.py": "from config import URL\nrequests.get(URL)\n",
        "config.py": 'URL = "https://imported.invalid/path"\n',
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    before = vars(_edge_for_task(context, task)).copy()
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(_edge_for_task(context, task).host, "imported.invalid", context)
    _expect(_edge_for_task(context, task).needs_review, True, context)
    _expect(report["joern_candidates"][task.edge_index], [
        _candidate("https://imported.invalid/path", "imported.invalid",
                   "config.py", 1, 6)
    ], context)
    after = vars(_edge_for_task(context, task)).copy()
    expected = dict(before, host="imported.invalid", needs_review=True,
                    review_reason="joern_result")
    _expect(after, expected, context)


def test_b_two_callers_with_different_hosts_are_ambiguous(tmp_path):
    root = _write(tmp_path / "repo", {
        "service.py": "import requests\ndef send(url):\n    requests.get(url)\n",
        "first.py": 'from service import send\nsend("https://first.invalid/a")\n',
        "second.py": 'from service import send\nsend("https://second.invalid/b")\n',
    })
    context = _prepare(root)
    task = _one_task(context, "service.py")
    _run(context, tmp_path)

    report = _apply(context)

    edge = _edge_for_task(context, task)
    _expect(edge.host, None, context)
    candidates = report["joern_candidates"][task.edge_index]
    _expect(len(candidates), 2, context)
    _expect({item["host"] for item in candidates},
            {"first.invalid", "second.invalid"}, context)
    _expect(_gap_kinds(context), ["joern_ambiguous_hosts"], context)
    _expect(context["gaps"][0].samples, ["first.invalid", "second.invalid"], context)
    _expect(report["counters"]["ambiguous"], 1, context)


def test_c_two_paths_on_one_host_set_host(tmp_path):
    root = _write(tmp_path / "repo", {
        "service.py": "import requests\ndef send(url):\n    requests.get(url)\n",
        "first.py": 'from service import send\nsend("https://shared.invalid/a")\n',
        "second.py": 'from service import send\nsend("https://shared.invalid/b")\n',
    })
    context = _prepare(root)
    task = _one_task(context, "service.py")
    _run(context, tmp_path)

    report = _apply(context)

    candidates = report["joern_candidates"][task.edge_index]
    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(len(candidates), 2, context)
    _expect(_edge_for_task(context, task).host, "shared.invalid", context)
    _expect({candidate["literal"] for candidate in candidates},
            {"https://shared.invalid/a", "https://shared.invalid/b"}, context)
    _expect({candidate["host"] for candidate in candidates}, {"shared.invalid"}, context)


def test_d_literal_dict_key_keeps_only_matching_dictionary_value(tmp_path):
    source = ('import requests\n'
              'config = {"url": "https://url-value.invalid", '
              '"name": "https://other-value.invalid"}\n'
              'requests.get(config["url"])\n')
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root, direct_tasks={"app.py": "config"})
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    candidates = report["joern_candidates"][task.edge_index]
    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect([candidate["literal"] for candidate in candidates],
            ["https://url-value.invalid"], context)
    _expect(candidates, [
        _candidate("https://url-value.invalid", "url-value.invalid", "app.py", 2, 17, True)
    ], context)
    _expect(_edge_for_task(context, task).host, "url-value.invalid", context)


def test_e_dynamic_dict_key_keeps_candidates_but_does_not_set_host(tmp_path):
    source = ('import requests\n'
              'config = {"url": "https://url-value.invalid", '
              '"name": "https://other-value.invalid"}\n'
              'requests.get(config[key])\n')
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root, direct_tasks={"app.py": "config"})
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    edge = _edge_for_task(context, task)
    candidates = report["joern_candidates"][task.edge_index]
    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(len(candidates), 2, context)
    _expect({candidate["literal"] for candidate in candidates},
            {"https://url-value.invalid", "https://other-value.invalid"}, context)
    _expect(edge.host, None, context)
    _expect(edge.needs_review, True, context)


def test_f_if_else_with_different_hosts_is_ambiguous(tmp_path):
    source = ('import requests\n'
              'if flag:\n    URL = "https://if.invalid"\n'
              'else:\n    URL = "https://else.invalid"\n'
              'requests.get(URL)\n')
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(len(report["joern_candidates"][task.edge_index]), 2, context)
    _expect(_edge_for_task(context, task).host, None, context)
    _expect({candidate["host"] for candidate in report["joern_candidates"][task.edge_index]},
            {"if.invalid", "else.invalid"}, context)
    _expect(_gap_kinds(context), ["joern_ambiguous_hosts"], context)


def test_g_if_else_with_same_host_sets_host(tmp_path):
    source = ('import requests\n'
              'if flag:\n    URL = "https://same.invalid/one"\n'
              'else:\n    URL = "https://same.invalid/two"\n'
              'requests.get(URL)\n')
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(len(report["joern_candidates"][task.edge_index]), 2, context)
    _expect(_edge_for_task(context, task).host, "same.invalid", context)
    _expect({candidate["literal"] for candidate in report["joern_candidates"][task.edge_index]},
            {"https://same.invalid/one", "https://same.invalid/two"}, context)


def test_h_sink_file_without_literal_has_unknown_gap(tmp_path):
    source = "import requests\nrequests.get(URL)\n"
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "no_sources_found", context)
    _expect(_edge_for_task(context, task).host, None, context)
    _expect(_gap_kinds(context), ["joern_no_sources_found"], context)
    _expect(report["counters"]["unknown"], 1, context)


def test_i_tiny_runner_timeout_maps_to_timeout_gap(tmp_path):
    root = _write(tmp_path / "repo", {
        "config.py": 'URL = "https://timeout.invalid"\n',
        "app.py": "from config import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path, timeout_seconds=0.001)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "failed", context)
    _expect(_edge_for_task(context, task).host, None, context)
    _expect(_gap_kinds(context), ["joern_timeout"], context)
    _expect(report["counters"]["failed"], 1, context)


def test_j_parameter_without_caller_is_incomplete(tmp_path):
    source = "import requests\ndef send(url):\n    requests.get(url)\n"
    root = _write(tmp_path / "repo", {"service.py": source})
    context = _prepare(root)
    task = _one_task(context, "service.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "incomplete", context)
    _expect(_edge_for_task(context, task).host, None, context)
    _expect(_gap_kinds(context), ["joern_incomplete"], context)


def test_k_relative_literal_is_retained_without_a_host(tmp_path):
    root = _write(tmp_path / "repo", {
        "config.py": 'URL = "/relative/path"\n',
        "app.py": "from config import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(_edge_for_task(context, task).host, None, context)
    _expect(report["joern_candidates"][task.edge_index], [
        _candidate("/relative/path", None, "config.py", 1, 6)
    ], context)
    _expect(report["counters"]["unknown"], 1, context)


def test_l_edge_with_existing_host_is_untouched(tmp_path):
    root = _write(tmp_path / "repo", {
        "config.py": 'URL = "https://existing.invalid"\n',
        "app.py": "from config import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)
    edge = _edge_for_task(context, task)
    edge.host = "preexisting.invalid"
    edge.needs_review = False
    edge.review_reason = "prior_reason"
    before = vars(edge).copy()

    report = _apply(context)

    _expect(vars(edge), before, context)
    _expect(report["counters"]["skipped"], {"has_host": 1}, context)
    _expect(report["joern_candidates"], {}, context)


def test_m_emoji_before_literal_round_trips_to_utf8_byte_column(tmp_path):
    source = 'prefix = "\U0001F600"; URL = "https://emoji.invalid/path"\n'
    root = _write(tmp_path / "repo", {
        "config.py": source,
        "app.py": "from config import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    literal_column = len(source.splitlines()[0][:source.index('"https://')].encode("utf-8"))
    _expect(_edge_for_task(context, task).host, "emoji.invalid", context)
    _expect(report["joern_candidates"][task.edge_index], [
        _candidate("https://emoji.invalid/path", "emoji.invalid", "config.py", 1,
                   literal_column)
    ], context)


def test_n_javascript_parameter_from_imported_caller_sets_host(tmp_path):
    root = _write(tmp_path / "repo", {
        "service.js": "export function send(url) { fetch(url); }\n",
        "caller.js": 'import { send } from "./service";\nsend("https://javascript.invalid/path");\n',
    })
    context = _prepare(root)
    task = _one_task(context, "service.js")
    _run(context, tmp_path)

    report = _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    _expect(len(report["joern_candidates"][task.edge_index]), 1, context)
    _expect(_edge_for_task(context, task).host, "javascript.invalid", context)
    _expect({item["literal"] for item in report["joern_candidates"][task.edge_index]},
            {"https://javascript.invalid/path"}, context)


def test_o_reapplying_results_preserves_state_and_deduplicates_gaps(tmp_path):
    source = ('import requests\n'
              'if flag:\n    URL = "https://first.invalid"\n'
              'else:\n    URL = "https://second.invalid"\n'
              'requests.get(URL)\n')
    root = _write(tmp_path / "repo", {"app.py": source})
    context = _prepare(root)
    _one_task(context, "app.py")
    _run(context, tmp_path)
    first_report = _apply(context)
    _expect(context["runner_results"][0]["status"], "complete", context)
    first_state = [vars(edge).copy() for edge in context["edges"]]
    first_gaps = [asdict(gap) for gap in context["gaps"]]

    second_report = apply_joern_results(
        context["edges"], context["tasks"], context["runner_results"],
        context["root"], task_slices=context["task_slices"], gaps=context["gaps"])
    context["apply_report"] = second_report

    _expect([vars(edge).copy() for edge in context["edges"]], first_state, context)
    _expect([asdict(gap) for gap in context["gaps"]], first_gaps, context)
    _expect(first_report["gaps"], second_report["gaps"], context)


def test_p_call_edge_changes_only_host_review_fields(tmp_path):
    root = _write(tmp_path / "repo", {
        "config.py": 'URL = "https://fields.invalid"\n',
        "app.py": "from config import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    task = _one_task(context, "app.py")
    edge = _edge_for_task(context, task)
    before = vars(edge).copy()
    _run(context, tmp_path)

    _apply(context)

    _expect(context["runner_results"][0]["status"], "complete", context)
    after = vars(edge).copy()
    _expect(set(after), set(before), context)
    expected = dict(before, host="fields.invalid", needs_review=True,
                    review_reason="joern_result")
    _expect(after, expected, context)


def test_q_unreadable_source_does_not_stop_other_edges(tmp_path):
    root = _write(tmp_path / "repo", {
        "bad.py": 'URL = "https://deleted.invalid"\n',
        "bad_client.py": "from bad import URL\nrequests.get(URL)\n",
        "good.py": 'URL = "https://survives.invalid"\n',
        "good_client.py": "from good import URL\nrequests.get(URL)\n",
    })
    context = _prepare(root)
    _expect(len(context["tasks"]), 2, context)
    _run(context, tmp_path)
    _expect(len(context["runner_results"]), 2, context)
    _expect([result["status"] for result in context["runner_results"]],
            ["complete", "complete"], context)
    (root / "bad.py").unlink()

    report = _apply(context)

    bad_task = _one_task(context, "bad_client.py")
    good_task = _one_task(context, "good_client.py")
    _expect(_edge_for_task(context, bad_task).host, None, context)
    _expect(_edge_for_task(context, good_task).host, "survives.invalid", context)
    _expect("joern_source_unreadable" in _gap_kinds(context), True, context)
    _expect(report["counters"]["host_set"], 1, context)
