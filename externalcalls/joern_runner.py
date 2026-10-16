"""Run real Joern data-flow queries for one merged source slice.

``no_sources_found`` means UNKNOWN: Joern produced no flow beginning at a
literal node. It is never evidence that a value has no source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Mapping, Optional, Sequence

from externalcalls.joern_positions import parser_to_joern_start, read_line
from externalcalls.joern_tasks import JoernLimits
from externalcalls.slice_builder import JoernTask, Slice


_JSON_PREFIX = "PEACH_JOERN_JSON:"
_DEFAULT_JOERN_HOME = Path("/opt/joern/joern-cli")
_DEFAULT_TIMEOUT_SECONDS = 300.0
_LANGUAGES = {
    "python": ("PYTHONSRC", "python"),
    "javascript": ("JSSRC", "javascript"),
    "typescript": ("JSSRC", "typescript"),
}


def _relative_path(value):
    value = str(value).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _scala_string(value):
    return json.dumps(str(value), ensure_ascii=False)


def _position(task, source_root):
    line_text, error = read_line(source_root, _relative_path(task.file), task.line)
    if error:
        return None, error
    language = str(task.language).lower()
    column, error = parser_to_joern_start(
        line_text, task.callee_expr, task.callee_col, language=language)
    if error:
        return None, error
    return column, None


def _selected_argument_condition(task):
    if task.url_arg_keyword:
        return f"a.argumentName.exists(_ == {_scala_string(task.url_arg_keyword)})"
    if task.url_arg_index is not None:
        # Catalogue indexes are zero-based among positional arguments; Joern
        # numbers the receiver at zero, so positional argument indexes +1.
        return f"a.argumentIndex == {int(task.url_arg_index) + 1}"
    return "false"


def _query_script(cpg_path, task_positions):
    lines = [
        "import io.joern.dataflowengineoss.layers.dataflows._",
        "import io.joern.dataflowengineoss.queryengine.EngineContext",
        "import io.joern.dataflowengineoss.queryengine.EngineConfig",
        f"importCpg({_scala_string(cpg_path)})",
        "implicit val depth4: EngineContext = EngineContext().copy(config = EngineContext().config.copy(maxCallDepth = 4))",
        "run.ossdataflow",
    ]
    json_helper = r'''def jsonString(value: String): String = {
  val escaped = value.flatMap {
    case '"' => "\\\""
    case '\\' => "\\\\"
    case '\b' => "\\b"
    case '\f' => "\\f"
    case '\n' => "\\n"
    case '\r' => "\\r"
    case '\t' => "\\t"
    case c if c < ' ' => "\\u" + ("%04x" format c.toInt)
    case c => c.toString
  }
  "\"" + escaped + "\""
}'''
    lines.extend(json_helper.splitlines())
    lines.append("val literalNodes = cpg.literal.l")

    for task, joern_column in task_positions:
        task_id = _scala_string(task.task_id)
        callee = _scala_string(task.callee_expr)
        sink_file = _scala_string(_relative_path(task.file))
        sink_basename = _scala_string(Path(_relative_path(task.file)).name)
        expected_line = int(task.line)
        expected_column = int(joern_column)
        condition = _selected_argument_condition(task)
        lines.extend([
            "{",
            f"  val taskId = {task_id}",
            "  val matchingSinks = cpg.call.l.filter(c => ",
            f"    c.lineNumber.contains({expected_line}) && c.columnNumber.contains({expected_column}) &&",
            f"    c.file.name.headOption.exists(file => file == {sink_file} || file.endsWith(\"/\" + {sink_file}) || file == {sink_basename}) &&",
            '    c.methodFullName != "<operator>.fieldAccess" &&',
            f"    c.code.trim.startsWith({callee} + \"(\") && c.code.trim.endsWith(\")\"))",
            '  if (matchingSinks.isEmpty) println("' + _JSON_PREFIX + '" + "{\\\"task_id\\\":" + jsonString(taskId) + ",\\\"sink_found\\\":false,\\\"reason\\\":\\\"sink_not_found\\\"}")',
            '  else if (matchingSinks.size != 1) println("' + _JSON_PREFIX + '" + "{\\\"task_id\\\":" + jsonString(taskId) + ",\\\"sink_found\\\":false,\\\"reason\\\":\\\"sink_ambiguous\\\"}")',
            "  else {",
            "    val sink = matchingSinks.head",
            f"    val selectedArguments = sink.argument.l.filter(a => {condition})",
            "    val selectedTraversal = sink.argument.filter(a => selectedArguments.exists(selected => selected.argumentIndex == a.argumentIndex && selected.argumentName == a.argumentName && selected.code == a.code))",
            "    val paths = selectedTraversal.reachableByFlows(cpg.literal).l.filter(path =>",
            "      path.elements.headOption.exists(first => literalNodes.exists(_ == first))",
            "    )",
            "    val sourcesJson = paths.map { path =>",
            "      val elements = path.elements.toList",
            "      val elementsJson = elements.map(n =>",
            '        "{\\\"file\\\":" + jsonString(n.file.name.headOption.getOrElse("")) +',
            '        ",\\\"line\\\":" + n.lineNumber.map(_.toString).getOrElse("null") +',
            '        ",\\\"column\\\":" + n.columnNumber.map(_.toString).getOrElse("null") +',
            '        ",\\\"code\\\":" + jsonString(n.code) + "}"',
            "      ).mkString(\"[\", \",\", \"]\")",
            "      val first = elements.head",
            '      "{\\\"file\\\":" + jsonString(first.file.name.headOption.getOrElse("")) +',
            '      ",\\\"line\\\":" + first.lineNumber.map(_.toString).getOrElse("null") +',
            '      ",\\\"column\\\":" + first.columnNumber.map(_.toString).getOrElse("null") +',
            '      ",\\\"code\\\":" + jsonString(first.code) +',
            '      ",\\\"elements\\\":" + elementsJson + "}"',
            "    }.mkString(\"[\", \",\", \"]\")",
            '    println("' + _JSON_PREFIX + '" + "{\\\"task_id\\\":" + jsonString(taskId) + ",\\\"sink_found\\\":true,\\\"sources\\\":" + sourcesJson + "}")',
            "  }",
            "}",
        ])
    return "\n".join(lines) + "\n"


def _kill_process_tree(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _run_streaming(command, cwd, env, timeout_seconds):
    """Run one command while draining both pipes and killing its whole tree."""
    options = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True

    try:
        process = subprocess.Popen(command, **options)
    except OSError as error:
        return {
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": str(error),
        }

    stdout_parts = []
    stderr_parts = []

    def drain(stream, parts):
        for line in iter(stream.readline, ""):
            parts.append(line)

    readers = [
        threading.Thread(target=drain, args=(process.stdout, stdout_parts), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr_parts), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        exit_code = process.wait(timeout=max(0.0, float(timeout_seconds)))
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(process)
        exit_code = process.wait()
    for reader in readers:
        reader.join()
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
    }


def _diagnostic(prefix, result):
    parts = [f"{prefix} (exit_code={result['exit_code']})"]
    if result["stdout"]:
        parts.append("stdout:\n" + result["stdout"])
    if result["stderr"]:
        parts.append("stderr:\n" + result["stderr"])
    return "\n".join(parts)


def _failed(task, reason):
    return {
        "task_id": task.task_id,
        "status": "failed",
        "sources": [],
        "reason": reason,
    }


def _parse_json_events(stdout):
    events = {}
    for line in stdout.splitlines():
        if not line.startswith(_JSON_PREFIX):
            continue
        payload = line[len(_JSON_PREFIX):]
        event = json.loads(payload)
        events[event["task_id"]] = event
    return events


_STRING_LITERAL = re.compile(
    r"^(?:(?i:rb|br|fr|rf|r|f|b|u))?(?P<quote>'''|\"\"\"|'|\"|`)[\s\S]*?(?P=quote)$"
)
_IMPORT_LINE = re.compile(r"^\s*(?:import\b|from\b)")
_IMPORT_CALL = re.compile(r"\b(?:require|import)\s*\((.*?)\)")


def _source_path(source_root, reported_file):
    """Resolve Joern's reported file name against the copied slice."""
    root = Path(source_root)
    reported = str(reported_file or "").replace("\\", "/")
    if not reported:
        return None
    direct = root / reported.lstrip("/")
    if direct.is_file():
        return direct
    suffix = "/" + reported.lstrip("/")
    matches = [path for path in root.rglob("*") if path.is_file()
               and (path.as_posix().endswith(suffix)
                    or path.name == Path(reported).name)]
    return matches[0] if len(matches) == 1 else None


def _source_is_import_literal(source, source_root):
    line_number = source.get("line")
    if line_number is None:
        return False
    path = _source_path(source_root, source.get("file"))
    if path is None:
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        line = lines[int(line_number) - 1]
    except (OSError, UnicodeError, ValueError, IndexError, TypeError):
        return False
    if _IMPORT_LINE.match(line):
        return True
    code = str(source.get("code", "")).strip()
    return any(code and code in match.group(1)
               for match in _IMPORT_CALL.finditer(line))


def _filter_literal_sources(sources, source_root):
    """Keep string literal sources, excluding dependency import literals."""
    return [source for source in sources
            if _STRING_LITERAL.fullmatch(str(source.get("code", "")).strip())
            and not _source_is_import_literal(source, source_root)]


def _same_file_parameter_caller(task, source_root):
    """Recognize a call to the parameter-owning function in the sink file."""
    path = _source_path(source_root, task.file)
    if path is None:
        return False
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False

    line_start = sum(len(line) + 1 for line in source.splitlines()[:int(task.line) - 1])
    declarations = re.finditer(
        r"(?m)^(?P<indent>[ \t]*)def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*:",
        source,
    )
    for declaration in declarations:
        if not re.search(rf"\b{re.escape(str(task.variable))}\b", declaration.group("params")):
            continue
        body_start = declaration.end()
        declaration_indent = len(declaration.group("indent").expandtabs(8))
        body_end = len(source)
        for line in source[body_start:].splitlines(keepends=True):
            content = line.strip()
            if content:
                indent = len(line) - len(line.lstrip(" \t"))
                if indent <= declaration_indent:
                    body_end = body_start + source[body_start:].find(line)
                    break
            body_start += len(line)
        if not (declaration.start() <= line_start < body_end):
            continue
        function_name = re.escape(declaration.group("name"))
        return any(match.start() < declaration.start() or match.start() >= body_end
                   for match in re.finditer(rf"\b{function_name}\s*\(", source))

    declarations = re.finditer(
        r"\bfunction\s+(?P<name>[A-Za-z_$][\w$]*)\s*\((?P<params>[^)]*)\)\s*\{",
        source,
    )
    for declaration in declarations:
        if not re.search(rf"\b{re.escape(str(task.variable))}\b", declaration.group("params")):
            continue
        depth = 1
        end = declaration.end()
        while end < len(source) and depth:
            if source[end] == "{":
                depth += 1
            elif source[end] == "}":
                depth -= 1
            end += 1
        if not (declaration.start() <= line_start < end):
            continue
        function_name = re.escape(declaration.group("name"))
        return any(match.start() < declaration.start() or match.start() >= end
                   for match in re.finditer(rf"\b{function_name}\s*\(", source))
    return False


def _task_is_incomplete(task, slice_metadata, source_root=None):
    marked = bool(getattr(task, "incomplete", False))
    marked = marked or bool(getattr(slice_metadata, "incomplete", False))
    if task.reason == "parameter":
        files = getattr(slice_metadata, "files", ()) if slice_metadata is not None else ()
        sink = _relative_path(task.file)
        caller_files = {
            _relative_path(path) for path in files
            if _relative_path(path) != sink
        }
        same_file_caller = (source_root is not None
                            and _same_file_parameter_caller(task, source_root))
        if not caller_files and not same_file_caller:
            return True, "parameter task has no caller files in its slice"
    return marked, "task or slice is marked incomplete" if marked else None


def _result_for_task(task, event, *, incomplete, incomplete_reason,
                     query_timed_out, query_diagnostic):
    if event is None:
        if query_timed_out:
            return {
                "task_id": task.task_id,
                "status": "incomplete",
                "sources": [],
                "reason": "Joern query timed out before this task emitted a result.\n"
                + query_diagnostic,
            }
        return _failed(task, "query_result_missing\n" + query_diagnostic)

    if not event.get("sink_found", False):
        return _failed(task, event.get("reason", "sink_not_found"))

    sources = event.get("sources", [])
    if query_timed_out:
        reason = "Joern query timed out; any emitted flows are partial.\n" + query_diagnostic
        return {
            "task_id": task.task_id,
            "status": "incomplete",
            "sources": sources,
            "reason": reason,
        }
    if incomplete:
        return {
            "task_id": task.task_id,
            "status": "incomplete",
            "sources": sources,
            "reason": incomplete_reason,
        }
    if not sources:
        return {
            "task_id": task.task_id,
            "status": "no_sources_found",
            "sources": [],
            "reason": "No literal-start flow found; result is UNKNOWN.",
        }
    return {
        "task_id": task.task_id,
        "status": "complete",
        "sources": sources,
        "reason": None,
    }


def run_joern_tasks(
    tasks: Sequence[JoernTask],
    slice_directory,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    java_tool_options: Optional[str] = "-Xmx512m",
    joern_home=_DEFAULT_JOERN_HOME,
    task_slices: Optional[Mapping[str, Slice]] = None,
):
    """Run one CPG and one baked-path script for tasks in a merged slice.

    ``task_slices`` maps task ids to their existing ``Slice`` values. It
    carries per-task truncation and caller-file information that is not stored
    on ``JoernTask``. A parameter task without a supplied caller file is
    incomplete, never a confident no-source result.

    Each temporary run gets a fresh cwd, so Joern cannot reuse stale workspace
    state. Every process has a hard timeout and its process tree is killed.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    source_root = Path(slice_directory).resolve()
    home = Path(joern_home)
    joern_parse = home / "joern-parse"
    joern = home / "joern"

    task_errors = {}
    positioned = []
    languages = set()
    for task in tasks:
        if not isinstance(task, JoernTask):
            task_errors[task.task_id] = "task must be an externalcalls.slice_builder.JoernTask"
            continue
        language = _LANGUAGES.get(str(task.language).lower())
        if language is None:
            task_errors[task.task_id] = f"unsupported_language:{task.language}"
            continue
        languages.add(language)
        column, error = _position(task, source_root)
        if error:
            task_errors[task.task_id] = error
            continue
        positioned.append((task, column))

    if len(languages) != 1:
        message = "one Joern CPG can contain tasks from only one language"
        return [_failed(task, task_errors.get(task.task_id, message)) for task in tasks]

    language = next(iter(languages))[0]
    env = os.environ.copy()
    if java_tool_options is None:
        env.pop("JAVA_TOOL_OPTIONS", None)
    else:
        env["JAVA_TOOL_OPTIONS"] = str(java_tool_options)

    results = {}
    with tempfile.TemporaryDirectory(prefix="peach-joern-") as temporary:
        root = Path(temporary)
        source_copy = root / "source"
        try:
            shutil.copytree(source_root, source_copy)
        except OSError as error:
            return [_failed(task, f"slice_copy_failed: {error}") for task in tasks]
        parse_cwd = root / "parse-cwd"
        query_cwd = root / "query-cwd"
        parse_cwd.mkdir()
        query_cwd.mkdir()
        cpg_path = root / "slice.cpg.bin"

        parse_result = _run_streaming(
            [str(joern_parse), str(source_copy), "--language", language,
             "--output", str(cpg_path)],
            parse_cwd,
            env,
            timeout_seconds,
        )
        if parse_result["timed_out"]:
            diagnostic = _diagnostic("joern-parse timeout", parse_result)
            return [_failed(task, diagnostic) for task in tasks]
        if parse_result["exit_code"] != 0:
            diagnostic = _diagnostic("joern-parse failed", parse_result)
            return [_failed(task, diagnostic) for task in tasks]

        for task in tasks:
            if task.task_id in task_errors:
                results[task.task_id] = _failed(task, task_errors[task.task_id])
        valid_positions = [(task, column) for task, column in positioned
                           if task.task_id not in task_errors]
        if not valid_positions:
            return [results[task.task_id] for task in tasks]

        script_path = query_cwd / "query.sc"
        script_path.write_text(
            _query_script(cpg_path, valid_positions), encoding="utf-8", newline="\n")
        query_result = _run_streaming(
            [str(joern), "--script", str(script_path)],
            query_cwd,
            env,
            timeout_seconds,
        )
        if query_result["exit_code"] != 0 and not query_result["timed_out"]:
            diagnostic = _diagnostic("Joern query failed", query_result)
            return [_failed(task, results.get(task.task_id, {}).get("reason", diagnostic)
                            if task.task_id in results else diagnostic)
                    for task in tasks]

        try:
            events = _parse_json_events(query_result["stdout"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            diagnostic = _diagnostic(
                f"Joern JSON output parse failed: {error}", query_result)
            return [_failed(task, diagnostic) for task in tasks]

        diagnostic = _diagnostic("Joern query timeout", query_result)
        for task in tasks:
            if task.task_id in results:
                continue
            metadata = (task_slices or {}).get(task.task_id)
            incomplete, incomplete_reason = _task_is_incomplete(task, metadata, source_copy)
            event = events.get(task.task_id)
            if event is not None and event.get("sink_found", False):
                event = dict(event)
                event["sources"] = _filter_literal_sources(
                    event.get("sources", []), source_copy)
            results[task.task_id] = _result_for_task(
                task,
                event,
                incomplete=incomplete,
                incomplete_reason=incomplete_reason,
                query_timed_out=query_result["timed_out"],
                query_diagnostic=diagnostic,
            )

    return [results[task.task_id] for task in tasks]


__all__ = ["JoernLimits", "run_joern_tasks"]
