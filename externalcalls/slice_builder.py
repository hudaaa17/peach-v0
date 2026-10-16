"""Build deterministic, bounded source slices for Joern escalation."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Optional

from externalcalls.import_resolver import import_sort_name, resolve_import


@dataclass
class SliceBudget:
    max_files: int = 20
    max_lines: int = 4000
    max_import_distance: int = 1
    max_callers: int = 5


@dataclass
class JoernTask:
    task_id: str
    edge_index: int
    file: str
    line: int
    callee_expr: str
    callee_col: int
    reason: str
    variable: str
    language: str
    url_arg_index: Optional[int] = None
    url_arg_keyword: Optional[str] = None
    url_arg_property: Optional[str] = None


@dataclass
class Slice:
    files: list[str] = field(default_factory=list)
    total_lines: int = 0
    incomplete: bool = False
    omitted: list[tuple[str, str]] = field(default_factory=list)


def _relative(value):
    value = str(value).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _line_count(path):
    return path.read_text(encoding="utf-8", errors="ignore").count("\n") + 1


def _language(records, path, fallback):
    record = records.get(path)
    return getattr(record, "language", fallback) if record is not None else fallback


def _imports(records, path):
    record = records.get(path)
    return list(getattr(record, "imports", ())) if record is not None else []


def _import_priority(item, variable, path, root, language):
    raw = str(getattr(item, "raw", item)).replace("\\", "/")
    name = (import_sort_name(path, item, root)
            if language == "python" else raw.rsplit(".", 1)[-1])
    return (0 if name == variable else 1, raw)


def _shared_prefix(a, b):
    left, right = PurePosixPath(a).parts, PurePosixPath(b).parts
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def _parameter_callers(task, records_by_path, edges):
    if not isinstance(task.edge_index, int) or not 0 <= task.edge_index < len(edges):
        return [], None
    sink_edge = edges[task.edge_index]
    caller = getattr(sink_edge, "caller", None)
    if not caller:
        return [], None
    candidates = {}
    sink_dir = PurePosixPath(task.file).parent
    for edge in edges:
        if caller not in (getattr(edge, "resolved_targets", ()) or ()):
            continue
        path = _relative(getattr(edge, "file", ""))
        if not path or path == task.file or path not in records_by_path:
            continue
        if _language(records_by_path, path, task.language) != task.language:
            continue
        candidates[path] = edge
    ordered = sorted(candidates, key=lambda path: (
        0 if PurePosixPath(path).parent == sink_dir else 1,
        -_shared_prefix(task.file, path),
        path,
    ))
    return ordered, caller


def build_slice(task, records_by_path, edges, repo_root, budget=None):
    """Build a same-language slice, always listing the sink file first."""
    budget = budget or SliceBudget()
    root = Path(repo_root)
    sink = _relative(getattr(task, "file", ""))
    if not sink:
        return Slice([], 0, True, [("", "sink_file_unreadable")])
    sink_path = root / sink
    try:
        sink_lines = _line_count(sink_path)
    except (OSError, TypeError, ValueError):
        return Slice([sink], 0, True, [(sink, "sink_file_unreadable")])
    result = Slice([sink], sink_lines, False, [])
    file_lines = {sink: sink_lines}
    setattr(result, "_file_lines", file_lines)
    if sink_lines > budget.max_lines:
        result.incomplete = True
        result.omitted.append((sink, "sink_file_too_large"))
        return result

    records_by_path = {_relative(path): record for path, record in records_by_path.items()}
    sink_language = _language(records_by_path, sink, getattr(task, "language", ""))
    reason = getattr(task, "reason", "")
    if reason in ("env_var", "value_not_static") or reason not in {
            "imported_constant", "not_found", "parameter", "multiple_assignments",
            "member_access", "call_expression", "complex_binding"}:
        result.incomplete = True
        result.omitted.append((sink, "not_joern_task"))
        return result
    if reason == "multiple_assignments":
        return result
    if reason == "parameter":
        candidates, _ = _parameter_callers(task, records_by_path, edges)
        chosen = candidates[:max(0, budget.max_callers)]
        for path in candidates[len(chosen):]:
            result.incomplete = True
            result.omitted.append((path, "caller_limit"))
        for path in chosen:
            _add_file(result, path, root, records_by_path, sink_language,
                      budget, file_lines)
        return result

    import_depth = budget.max_import_distance if reason in ("imported_constant", "not_found") else min(
        1, budget.max_import_distance)
    queue = [(sink, 0)]
    visited = {sink}
    while queue:
        current, depth = queue.pop(0)
        current_lang = _language(records_by_path, current, sink_language)
        imports = sorted(_imports(records_by_path, current),
                         key=lambda item: _import_priority(item, getattr(task, "variable", ""),
                                                          current, root, current_lang))
        if depth >= import_depth:
            for item in imports:
                target, status = resolve_import(current, item, set(records_by_path),
                                                current_lang, repo_root=root)
                if status == "external":
                    continue
                if status == "unresolved":
                    result.incomplete = True
                    result.omitted.append((str(getattr(item, "raw", item)), "unresolved_import"))
                    continue
                if status == "alias_unresolved":
                    result.incomplete = True
                    result.omitted.append((str(getattr(item, "raw", item)),
                                           "alias_import_unresolved"))
                    continue
                if target and target not in visited:
                    result.incomplete = True
            continue
        for item in imports:
            target, status = resolve_import(current, item, set(records_by_path),
                                            current_lang, repo_root=root)
            if status == "external":
                continue
            if status == "unresolved":
                result.incomplete = True
                result.omitted.append((str(getattr(item, "raw", item)), "unresolved_import"))
                continue
            if status == "alias_unresolved":
                result.incomplete = True
                result.omitted.append((str(getattr(item, "raw", item)), "alias_import_unresolved"))
                continue
            if not target or target in visited:
                continue
            if _language(records_by_path, target, "") != sink_language:
                result.incomplete = True
                result.omitted.append((target, "unresolved_import"))
                continue
            visited.add(target)
            if _add_file(result, target, root, records_by_path, sink_language,
                         budget, file_lines):
                queue.append((target, depth + 1))
    return result


def _add_file(result, path, root, records_by_path, language, budget, file_lines):
    if path in result.files:
        return True
    if len(result.files) >= budget.max_files:
        result.incomplete = True
        result.omitted.append((path, "budget_files"))
        return False
    if _language(records_by_path, path, "") != language:
        result.incomplete = True
        result.omitted.append((path, "unresolved_import"))
        return False
    try:
        count = _line_count(root / path)
    except (OSError, TypeError, ValueError):
        result.incomplete = True
        result.omitted.append((path, "unresolved_import"))
        return False
    if result.total_lines + count > budget.max_lines:
        result.incomplete = True
        result.omitted.append((path, "budget_lines"))
        return False
    result.files.append(path)
    result.total_lines += count
    file_lines[path] = count
    return True


def group_tasks(items, budget=None):
    """Greedily group overlapping same-language slices within both budgets."""
    budget = budget or SliceBudget()
    ordered = sorted(items, key=lambda item: (getattr(item[0], "file", ""),
                                              getattr(item[0], "task_id", "")))
    runs = []
    run_counts = []
    run_excluded = []
    for task, source_slice in ordered:
        language = getattr(task, "language", "")
        files = list(dict.fromkeys(source_slice.files))
        excluded = any(why in ("not_joern_task", "sink_file_unreadable", "sink_file_too_large")
                       for _, why in source_slice.omitted)
        merged = False
        if not excluded:
            counts = _file_counts(source_slice)
            for index, (run_language, run_files, task_ids) in enumerate(runs):
                if (run_excluded[index] or run_language != language
                        or not set(run_files).intersection(files)):
                    continue
                union = run_files + [path for path in files if path not in run_files]
                union_counts = dict(run_counts[index])
                for path in union:
                    if path not in union_counts and path in counts:
                        union_counts[path] = counts[path]
                union_lines = sum(union_counts.values())
                if (len(union) <= budget.max_files and union_lines <= budget.max_lines
                        and len(union_counts) == len(union)):
                    runs[index] = (run_language, union, task_ids + [task.task_id])
                    run_counts[index] = union_counts
                    merged = True
                    break
        if not merged:
            runs.append((language, files, [task.task_id]))
            run_counts.append(_file_counts(source_slice))
            run_excluded.append(excluded)
    return runs


def _file_counts(source_slice):
    counts = dict(getattr(source_slice, "_file_lines", {}))
    missing = [path for path in source_slice.files if path not in counts]
    if len(missing) == 1 and len(source_slice.files) == 1:
        counts[missing[0]] = source_slice.total_lines
    elif missing:
        # Slice's public shape has only a total line count. For caller-created
        # slices without the builder's private per-file counts, use an even,
        # upward-rounded estimate so overlapping groups can still coalesce.
        estimate = max(1, (source_slice.total_lines + len(source_slice.files) - 1)
                       // max(1, len(source_slice.files)))
        for path in missing:
            counts[path] = estimate
    return counts


def materialize_slice(files, repo_root, dest_dir):
    """Copy selected repository files byte-for-byte into a separate directory."""
    try:
        root = Path(repo_root).resolve()
        destination = Path(dest_dir).resolve()
        if destination == root or root in destination.parents:
            return None, "destination_inside_repo"
        sources = []
        for value in files:
            relative = PurePosixPath(str(value).replace("\\", "/"))
            windows_relative = PureWindowsPath(str(value))
            if (relative.is_absolute() or windows_relative.is_absolute()
                    or (relative.parts and relative.parts[0].endswith(":"))
                    or ".." in relative.parts):
                return None, "path_escapes_repo"
            source = (root / Path(*relative.parts)).resolve()
            if source != root and root not in source.parents:
                return None, "path_escapes_repo"
            if not source.is_file():
                return None, "file_missing"
            target = (destination / Path(*relative.parts)).resolve()
            if target != destination and destination not in target.parents:
                return None, "path_escapes_repo"
            sources.append((source, target))
        for source, target in sources:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        return destination, None
    except (OSError, TypeError, ValueError):
        return None, "copy_error"
