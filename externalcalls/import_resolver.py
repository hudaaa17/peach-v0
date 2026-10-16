"""Resolve source imports to files in a repository without reading dependencies."""

import ast
import posixpath
from pathlib import Path, PurePosixPath


_JS_EXTENSIONS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


def _normal_path(value):
    return str(value).replace("\\", "/").lstrip("./")


def _python_ast_specs(node):
    """Return (raw spelling, module spelling, relative level) for an AST import."""
    if isinstance(node, ast.Import):
        return [(alias.name, alias.name, 0) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        result = []
        for alias in node.names:
            suffix = f".{alias.name}" if module else alias.name
            spec = ("." * node.level) + module + suffix
            result.append((spec.lstrip("."), spec, node.level))
        return result
    return []


def _python_specs(importing_file, import_spec, repo_root):
    raw = getattr(import_spec, "raw", import_spec)
    line = getattr(import_spec, "line", None)
    if not isinstance(raw, str):
        return None, "alias_unresolved"

    source_path = None
    if repo_root is not None:
        source_path = Path(repo_root) / Path(str(importing_file).replace("\\", "/"))
        try:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, TypeError, ValueError):
            text = None
        if text is not None:
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError, TypeError):
                return None, "alias_unresolved"
            nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                     and (line is None or node.lineno == line)]
            choices = [choice for node in nodes for choice in _python_ast_specs(node)]
            if choices:
                matched = [choice for choice in choices if choice[0] == raw]
                if not matched:
                    matched = [choice for choice in choices if choice[0].endswith(raw)]
                if len(matched) == 1:
                    return [(matched[0][1], matched[0][2])], None
                if len(matched) > 1:
                    return [(item[1], item[2]) for item in matched], None
                if len(choices) == 1:
                    return [(choices[0][1], choices[0][2])], None
                return None, "alias_unresolved"

    # A parser Import object can have lost the leading relative dots. Without
    # the source file there is no safe way to reconstruct that information.
    if line is not None and str(raw) and not str(raw).startswith("."):
        return [(raw, 0)], None
    if str(raw).startswith("."):
        return [(raw, len(raw) - len(raw.lstrip(".")))], None
    return [(raw, 0)], None


def import_sort_name(importing_file, import_spec, repo_root=None):
    """Return the local Python binding name when the AST can identify it."""
    raw = getattr(import_spec, "raw", import_spec)
    line = getattr(import_spec, "line", None)
    if repo_root is not None and line is not None:
        path = Path(repo_root) / str(importing_file).replace("\\", "/")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError, TypeError):
            return str(raw).rsplit(".", 1)[-1]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.lineno != line:
                continue
            choices = []
            for alias in node.names:
                if isinstance(node, ast.Import):
                    choices.append((alias.name, alias.asname or alias.name.split(".")[0]))
                else:
                    module = node.module or ""
                    spelling = f"{module}.{alias.name}" if module else alias.name
                    choices.append((spelling, alias.asname or alias.name))
            matched = [name for spelling, name in choices
                       if spelling == raw or spelling.endswith(str(raw))]
            if len(matched) == 1:
                return matched[0]
    return str(raw).rsplit(".", 1)[-1]


def _python_candidates(spec, level):
    text = spec
    if level:
        text = text[level:]
    return [part for part in text.split(".") if part]


def _python_module_path(parts, roots, repo_files):
    for root_prefix in roots:
        for stop in range(len(parts), 0, -1):
            stem_parts = ([root_prefix] if root_prefix else []) + parts[:stop]
            stem = "/".join(stem_parts)
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if candidate in repo_files:
                    return candidate
    return None


def _python_resolve(importing_file, spec, level, repo_files):
    parts = _python_candidates(spec, level)
    if not parts:
        return None, "unresolved"
    roots = [""]
    if level:
        importer = PurePosixPath(_normal_path(importing_file))
        base = list(importer.parent.parts)
        for _ in range(max(0, level - 1)):
            if not base:
                return None, "unresolved"
            base.pop()
        prefix = "/".join(base)
        for stop in range(len(parts), 0, -1):
            stem = "/".join(([prefix] if prefix else []) + parts[:stop])
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if candidate in repo_files:
                    return candidate, None
        return None, "unresolved"

    candidate = _python_module_path(parts, roots, repo_files)
    if candidate:
        return candidate, None
    candidate = _python_module_path(parts, ["src"], repo_files)
    if candidate:
        return candidate, None
    top = parts[0]
    found_top = any(path == f"{top}.py" or path.startswith(f"{top}/")
                    or path == f"src/{top}.py" or path.startswith(f"src/{top}/")
                    for path in repo_files)
    return (None, "unresolved") if found_top else (None, "external")


def _js_candidates(import_spec, repo_files):
    path = import_spec.replace("\\", "/")
    exact = [path]
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".js":
        stem = path[:-3]
        exact.extend(stem + ext for ext in (".ts", ".tsx"))
    elif not suffix:
        exact.extend(path + ext for ext in _JS_EXTENSIONS)
    exact.extend(path.rstrip("/") + "/index" + ext for ext in _JS_EXTENSIONS)
    return next((candidate for candidate in exact if candidate in repo_files), None)


def resolve_import(importing_file, import_spec, repo_files, language, repo_root=None):
    """Resolve an import to a repo-relative path, or return its classification.

    ``import_spec`` may be a parser ``Import`` record or a module string. The
    optional ``repo_root`` lets Python imports recover relative dots and aliases
    from the original AST line, which the parser's raw string can omit.
    """
    try:
        files = {_normal_path(item) for item in repo_files}
        lang = str(language).lower()
        if lang in ("python", "py"):
            specs, error = _python_specs(importing_file, import_spec, repo_root)
            if error:
                return None, error
            results = [_python_resolve(importing_file, spec, level, files)
                       for spec, level in specs]
            for target, reason in results:
                if target:
                    return target, None
            if results and all(reason == "external" for _, reason in results):
                return None, "external"
            return None, "unresolved"

        if lang in ("javascript", "typescript", "js", "ts"):
            spec = getattr(import_spec, "raw", import_spec)
            if not isinstance(spec, str):
                return None, "unresolved"
            if spec.startswith(("@/", "~/")):
                return None, "alias_unresolved"
            if spec.startswith(("./", "../")):
                importer = PurePosixPath(_normal_path(importing_file))
                path = posixpath.normpath(posixpath.join(str(importer.parent), spec))
                target = _js_candidates(path, files)
                return (target, None) if target else (None, "unresolved")
            first = spec.split("/", 1)[0]
            if first.startswith("@"):
                return None, "external"
            if any(path.startswith(first + "/") for path in files):
                return None, "alias_unresolved"
            return None, "external"
        return None, "unresolved"
    except (OSError, TypeError, ValueError, AttributeError):
        return None, "unresolved"
