"""
Stage 1: Tree-sitter parse — raw structural pass over each source file.

For Python we use the real `ast` module (a proper concrete syntax tree).
For other common languages (JS/TS/Java/Go/etc.) we use a light structural
regex pass that plays the same role tree-sitter plays in the diagram:
fast, language-agnostic extraction of functions/classes/imports/calls
without compiling or executing anything.
"""
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("parser")

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".next", "coverage",
}

LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".rs": "rust",
}

CONFIG_EXTS = {".yml", ".yaml", ".json", ".env", ".toml", ".ini", ".properties"}
CONFIG_NAME_HINTS = ("docker-compose", "k8s", "deployment", "service", "configmap", "kustomiz", "helm")


@dataclass
class Definition:
    qualified_name: str
    name: str
    kind: str          # "function" | "method" | "class"
    file: str
    start_line: int
    end_line: int


@dataclass
class CallSite:
    caller: str         # qualified name of enclosing function, or "<module>:<file>"
    callee_expr: str     # raw text of the call target, e.g. "requests.get" or "self.helper"
    line: int
    arg_text: str        # raw text inside the parentheses (best effort)
    file: str


@dataclass
class Import:
    file: str
    raw: str
    line: int


@dataclass
class FileRecord:
    path: str            # relative path
    language: str
    source_lines: list = field(default_factory=list)
    imports: list = field(default_factory=list)      # list[Import]
    definitions: list = field(default_factory=list)  # list[Definition]
    calls: list = field(default_factory=list)        # list[CallSite]
    is_config: bool = False


def discover_files(repo_root: Path, max_files: int = 400, ctx=None):
    """Walk the repo, applying the skip rules. Every skip is counted with
    a reason so a surprisingly small `files_parsed` can be explained
    without re-running the walk by hand."""
    files = []
    truncated = False
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            _bump(ctx, "parse.skipped.vendored_dir")
            continue
        try:
            if p.stat().st_size > 400_000:
                _bump(ctx, "parse.skipped.too_large")
                if TRACE_EDGES:
                    log(LOG, "debug", "skip: over size cap", path=str(p))
                continue
        except OSError as exc:
            _bump(ctx, "parse.skipped.stat_error")
            log(LOG, "warning", "could not stat file", path=str(p), error=str(exc))
            continue
        files.append(p)
        if len(files) >= max_files:
            truncated = True
            break

    _bump(ctx, "parse.files_discovered", len(files))
    if truncated:
        log(LOG, "warning", "file cap reached; repo truncated",
            cap=max_files, note="raise max_files if the graph looks incomplete")
        if ctx:
            ctx.note("warning", "parse",
                     f"Hit the {max_files}-file cap; some of the repo was not analyzed.")
    return files


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)


def _looks_like_config(rel_path: str) -> bool:
    ext = Path(rel_path).suffix.lower()
    name = Path(rel_path).name.lower()
    if ext in CONFIG_EXTS:
        return True
    if name in ("dockerfile",):
        return True
    if any(h in rel_path.lower() for h in CONFIG_NAME_HINTS):
        return True
    return False


# ---------------------------------------------------------------- Python ----

def _parse_python(rel_path: str, text: str, ctx=None) -> FileRecord:
    rec = FileRecord(path=rel_path, language="python", source_lines=text.splitlines())
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        _bump(ctx, "parse.python.syntax_error")
        log(LOG, "warning", "python file failed to parse; skipped",
            file=rel_path, line=getattr(exc, "lineno", 0), error=exc.msg)
        if ctx:
            ctx.note("warning", "parse",
                     f"{rel_path} could not be parsed (line {getattr(exc, 'lineno', 0)}): "
                     f"{exc.msg}. Its calls are missing from the graph.")
        return rec

    def qname(*parts):
        return f"{rel_path}::" + ".".join(p for p in parts if p)

    class CallVisitor(ast.NodeVisitor):
        """Collects calls made *inside* a given enclosing scope only."""
        def __init__(self, caller_qname):
            self.caller = caller_qname

        def visit_Call(self, node: ast.Call):
            expr = _dotted_name(node.func)
            arg_text = ""
            if node.args:
                arg_text = ", ".join(_arg_repr(a) for a in node.args)
            rec.calls.append(CallSite(
                caller=self.caller, callee_expr=expr,
                line=getattr(node, "lineno", 0), arg_text=arg_text, file=rel_path,
            ))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            pass  # don't descend into nested defs; they get their own visitor

        def visit_AsyncFunctionDef(self, node):
            pass

    def _dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _dotted_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return "<expr>"

    def _arg_repr(node):
        """Source text for one call argument.

        This used to return the placeholder "<expr>" for anything that
        wasn't a bare constant/name/attribute — which silently threw away
        the argument of every f-string, concatenation, `.format()` and
        nested call. Since literal_check, constprop_check and *all four*
        Joern traces key off this text, an "<expr>" argument is
        unresolvable by construction: it cannot match a URL, it is not a
        bare identifier, and no assignment line will ever be named
        "<expr>". Those calls were guaranteed to fall through every
        deterministic stage and land on the SLM.

        `ast.unparse` (3.9+) round-trips the real expression instead, so
        downstream stages see `f"https://api.example.com/v1/{key}"` rather
        than a placeholder. The placeholder is kept only as a last resort
        and is counted, so degradation stays visible instead of silent."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        # NB: Attribute deliberately falls through to unparse. Routing it
        # to _dotted_name() reintroduced the same bug one level down —
        # `thing.resolve().url` came back as "<expr>.url", because
        # _dotted_name bails to the placeholder on a non-name base.
        try:
            text = ast.unparse(node)
        except Exception as exc:  # noqa: BLE001 - unparse is best-effort
            _bump(ctx, "parse.arg_repr.placeholder")
            if TRACE_EDGES:
                log(LOG, "debug", "ast.unparse failed for argument",
                    file=rel_path, line=getattr(node, "lineno", 0),
                    node_type=type(node).__name__, error=str(exc))
            return "<expr>"
        _bump(ctx, "parse.arg_repr.unparsed")
        # Guard against pathological one-liners blowing up log lines and
        # regex scans downstream; 400 chars is far past any real URL.
        if len(text) > 400:
            _bump(ctx, "parse.arg_repr.truncated")
            return text[:400]
        return text

    def walk(node, class_name=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    rec.imports.append(Import(file=rel_path, raw=alias.name, line=child.lineno))
            elif isinstance(child, ast.ImportFrom):
                mod = child.module or ""
                for alias in child.names:
                    rec.imports.append(Import(file=rel_path, raw=f"{mod}.{alias.name}" if mod else alias.name, line=child.lineno))
            elif isinstance(child, ast.ClassDef):
                cdef = Definition(
                    qualified_name=qname(child.name), name=child.name, kind="class",
                    file=rel_path, start_line=child.lineno, end_line=getattr(child, "end_lineno", child.lineno),
                )
                rec.definitions.append(cdef)
                walk(child, class_name=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if class_name else "function"
                qn = qname(class_name, child.name) if class_name else qname(child.name)
                fdef = Definition(
                    qualified_name=qn, name=child.name, kind=kind,
                    file=rel_path, start_line=child.lineno, end_line=getattr(child, "end_lineno", child.lineno),
                )
                rec.definitions.append(fdef)
                cv = CallVisitor(qn)
                for stmt in ast.iter_child_nodes(child):
                    cv.visit(stmt)
                walk(child, class_name=class_name)  # nested funcs/classes
            else:
                walk(child, class_name=class_name)

    # module-level calls (outside any function)
    module_scope = f"{rel_path}::<module>"
    top_cv = CallVisitor(module_scope)
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        top_cv.visit(stmt)

    walk(tree)
    return rec


# ------------------------------------------------------ generic (regex) ----

_GENERIC_IMPORT_PATTERNS = [
    re.compile(r'^\s*import\s+.*?[\'"](.+?)[\'"]'),                 # es modules
    re.compile(r'^\s*import\s+\{[^}]*\}\s+from\s+[\'"](.+?)[\'"]'),
    re.compile(r'require\([\'"](.+?)[\'"]\)'),                       # commonjs
    re.compile(r'^\s*import\s+"(.+?)"'),                             # go
    re.compile(r'^\s*import\s+[\w.]+;'),                             # java (kept generic below)
    re.compile(r'^\s*using\s+([\w.]+);'),                            # C#
]

_GENERIC_FUNC_PATTERNS = [
    # JS/TS function decl, method, arrow assigned to const
    re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\('),
    re.compile(r'^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\('),
    re.compile(r'^\s*(?:public|private|protected|static|\s)*[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{'),  # java/c#-ish method
    # Go
    re.compile(r'^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\('),
    # Ruby
    re.compile(r'^\s*def\s+([A-Za-z_]\w*[?!]?)'),
]

_GENERIC_CLASS_PATTERNS = [
    re.compile(r'^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)'),
    re.compile(r'^\s*(?:public\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)'),
    re.compile(r'^\s*type\s+([A-Za-z_]\w*)\s+struct\b'),  # go struct as pseudo-class
    re.compile(r'^\s*module\s+([A-Za-z_]\w*)'),
]

_GENERIC_CALL_PATTERN = re.compile(r'([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(([^()]*)\)')

_GENERIC_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "function", "return", "new",
    "typeof", "instanceof", "await", "async", "class", "const", "let", "var",
    "def", "func", "public", "private", "static", "void", "int", "string",
}


def _parse_generic(rel_path: str, text: str, language: str, ctx=None) -> FileRecord:
    rec = FileRecord(path=rel_path, language=language, source_lines=text.splitlines())
    lines = rec.source_lines

    # imports
    for i, line in enumerate(lines, start=1):
        for pat in _GENERIC_IMPORT_PATTERNS:
            m = pat.search(line)
            if m:
                raw = m.group(1) if m.groups() else line.strip()
                rec.imports.append(Import(file=rel_path, raw=raw, line=i))
                break

    # find definitions with a naive brace/indent based body-end estimate
    open_defs = []  # stack of (Definition, brace_depth_at_open) for brace langs
    brace_depth = 0
    current_class = None
    class_stack = []

    def try_match_def(line, i):
        for pat in _GENERIC_CLASS_PATTERNS:
            m = pat.search(line)
            if m:
                return "class", m.group(1)
        for pat in _GENERIC_FUNC_PATTERNS:
            m = pat.search(line)
            if m:
                return "function", m.group(1)
        return None, None

    # naive scope tracking for "which function is this call inside"
    scope_stack = [f"{rel_path}::<module>"]
    scope_close_depth = []

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        kind, name = try_match_def(line, i)
        if name:
            qn = f"{rel_path}::{class_stack[-1] + '.' if class_stack else ''}{name}"
            def_kind = "method" if (kind == "function" and class_stack) else kind
            rec.definitions.append(Definition(
                qualified_name=qn, name=name, kind=def_kind,
                file=rel_path, start_line=i, end_line=i,
            ))
            if kind == "class":
                class_stack.append(name)
                scope_close_depth.append(brace_depth + line.count("{"))
            else:
                scope_stack.append(qn)
                scope_close_depth.append(brace_depth + line.count("{"))

        brace_depth += line.count("{") - line.count("}")

        # pop scopes whose opening brace depth has been closed
        while scope_close_depth and brace_depth < scope_close_depth[-1] and line.count("}") > 0:
            scope_close_depth.pop()
            if len(scope_stack) > 1:
                scope_stack.pop()
            elif class_stack:
                class_stack.pop()

        # calls on this line
        for m in _GENERIC_CALL_PATTERN.finditer(line):
            callee = m.group(1)
            head = callee.split(".")[0]
            if head in _GENERIC_KEYWORDS or kind:  # skip the def line itself
                continue
            rec.calls.append(CallSite(
                caller=scope_stack[-1], callee_expr=callee, line=i,
                arg_text=m.group(2), file=rel_path,
            ))

    return rec


def parse_file(repo_root: Path, path: Path, ctx=None) -> FileRecord:
    # Normalise to POSIX separators at the one place a relative path is
    # created. These strings are used as graph node ids, as dict keys in
    # records_by_path, and — critically — are split on "/" by the import
    # resolution in joern_check and pipeline. On Windows, str(PurePath)
    # yields backslashes, so `"auth\\auth_functions.py".rsplit("/", 1)[-1]`
    # returns the whole path and never matches an import tail like
    # "auth_functions". That silently killed the cross-file
    # imported-constant trace on Windows while leaving it working on
    # POSIX, which is exactly the kind of bug that only shows up in
    # someone else's log.
    rel_path = path.relative_to(repo_root).as_posix()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        _bump(ctx, "parse.read_error")
        log(LOG, "warning", "could not read file", file=rel_path, error=str(exc))
        return FileRecord(path=rel_path, language="unknown", is_config=_looks_like_config(rel_path))

    if _looks_like_config(rel_path):
        _bump(ctx, "parse.config_files")
        return FileRecord(path=rel_path, language="config", source_lines=text.splitlines(), is_config=True)

    ext = path.suffix.lower()
    lang = LANG_BY_EXT.get(ext)
    if lang == "python":
        return _parse_python(rel_path, text, ctx=ctx)
    if lang:
        return _parse_generic(rel_path, text, lang, ctx=ctx)
    _bump(ctx, "parse.language.unsupported")
    return FileRecord(path=rel_path, language="other", source_lines=text.splitlines())


def parse_repo(repo_root: Path, ctx=None):
    files = discover_files(repo_root, ctx=ctx)
    records = []
    for f in files:
        try:
            rec = parse_file(repo_root, f, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the run
            _bump(ctx, "parse.file_exception")
            log(LOG, "error", "parser raised on file", file=str(f),
                error=f"{type(exc).__name__}: {exc}")
            continue
        records.append(rec)
        _bump(ctx, f"parse.language.{rec.language}")
        _bump(ctx, "parse.definitions", len(rec.definitions))
        _bump(ctx, "parse.calls", len(rec.calls))
        _bump(ctx, "parse.imports", len(rec.imports))
        if TRACE_EDGES:
            log(LOG, "debug", "parsed file", file=rec.path, language=rec.language,
                defs=len(rec.definitions), calls=len(rec.calls), imports=len(rec.imports))

    if ctx is not None:
        opaque = ctx.count("parse.arg_repr.placeholder")
        total_calls = ctx.count("parse.calls")
        if opaque:
            # This is the signal that used to be invisible. An opaque
            # argument cannot be resolved by *any* downstream stage, so a
            # nonzero count here caps how well the literal / const-prop /
            # Joern stages can possibly do.
            log(LOG, "warning", "some call arguments could not be recovered as source",
                opaque=opaque, total_calls=total_calls)
            ctx.note("warning", "parse",
                     f"{opaque} call argument(s) could not be recovered as source text. "
                     "Those calls cannot be resolved by the literal, const-prop or "
                     "Joern stages and will fall through to the SLM fallback.")
        log(LOG, "info", "parse complete",
            files=len(records),
            code_files=sum(1 for r in records if not r.is_config
                           and r.language not in ("unknown", "other")),
            config_files=sum(1 for r in records if r.is_config),
            definitions=ctx.count("parse.definitions"),
            calls=total_calls)
    return records
