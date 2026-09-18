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


# --------------------------------------------------- tree-sitter (real) ----
#
# For every non-Python language this used to be a brace-counting regex pass
# that only *played the role* of tree-sitter. This is the real thing: an
# actual concrete syntax tree from the `tree_sitter_languages` grammars
# (prebuilt `tree-sitter` parsers for JS/TS/TSX, Java, Go, Ruby, PHP, C#,
# Rust), walked structurally the same way `_parse_python` walks Python's
# `ast` tree above — real node types and real field names, not
# "does this line look like a function signature".
#
# The regex pass isn't deleted: it's now the graceful-degradation fallback
# for an environment where `tree_sitter_languages` isn't installed or a
# specific grammar fails to load, exactly like `llm_fallback.py` degrades
# to "needs manual review" when `transformers`/`torch` are missing rather
# than crashing the run.

# maps a file extension to the tree-sitter grammar name in
# tree_sitter_languages — finer-grained than LANG_BY_EXT because JSX/TSX
# need their own grammars to parse JSX syntax correctly (the plain
# "javascript" grammar already includes JSX; "typescript" does not, so
# .tsx gets the dedicated "tsx" grammar).
_TS_GRAMMAR_BY_EXT = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".rs": "rust",
}

# node types that represent a class-like definition, per grammar. Rust has
# no classes; structs/enums/traits stand in, matching the "struct as
# pseudo-class" treatment the old regex pass gave Go.
_TS_CLASS_TYPES = {
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "ruby": {"class", "module"},
    "php": {"class_declaration", "interface_declaration"},
    "c_sharp": {"class_declaration", "interface_declaration", "struct_declaration"},
    "rust": {"struct_item", "enum_item", "trait_item"},
}

# node types that represent a function/method definition, per grammar.
# Whether a given node counts as "function" or "method" is decided at walk
# time by whether we're currently inside a class/struct/impl scope, not by
# the node type alone (same rule the old regex pass used).
_TS_FUNC_TYPES = {
    "javascript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "tsx": {"function_declaration", "method_definition", "generator_function_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "ruby": {"method", "singleton_method"},
    "php": {"function_definition", "method_declaration"},
    "c_sharp": {"method_declaration", "constructor_declaration", "local_function_statement"},
    "rust": {"function_item"},
}

# node types that represent an import/include, per grammar, and how to
# read the module string back out of them. `None` means "no dedicated
# import node in this grammar" — Ruby's require/require_relative are
# ordinary call expressions instead, so those are handled in the call-site
# branch of the walk (see _handle_call below).
_TS_IMPORT_TYPES = {
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "java": {"import_declaration"},
    "go": {"import_spec"},
    "ruby": set(),
    "php": {"namespace_use_declaration"},
    "c_sharp": {"using_directive"},
    "rust": {"use_declaration"},
}

# (call_node_type, [callee_field_names_to_join_with_"."], arguments_field_name)
# — the field names tree-sitter attaches to a call's children differ per
# grammar (e.g. JS calls it "function", Java splits "object"/"name"), so
# each grammar gets its own small table rather than one shape fitting all.
_TS_CALL_SPECS = {
    "javascript": [("call_expression", ["function"], "arguments")],
    "typescript": [("call_expression", ["function"], "arguments")],
    "tsx": [("call_expression", ["function"], "arguments")],
    "java": [("method_invocation", ["object", "name"], "arguments")],
    "go": [("call_expression", ["function"], "arguments")],
    "ruby": [
        ("call", ["receiver", "method"], "arguments"),
        ("method_call", ["method"], "arguments"),
    ],
    "php": [
        ("function_call_expression", ["function"], "arguments"),
        ("member_call_expression", ["object", "name"], "arguments"),
        ("scoped_call_expression", ["scope", "name"], "arguments"),
    ],
    "c_sharp": [("invocation_expression", ["function"], "arguments")],
    "rust": [
        ("call_expression", ["function"], "arguments"),
        ("macro_invocation", ["macro"], "token_tree"),
    ],
}

# require()/require_relative() aren't a dedicated import node in every
# grammar's call table above — they're ordinary calls whose *name* happens
# to be one of these. When the walk sees a call to one of them with a
# single string-literal argument, it records an Import in addition to the
# CallSite (matching the old regex pass, which matched both patterns on
# the same line independently).
_REQUIRE_LIKE_CALLEES = {
    "javascript": {"require"},
    "typescript": {"require"},
    "tsx": {"require"},
    "ruby": {"require", "require_relative"},
    "php": {"require", "require_once", "include", "include_once"},
}

_BRACKET_PAIRS = {("(", ")"), ("{", "}"), ("[", "]")}

_ts_available = None          # True / False once probed, else None
_ts_parser_cache = {}          # grammar_name -> Parser | None
_ts_load_errors = {}           # grammar_name (or "_module") -> error string


def _get_ts_parser(grammar_name: str):
    """Lazily load one tree-sitter grammar's parser, memoized (both hits
    and misses) so a missing package or an unsupported grammar is only
    ever probed once per process, not once per file."""
    global _ts_available
    if grammar_name in _ts_parser_cache:
        return _ts_parser_cache[grammar_name]
    if _ts_available is False:
        return None
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        _ts_available = False
        _ts_load_errors["_module"] = (
            f"tree_sitter_languages not installed ({exc}); falling back to "
            "the regex structural pass for non-Python languages. "
            "`pip install tree_sitter_languages` to get real parse trees."
        )
        return None
    _ts_available = True
    try:
        parser = get_parser(grammar_name)
    except Exception as exc:  # noqa: BLE001 - a bad/missing grammar shouldn't crash the run
        parser = None
        _ts_load_errors[grammar_name] = f"{type(exc).__name__}: {exc}"
    _ts_parser_cache[grammar_name] = parser
    return parser


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "ignore")


def _field_text(node, source: bytes, field_name: str):
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return _node_text(child, source)


def _inner_arg_text(node, source: bytes) -> str:
    """Strip one layer of matching delimiters, e.g. "(url, 3)" -> "url, 3",
    mirroring the old regex pass's `m.group(2)` (raw text *inside* the
    parens) so downstream stages (literal_check etc.) see the same shape
    of string regardless of which parser produced it."""
    text = _node_text(node, source)
    if len(text) >= 2 and (text[0], text[-1]) in _BRACKET_PAIRS:
        return text[1:-1]
    return text


def _callee_text(node, source: bytes, field_names) -> str:
    parts = []
    for name in field_names:
        t = _field_text(node, source, name)
        if t:
            parts.append(t)
    return ".".join(parts) if parts else _node_text(node, source).split("(")[0].strip()


def _go_receiver_type(node, source: bytes):
    """Go methods are declared as `func (r *Foo) Bar(...)`; the receiver's
    *type* (not the receiver variable name) is what makes `Bar` belong to
    `Foo`. Best-effort: walk the receiver parameter list for a
    type_identifier, looking past a leading pointer `*`."""
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for child in receiver.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is None:
                continue
            if type_node.type == "pointer_type" and type_node.child_count:
                type_node = type_node.children[-1]
            if type_node.type in ("type_identifier", "identifier"):
                return _node_text(type_node, source)
    return None


def _parse_treesitter(rel_path: str, text: str, grammar_name: str, ctx=None) -> FileRecord:
    parser = _get_ts_parser(grammar_name)
    if parser is None:
        return None  # caller falls back to the regex pass

    rec = FileRecord(path=rel_path, language=grammar_name, source_lines=text.splitlines())
    source = text.encode("utf-8", "ignore")
    try:
        tree = parser.parse(source)
    except Exception as exc:  # noqa: BLE001 - a parser crash on one odd file shouldn't kill the run
        _bump(ctx, "parse.treesitter.parse_error")
        log(LOG, "warning", "tree-sitter failed to parse file; falling back to regex pass",
            file=rel_path, grammar=grammar_name, error=f"{type(exc).__name__}: {exc}")
        return None

    class_types = _TS_CLASS_TYPES.get(grammar_name, set())
    func_types = _TS_FUNC_TYPES.get(grammar_name, set())
    import_types = _TS_IMPORT_TYPES.get(grammar_name, set())
    call_specs = _TS_CALL_SPECS.get(grammar_name, [])
    require_like = _REQUIRE_LIKE_CALLEES.get(grammar_name, set())

    def qname(class_name, func_name):
        return f"{rel_path}::" + (f"{class_name}.{func_name}" if class_name else func_name)

    def handle_call(node, caller: str):
        for call_type, callee_fields, args_field in call_specs:
            if node.type != call_type:
                continue
            callee = _callee_text(node, source, callee_fields)
            args_node = node.child_by_field_name(args_field)
            arg_text = _inner_arg_text(args_node, source) if args_node is not None else ""
            rec.calls.append(CallSite(
                caller=caller, callee_expr=callee,
                line=node.start_point[0] + 1, arg_text=arg_text, file=rel_path,
            ))
            bare_name = callee.rsplit(".", 1)[-1]
            if bare_name in require_like and args_node is not None:
                first_str = next(
                    (c for c in args_node.children if c.type in ("string", "string_literal")),
                    None,
                )
                if first_str is not None:
                    raw = _node_text(first_str, source).strip("'\"")
                    rec.imports.append(Import(file=rel_path, raw=raw, line=node.start_point[0] + 1))
            return True
        return False

    def handle_import(node) -> bool:
        if node.type not in import_types:
            return False
        raw = None
        if grammar_name == "go" and node.type == "import_spec":
            path_node = node.child_by_field_name("path") or next(
                (c for c in node.children if c.type == "interpreted_string_literal"), None,
            )
            if path_node is not None:
                raw = _node_text(path_node, source).strip("\"")
        elif grammar_name in ("javascript", "typescript", "tsx"):
            src_node = node.child_by_field_name("source")
            if src_node is not None:
                raw = _node_text(src_node, source).strip("'\"")
        if raw is None:
            # java / php / c_sharp / rust: no single "source" field, so
            # fall back to the whole statement's text, trimmed of the
            # keyword and trailing punctuation — still gives downstream
            # cross-file resolution a real string to match against.
            raw = _node_text(node, source).strip()
            for kw in ("import ", "using ", "use ", "require "):
                if raw.startswith(kw):
                    raw = raw[len(kw):]
            raw = raw.rstrip(";").strip()
        rec.imports.append(Import(file=rel_path, raw=raw, line=node.start_point[0] + 1))
        return True

    def walk(node, caller: str, class_name):
        if handle_import(node):
            return  # don't descend into import statements looking for calls
        if handle_call(node, caller):
            # still descend — a call's arguments can contain nested calls,
            # e.g. `fetch(buildUrl(id))`
            for child in node.children:
                walk(child, caller, class_name)
            return

        if node.type in class_types:
            name = _field_text(node, source, "name") or "<anonymous>"
            rec.definitions.append(Definition(
                qualified_name=f"{rel_path}::{name}", name=name, kind="class",
                file=rel_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            ))
            for child in node.children:
                walk(child, caller, name)
            return

        if grammar_name == "rust" and node.type == "impl_item":
            type_node = node.child_by_field_name("type")
            impl_name = _node_text(type_node, source) if type_node is not None else class_name
            for child in node.children:
                walk(child, caller, impl_name)
            return

        if node.type in func_types:
            name = _field_text(node, source, "name") or "<anonymous>"
            if grammar_name == "go" and node.type == "method_declaration":
                receiver_type = _go_receiver_type(node, source)
                qn = qname(receiver_type, name)
                kind = "method"
            else:
                kind = "method" if class_name else "function"
                qn = qname(class_name, name)
            rec.definitions.append(Definition(
                qualified_name=qn, name=name, kind=kind,
                file=rel_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            ))
            for child in node.children:
                walk(child, qn, class_name)
            return

        # go's `type Foo struct { ... }` is the "struct as pseudo-class"
        # case: there's no dedicated class node type, so catch it via the
        # type_spec wrapper and treat it exactly like a class definition.
        if grammar_name == "go" and node.type == "type_spec":
            type_node = node.child_by_field_name("type")
            if type_node is not None and type_node.type == "struct_type":
                name = _field_text(node, source, "name") or "<anonymous>"
                rec.definitions.append(Definition(
                    qualified_name=f"{rel_path}::{name}", name=name, kind="class",
                    file=rel_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
                ))
                return  # struct bodies are field lists, nothing to walk

        for child in node.children:
            walk(child, caller, class_name)

    module_scope = f"{rel_path}::<module>"
    walk(tree.root_node, module_scope, None)
    return rec


# ------------------------------------------------------ generic (regex) ----
# Fallback path for when tree_sitter_languages isn't installed, or a given
# grammar fails to load — see `_parse_generic` below, which tries the real
# tree-sitter pass first and only reaches this on a `None` result.

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


def _parse_regex_fallback(rel_path: str, text: str, language: str, ctx=None) -> FileRecord:
    """Brace/indent-based regex structural pass — the language-agnostic
    stand-in tree-sitter's docstring at the top of this file describes.
    Only reached when `_parse_treesitter` returns `None` (grammar missing,
    package not installed, or the parse itself raised)."""
    _bump(ctx, "parse.treesitter.fallback_used")
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


def _parse_generic(rel_path: str, text: str, language: str, ext: str, ctx=None) -> FileRecord:
    """Dispatcher for every non-Python language: try a real tree-sitter
    parse first, and only drop to the regex fallback if that couldn't
    happen at all (package missing, grammar unavailable, or the parse
    itself raised) — `_parse_treesitter` returns `None` in exactly those
    cases rather than a half-filled record, so there's no silent mixing
    of a partial tree-sitter result with regex output for the same file.
    """
    grammar_name = _TS_GRAMMAR_BY_EXT.get(ext)
    if grammar_name:
        rec = _parse_treesitter(rel_path, text, grammar_name, ctx=ctx)
        if rec is not None:
            _bump(ctx, "parse.treesitter.used")
            return rec
    return _parse_regex_fallback(rel_path, text, language, ctx=ctx)


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
        return _parse_generic(rel_path, text, lang, ext, ctx=ctx)
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
        fell_back = ctx.count("parse.treesitter.fallback_used")
        used_ts = ctx.count("parse.treesitter.used")
        if fell_back:
            reason = _ts_load_errors.get("_module") or next(iter(_ts_load_errors.values()), None)
            log(LOG, "warning", "tree-sitter unavailable for some/all non-Python files; used regex fallback",
                fallback_files=fell_back, treesitter_files=used_ts, reason=reason)
            ctx.note("warning", "parse",
                     f"{fell_back} non-Python file(s) were parsed with the regex structural "
                     "pass instead of tree-sitter"
                     + (f" ({reason})" if reason else "")
                     + ". Definitions/calls for those files are a best-effort approximation.")

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