"""
Stage 1: Tree-sitter parse — structural pass over each source file.

Python, JavaScript (incl. JSX) and TypeScript (incl. TSX) are parsed with
Tree-sitter and nothing else. Definitions, calls and imports are recognised
from the *shape* of the parse tree (which named fields a node carries), not
from lists of node-type names and not from anything specific to the code
being analysed. For example:

    * a node with a `body` field is a scope; with a `parameters` field it is a
      function-like scope, without one it is a class-like scope
    * a node with an `arguments` field and a `function`/`constructor` field
      is a call
    * a node with an `import` keyword token (or a `source` field) is an import

The same rules therefore cover `function_definition`, `function_declaration`,
`method_definition`, `arrow_function`, `class_definition`,
`interface_declaration`, ... without any of those names appearing below.

A deliberately small line-based fallback exists for the only two cases
Tree-sitter cannot serve: the grammar package is not installed / the grammar
fails to load, or the parser/extractor raises on a file. Syntax errors are NOT
one of those cases — Tree-sitter is error-tolerant, so a file with a typo still
yields a (partial) tree and is parsed normally.

Files that are deliberately NOT parsed (over the size limit, minified,
unreadable) are not dropped silently: each one is returned as a `FileRecord`
with `skipped=True` plus a `skip_reason`, so later stages and the Peach UI can
show exactly what was left out and why. A file whose parse raised is also
returned as a skipped record (`skip_reason="parse_exception"`), never dropped.

Everything that was NOT analysed is additionally reported as structured `Gap`
entries (see gaps.py): skipped files, files parsed only by the line-based
fallback (no call sites), files with syntax errors, and two repo-level entries
for what cannot be listed one by one - vendored directories (`SKIP_DIRS`) and
files beyond the `max_files` cap.

Public API: `Definition`, `CallSite`, `Import`, `FileRecord`,
`discover_files`, `parse_file`, `parse_repo`. All additions over the
prototype's signatures are optional keyword arguments / defaulted fields.
"""
from dataclasses import dataclass, field
from pathlib import Path

from .gaps import Gap, emit, gap_sink
from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("parser")

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".next", "coverage",
}

# Logical language reported on each FileRecord.
LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
}

# Tree-sitter grammar to load per extension. Finer-grained than LANG_BY_EXT:
# the "javascript" grammar already understands JSX, but "typescript" does not,
# so `.tsx` needs its own grammar. Supporting another language is a matter of
# adding its extension to both tables (the extraction below is grammar-agnostic).
GRAMMAR_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

CONFIG_EXTS = {".yml", ".yaml", ".json", ".env", ".toml", ".ini", ".properties"}
CONFIG_NAME_HINTS = ("docker-compose", "k8s", "deployment", "service", "configmap", "kustomiz", "helm")

# Call arguments longer than this are truncated: keeps log lines and
# downstream scans bounded; far past the length of any real URL/literal.
MAX_ARG_CHARS = 400

# Per-file size limits in bytes (overridable via discover_files / parse_repo).
# Measured cost of parsing (Python-style source): ~0.5 s and ~75 MB at 0.4 MB,
# ~7 s and ~750 MB at 5 MB, and memory grows ~140x the file size because every
# call/definition becomes a retained Python object. A 20 MB file was OOM-killed,
# which cannot be caught in-process, so a bound is required; 2 MB for source is
# comfortably inside the safe range. Non-source files (config, data) keep the
# prototype's tighter 400 KB limit.
MAX_SOURCE_BYTES = 2_000_000
MAX_OTHER_BYTES = 400_000

# Minified / bundled code is detected by SHAPE (very long lines), not by name or
# size alone: the file must be at least MINIFIED_MIN_BYTES and average more than
# MINIFIED_AVG_LINE_CHARS characters per line. The size floor keeps a small
# hand-written file with one long line (an inline data array, say) parseable.
MINIFIED_MIN_BYTES = 50_000
MINIFIED_AVG_LINE_CHARS = 300

# ----------------------------------------------------------------------------
# Tree-sitter *field* vocabulary.
#
# Field names are the grammars' own schema for "what role does this child
# play" and are shared across the Python / JS / TS grammars. They are the only
# grammar knowledge this module relies on — no named node types are listed.
# ----------------------------------------------------------------------------
_F_NAME = "name"                          # the identifier a definition declares
_F_BODY = "body"                          # present on every scope (function, class, loop, ...)
_F_PARAMS = ("parameters", "parameter")   # present on function-like scopes (incl. `x => x`)
_F_CALLEE = ("function", "constructor")   # what a call / `new` expression invokes
_F_ARGS = "arguments"                     # the argument list of a call
_F_MODULE = ("source", "module_name")     # the module an import/re-export pulls from
_F_BIND_TARGET = ("name", "key", "left", "property")  # `const NAME = ...`, `{ KEY: ... }`, `LEFT = ...`, class field `PROP = ...`
_F_BIND_VALUE = ("value", "right")        # ... the value side of those bindings
_F_MEMBER_BASE = "object"                 # `OBJECT.attr`
_F_MEMBER_ATTR = ("property", "attribute")  # `object.ATTR`

# Anonymous keyword tokens (a token's `.type` is its literal text).
_IMPORT_KEYWORD = "import"
_DOT_TOKEN = "."                          # distinguishes `import.meta` from an import

# Calls that load a module at runtime. They are ordinary calls to the grammar,
# so besides the CallSite they also yield an Import when given a string literal.
_MODULE_LOADER_CALLEES = {"require", "import"}

_QUOTES = "'\"`"
_OPAQUE = "<expr>"   # callee we cannot express as a dotted name, e.g. `f()()`


@dataclass
class Definition:
    qualified_name: str
    name: str
    kind: str          # "function" | "method" | "class"
    file: str
    start_line: int
    end_line: int
    name_line: int = 0   # 1-based line of the name identifier (SCIP matches on this)
    name_col: int = -1   # 0-based column of the name identifier; -1 = unknown


@dataclass
class CallSite:
    caller: str         # qualified name of enclosing function, or "<module>:<file>"
    callee_expr: str     # raw text of the call target, e.g. "requests.get" or "self.helper"
    line: int
    arg_text: str        # raw text inside the parentheses (best effort)
    file: str
    callee_line: int = 0   # 1-based line of the callee's final identifier
    callee_col: int = -1   # 0-based column of that identifier; -1 = unknown


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
    # Set when the file was deliberately not parsed (see module docstring).
    # `language` / `is_config` still describe what the file is; the lists above
    # are empty. skip_reason is one of: "too_large", "minified", "stat_error",
    # "read_error", "parse_exception". skip_detail is a short human-readable explanation for the UI.
    skipped: bool = False
    skip_reason: str = ""
    skip_detail: str = ""
    size_bytes: int = 0
    # How a (non-skipped) source file was parsed: "treesitter" = full parse,
    # "fallback" = minimal line-based parse (definitions/imports only, NO call
    # sites), "" = not a parsed source file (config, other, skipped).
    # A "fallback" record is NOT "analysed, no calls found" - it is incomplete.
    parse_mode: str = ""
    has_syntax_errors: bool = False   # tree-sitter recovered from syntax errors; extraction may be partial


def discover_files(repo_root: Path, max_files: int = 400, ctx=None, skipped=None,
                   max_source_bytes: int = MAX_SOURCE_BYTES,
                   max_other_bytes: int = MAX_OTHER_BYTES, gaps=None):
    """Walk the repo, applying the skip rules. Every skip is counted with
    a reason so a surprisingly small `files_parsed` can be explained
    without re-running the walk by hand.

    Returns the list of files to parse (unchanged contract). Files skipped for
    size or because they could not be stat'ed are, if a `skipped` list is
    passed in, appended to it as FileRecord(skipped=True, ...) so the caller
    can pass them ahead; each is also always logged with its path.
    Vendored directories are counted only, never listed (there can be
    thousands) but are summarised by one repo-level Gap. Source files
    (LANG_BY_EXT) get `max_source_bytes`; everything else gets the tighter
    `max_other_bytes`.

    Once `max_files` is reached the walk continues only to COUNT what is left,
    so the cap is reported as "N further files were not examined" instead of
    silently cutting the repo short. Directory names are tested on the path
    RELATIVE to `repo_root`, so a repo that happens to live under a folder
    called `build` or `vendor` is not skipped wholesale."""
    sink = gap_sink(ctx, gaps)
    files = []
    vendored = {}             # skip-dir name -> number of files ignored inside it
    over_cap = 0
    over_cap_samples = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(repo_root).parts
        hit = next((part for part in rel_parts if part in SKIP_DIRS), None)
        if hit is not None:
            _bump(ctx, "parse.skipped.vendored_dir")
            vendored[hit] = vendored.get(hit, 0) + 1
            continue
        if len(files) >= max_files:
            over_cap += 1
            if len(over_cap_samples) < 10:
                over_cap_samples.append("/".join(rel_parts))
            continue
        try:
            size = p.stat().st_size
        except OSError as exc:
            _record_skip(skipped, repo_root, p, "stat_error", 0, str(exc), ctx)
            continue
        is_source = p.suffix.lower() in LANG_BY_EXT
        limit = max_source_bytes if is_source else max_other_bytes
        if size > limit:
            kind = "source" if is_source else "non-source"
            _record_skip(skipped, repo_root, p, "too_large", size,
                         f"{size:,} bytes exceeds the {limit:,}-byte limit for {kind} files", ctx)
            continue
        files.append(p)

    _bump(ctx, "parse.files_discovered", len(files))
    if vendored:
        total = sum(vendored.values())
        top = sorted(vendored.items(), key=lambda kv: -kv[1])
        emit(sink, Gap(
            stage="parse", scope="repo", kind="vendored_dir", count=total,
            samples=[name for name, _ in top[:10]],
            detail=(f"{total:,} file(s) inside vendored/build directories were ignored by design: "
                    + ", ".join(f"{name} ({n:,})" for name, n in top[:5])
                    + ("" if len(top) <= 5 else f", and {len(top) - 5} more")),
            impact=("Code in these directories is not part of the graph. Calls into it are not "
                    "resolved to it and may be reported as external.")))
    if over_cap:
        log(LOG, "warning", "file cap reached; repo truncated",
            cap=max_files, not_examined=over_cap,
            note="raise max_files if the graph looks incomplete")
        if ctx:
            ctx.note("warning", "parse",
                     f"Hit the {max_files}-file cap; {over_cap:,} further file(s) were not analyzed.")
        emit(sink, Gap(
            stage="parse", scope="repo", kind="file_cap", count=over_cap, samples=over_cap_samples,
            detail=f"The {max_files}-file cap was reached; {over_cap:,} further file(s) were not examined.",
            impact=("Those files are missing from the graph entirely: their definitions and calls "
                    "do not exist as far as later stages are concerned.")))
    return files


def _bump(ctx, key, n=1):
    if ctx is not None:
        ctx.bump(key, n)


def _looks_like_config(rel_path: str) -> bool:
    """Decide whether a file is a config file (read as raw text by the config
    check stage) rather than source code (parsed by Tree-sitter).

    Order matters:
      1. Config extensions and Dockerfiles are always config.
      2. A file with a supported source extension is NEVER config, no matter
         what its path contains. The name hints below are substring matches
         on the whole path, so without this guard `services/user.py` or
         `deployment_utils.ts` would be mislabelled as config and silently
         lose all of their definitions, calls and imports.
      3. Only for everything else (extensionless or unknown extensions, e.g.
         `k8s/deployment`) do the name hints apply.
    """
    ext = Path(rel_path).suffix.lower()
    name = Path(rel_path).name.lower()
    if ext in CONFIG_EXTS:
        return True
    if name in ("dockerfile",):
        return True
    # `Path(".env").suffix` is "" (a leading dot is not an extension), so the
    # ".env" entry in CONFIG_EXTS only matches names like `prod.env`. Dotenv
    # files proper (`.env`, `.env.local`, `.env.production`) are matched by name.
    if name == ".env" or name.startswith(".env."):
        return True
    if ext in LANG_BY_EXT:
        return False
    if any(h in rel_path.lower() for h in CONFIG_NAME_HINTS):
        return True
    return False


def _skipped_record(rel_path: str, reason: str, size_bytes: int, detail: str) -> FileRecord:
    """A FileRecord for a file we chose not to parse. `language`/`is_config`
    still say what kind of file it is, so the UI can group it sensibly."""
    if _looks_like_config(rel_path):
        language, is_config = "config", True
    else:
        language, is_config = LANG_BY_EXT.get(Path(rel_path).suffix.lower(), "other"), False
    return FileRecord(path=rel_path, language=language, is_config=is_config,
                      skipped=True, skip_reason=reason, skip_detail=detail,
                      size_bytes=size_bytes)


def _record_skip(skipped, repo_root: Path, path: Path, reason: str, size_bytes: int,
                 detail: str, ctx=None) -> None:
    """Count, log (always, with the path) and optionally collect one skip."""
    rel_path = path.relative_to(repo_root).as_posix()
    _bump(ctx, f"parse.skipped.{reason}")
    log(LOG, "warning", "skipped file", path=rel_path, reason=reason,
        size_bytes=size_bytes, detail=detail)
    if skipped is not None:
        skipped.append(_skipped_record(rel_path, reason, size_bytes, detail))


def _looks_minified(text: str) -> bool:
    """Shape-based minified/bundled check: big file, extremely long lines."""
    if len(text) < MINIFIED_MIN_BYTES:
        return False
    lines = text.count("\n") + 1
    return len(text) / lines > MINIFIED_AVG_LINE_CHARS


# ------------------------------------------------------ grammar loading ----

_ts_available = None          # True / False once probed, else None
_ts_parser_cache = {}          # grammar name -> Parser | None
_ts_load_errors = {}           # grammar name (or "_module") -> error string


def _get_ts_parser(grammar_name: str):
    """Lazily load one Tree-sitter grammar's parser, memoized (hits *and*
    misses) so a missing package or unsupported grammar is probed once per
    process rather than once per file. Returns None if it cannot be loaded."""
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
            f"tree_sitter_language_pack not installed ({exc}); using the line-based "
            "fallback. `pip install tree-sitter-language-pack` to get real parse trees."
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


# ----------------------------------------------------------- node helpers ----

def _text(node, source: bytes) -> str:
    """Exact source text covered by a node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", "ignore")


def _first_field(node, field_names):
    """First child found under any of `field_names`, else None."""
    for name in field_names:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    return None


def _leaf_name(node, source: bytes):
    """Plain identifier text for a naming node, or None if it isn't one.

    A leaf node (no named children) is an identifier-like token and is used
    as is. A member chain (`this.handler`, `exports.foo`) yields its final
    attribute. Anything else (destructuring patterns, computed keys, string
    keys) is not a usable name."""
    if node is None:
        return None
    if node.named_child_count == 0:
        return _text(node, source).strip() or None
    return _leaf_name(_first_field(node, _F_MEMBER_ATTR), source)


def _leaf_node(node):
    """Same walk as _leaf_name, but returns the identifier NODE (for its
    position) instead of its text. None if there is no usable identifier."""
    if node is None:
        return None
    if node.named_child_count == 0:
        return node
    return _leaf_node(_first_field(node, _F_MEMBER_ATTR))


def _binding_name_node(node):
    """Identifier node of the binding `node` is the value of (see _binding_name)."""
    parent = node.parent
    if parent is None:
        return None
    for value_field in _F_BIND_VALUE:
        value = parent.child_by_field_name(value_field)
        if value is not None and value.id == node.id:
            for target_field in _F_BIND_TARGET:
                leaf = _leaf_node(parent.child_by_field_name(target_field))
                if leaf is not None:
                    return leaf
    return None


def _dotted(node, source: bytes) -> str:
    """Callee text as a dotted name: `requests.get`, `self.helper`, `this.a.b`.

    Mirrors the prototype's contract: a non-name base (e.g. a call result,
    `thing.resolve().url`) becomes "<expr>" at that position, and a callee
    that is not a name/member chain at all (`f()()`, `a[0]()`) is "<expr>"."""
    if node.named_child_count == 0:
        return _text(node, source)
    base = node.child_by_field_name(_F_MEMBER_BASE)
    attr = _first_field(node, _F_MEMBER_ATTR)
    if base is not None and attr is not None:
        return f"{_dotted(base, source)}.{_text(attr, source)}"
    return _OPAQUE


def _inner_arg_text(node, source: bytes) -> str:
    """Raw text inside the call's delimiters: "(url, 3)" -> "url, 3".

    One layer of matching parentheses is removed so downstream stages see the
    same shape of string regardless of language. A tagged-template argument
    (a bare template string) has no parentheses and is returned unchanged."""
    text = _text(node, source).strip()
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        return text[1:-1].strip()
    return text


# ------------------------------------------------- structure recognition ----

def _is_import(node) -> bool:
    """Import/re-export statement?

    Recognised structurally: the node either has a module-source field
    (`import x from "m"`, `export * from "m"`) or contains the `import`
    keyword token as a direct child (`import a.b`, `from m import n`,
    `import x = require("m")`). The keyword followed by `.` is a meta
    property (`import.meta`), not a module import."""
    if node.child_count == 0:
        return False
    if _first_field(node, _F_MODULE) is not None:
        return True
    kids = node.children
    if len(kids) < 2:       # a node that merely wraps the keyword token (e.g. the callee of `import(...)`)
        return False
    for i, kid in enumerate(kids):
        if not kid.is_named and kid.type == _IMPORT_KEYWORD:
            nxt = kids[i + 1] if i + 1 < len(kids) else None
            return not (nxt is not None and not nxt.is_named and nxt.type == _DOT_TOKEN)
    return False


def _module_text(node, source: bytes) -> str:
    """Module specifier text. Quoted (`"./x"`) -> unquoted, dots preserved.
    Bare (Python `.pkg.mod`) -> leading relative-import dots dropped, which
    matches how the prototype's `ast`-based pass reported relative imports."""
    text = _text(node, source).strip()
    if not text:
        return ""
    if text[0] in _QUOTES:
        return text.strip(_QUOTES)
    return text.lstrip(".")


def _import_targets(node, source: bytes):
    """The `raw` strings for one import node.

    * module + imported names  -> one per name, "module.name"   (from m import a, b)
    * module only              -> the module                      (import x from "m")
    * names only               -> one per name                    (import a.b, c)
    * neither                  -> the statement text, keyword and `;` trimmed
    """
    module_node = _first_field(node, _F_MODULE)
    if module_node is None:     # e.g. `import x = require("m")`: the source sits in a child clause
        for child in node.named_children:
            module_node = _first_field(child, _F_MODULE)
            if module_node is not None:
                break
    module = _module_text(module_node, source) if module_node is not None else None

    names = []
    for item in node.children_by_field_name(_F_NAME):
        inner = item.child_by_field_name(_F_NAME)      # `a as b` -> `a`
        names.append(_text(item if inner is None else inner, source))

    if module is not None and names:
        return [f"{module}.{n}" if module else n for n in names]
    if module is not None:
        return [module]
    if names:
        return names
    raw = " ".join(_text(node, source).split())
    return [raw.removeprefix(_IMPORT_KEYWORD + " ").rstrip(";").strip()]


def _as_call(node):
    """(callee_node, arguments_node) if `node` is a call/`new`, else None."""
    args = node.child_by_field_name(_F_ARGS)
    if args is None:
        return None
    callee = _first_field(node, _F_CALLEE)
    if callee is None:
        return None
    return callee, args


def _binding_name(node, source: bytes):
    """Name for an anonymous function/class from the binding it is the value
    of: `const f = () => {}`, `handler = function () {}`, `{ run: () => {} }`,
    `f = lambda: 0`. Returns None if `node` is not the value side of one."""
    parent = node.parent
    if parent is None:
        return None
    for value_field in _F_BIND_VALUE:
        value = parent.child_by_field_name(value_field)
        if value is not None and value.id == node.id:
            for target_field in _F_BIND_TARGET:
                name = _leaf_name(parent.child_by_field_name(target_field), source)
                if name:
                    return name
    return None


# --------------------------------------------------------- the extraction ----

def _extract_structure(rec: FileRecord, rel_path: str, root, source: bytes, ctx=None) -> None:
    """Single pre-order walk over the parse tree filling `rec`.

    Iterative (explicit stack) so deeply nested code cannot hit Python's
    recursion limit. Each stack frame carries the context the children
    inherit:

        caller      qualified name of the enclosing function (or module scope);
                    this is what a CallSite is attributed to
        class_name  innermost enclosing class, or None. Reset to None on
                    entering a function body, so only members declared
                    directly in a class body are reported as "method"
        in_args     True while inside a call's argument list. Anonymous
                    functions there (callbacks, `{ cb: () => {} }`) are not
                    named after their binding — it is an argument, not a
                    declaration — so they don't pollute the definitions list.
                    Reset by any scope (nodes with a body).
    """
    stack = [(root, f"{rel_path}::<module>", None, False)]

    while stack:
        node, caller, class_name, in_args = stack.pop()
        line = node.start_point[0] + 1

        # -- imports: record and don't descend (nothing inside is a real call)
        if _is_import(node):
            for raw in _import_targets(node, source):
                if raw:
                    rec.imports.append(Import(file=rel_path, raw=raw, line=line))
            continue

        child_caller, child_class, child_in_args = caller, class_name, in_args
        args_node = None

        call = _as_call(node)
        if call is not None:
            # -- calls: record, then keep descending (nested calls live in args)
            callee, args_node = call
            callee_expr = _dotted(callee, source)
            arg_text = _inner_arg_text(args_node, source)
            if len(arg_text) > MAX_ARG_CHARS:
                _bump(ctx, "parse.arg_repr.truncated")
                arg_text = arg_text[:MAX_ARG_CHARS]
            callee_leaf = _leaf_node(callee)
            rec.calls.append(CallSite(
                caller=caller, callee_expr=callee_expr, line=line,
                arg_text=arg_text, file=rel_path,
                callee_line=(callee_leaf.start_point[0] + 1) if callee_leaf is not None else line,
                callee_col=callee_leaf.start_point[1] if callee_leaf is not None else -1,
            ))
            if callee_expr in _MODULE_LOADER_CALLEES:
                module = _first_string_arg(args_node, source)
                if module:
                    rec.imports.append(Import(file=rel_path, raw=module, line=line))

        elif node.child_by_field_name(_F_BODY) is not None:
            # -- a scope. Function-like if it takes parameters, else class-like.
            is_function = _first_field(node, _F_PARAMS) is not None
            name_node = _leaf_node(node.child_by_field_name(_F_NAME))
            name = _leaf_name(node.child_by_field_name(_F_NAME), source)
            if name is None and not in_args:
                # Anonymous value bound to a name. Treated as a function: a
                # parameterless lambda has no `parameters` field, so the
                # shape alone cannot tell it from a class expression, and
                # function values are by far the common case.
                name = _binding_name(node, source)
                if name is not None:
                    name_node = _binding_name_node(node)
                is_function = is_function or name is not None
            child_in_args = False
            if name is not None:
                if is_function:
                    kind = "method" if class_name else "function"
                    qn = f"{rel_path}::{class_name}.{name}" if class_name else f"{rel_path}::{name}"
                    child_caller, child_class = qn, None
                else:
                    kind, qn = "class", f"{rel_path}::{name}"
                    child_class = name
                rec.definitions.append(Definition(
                    qualified_name=qn, name=name, kind=kind, file=rel_path,
                    start_line=line, end_line=node.end_point[0] + 1,
                    name_line=(name_node.start_point[0] + 1) if name_node is not None else line,
                    name_col=name_node.start_point[1] if name_node is not None else -1,
                ))

        # push children reversed so they pop in source order
        for child in reversed(node.children):
            entering_args = args_node is not None and child.id == args_node.id
            stack.append((child, child_caller, child_class, child_in_args or entering_args))


def _first_string_arg(args_node, source: bytes):
    """Module name from the first argument of a module-loader call, if that
    argument is a plain (non-interpolated) string literal."""
    for arg in args_node.named_children:
        if arg.is_extra:        # skip comments between the parentheses
            continue
        text = _text(arg, source)
        if len(text) >= 2 and text[0] in _QUOTES and text[-1] == text[0] and "${" not in text:
            return text[1:-1]
        return None
    return None


def _parse_treesitter(rel_path: str, text: str, language: str, grammar: str, ctx=None):
    """Parse one file with Tree-sitter. Returns a FileRecord, or None when
    Tree-sitter could not be used (grammar unavailable, or the parse/extract
    raised) so the caller can fall back. Never returns a half-filled record."""
    parser = _get_ts_parser(grammar)
    if parser is None:
        return None

    source = text.encode("utf-8", "ignore")
    rec = FileRecord(path=rel_path, language=language, source_lines=text.splitlines(),
                     parse_mode="treesitter")
    try:
        tree = parser.parse(source)
        _extract_structure(rec, rel_path, tree.root_node, source, ctx=ctx)
    except Exception as exc:  # noqa: BLE001 - one odd file shouldn't kill the run
        _bump(ctx, "parse.treesitter.parse_error")
        log(LOG, "warning", "tree-sitter failed on file; using fallback",
            file=rel_path, grammar=grammar, error=f"{type(exc).__name__}: {exc}")
        return None

    # Tree-sitter recovers from syntax errors, so the record above is still
    # useful — just count it so incomplete extraction is visible.
    if tree.root_node.has_error:
        rec.has_syntax_errors = True
        _bump(ctx, "parse.treesitter.syntax_error")
        if TRACE_EDGES:
            log(LOG, "debug", "file has syntax errors; extracted what the tree contains",
                file=rel_path, grammar=grammar)
    return rec


# ------------------------------------------------------ simple fallback ----
# Only reached when Tree-sitter is unavailable or raised (see module docstring).
# Deliberately minimal and line-based: imports and definitions only, flat
# naming, no call sites. It exists so the file still appears in the graph, not
# to compete with the real parse.

_FALLBACK_MODIFIERS = ("export ", "default ", "async ", "abstract ", "declare ")
_FALLBACK_DEFS = (("def ", "function"), ("function ", "function"),
                  ("class ", "class"), ("interface ", "class"))


def _strip_modifiers(line: str) -> str:
    """Drop leading `export default async ...` style modifiers."""
    stripped = True
    while stripped:
        stripped = False
        for mod in _FALLBACK_MODIFIERS:
            if line.startswith(mod):
                line, stripped = line[len(mod):], True
    return line


def _leading_identifier(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in "_$":
            chars.append(ch)
        else:
            break
    return "".join(chars)


def _fallback_import(line: str):
    """Module string from an import line (best effort, first name only)."""
    for quote in ("'", '"'):
        start = line.find(quote)
        if start != -1:
            end = line.find(quote, start + 1)
            if end != -1:
                return line[start + 1:end]
    if line.startswith("from ") and " import " in line:
        module, _, names = line[len("from "):].partition(" import ")
        module = module.strip().lstrip(".")
        first = names.split(",")[0].split(" as ")[0].strip("() ")
        return f"{module}.{first}" if module and first not in ("", "*") else (module or first)
    return line[len("import "):].split(",")[0].split(" as ")[0].strip()


def _parse_fallback(rel_path: str, text: str, language: str, ctx=None) -> FileRecord:
    _bump(ctx, "parse.treesitter.fallback_used")
    rec = FileRecord(path=rel_path, language=language, source_lines=text.splitlines(),
                     parse_mode="fallback")
    for i, line in enumerate(rec.source_lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            raw = _fallback_import(stripped)
            if raw:
                rec.imports.append(Import(file=rel_path, raw=raw, line=i))
            continue
        stripped = _strip_modifiers(stripped)
        for keyword, kind in _FALLBACK_DEFS:
            if stripped.startswith(keyword):
                name = _leading_identifier(stripped[len(keyword):].lstrip("* "))
                if name:
                    kw_at = line.find(keyword)
                    col = line.find(name, kw_at + len(keyword)) if kw_at >= 0 else -1
                    rec.definitions.append(Definition(
                        qualified_name=f"{rel_path}::{name}", name=name, kind=kind,
                        file=rel_path, start_line=i, end_line=i,
                        name_line=i, name_col=col,
                    ))
                break
    return rec


# ------------------------------------------------------------ entry points ----

def _parse_source(rel_path: str, text: str, language: str, ext: str, ctx=None) -> FileRecord:
    """Tree-sitter first; the line-based fallback only if it returned None."""
    rec = _parse_treesitter(rel_path, text, language, GRAMMAR_BY_EXT[ext], ctx=ctx)
    if rec is not None:
        _bump(ctx, "parse.treesitter.used")
        return rec
    return _parse_fallback(rel_path, text, language, ctx=ctx)


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
        rec = _skipped_record(rel_path, "read_error", 0, f"{type(exc).__name__}: {exc}")
        rec.language = "unknown"
        return rec

    if _looks_like_config(rel_path):
        _bump(ctx, "parse.config_files")
        return FileRecord(path=rel_path, language="config", source_lines=text.splitlines(), is_config=True)

    ext = path.suffix.lower()
    lang = LANG_BY_EXT.get(ext)
    if lang:
        if _looks_minified(text):
            size = len(text.encode("utf-8", "ignore"))
            avg = size // (text.count("\n") + 1)
            _bump(ctx, "parse.skipped.minified")
            detail = (f"looks minified/bundled: {size:,} bytes averaging {avg:,} chars per line "
                      f"(limit {MINIFIED_AVG_LINE_CHARS})")
            log(LOG, "warning", "skipped file", path=rel_path, reason="minified",
                size_bytes=size, detail=detail)
            return _skipped_record(rel_path, "minified", size, detail)
        return _parse_source(rel_path, text, lang, ext, ctx=ctx)
    _bump(ctx, "parse.language.unsupported")
    return FileRecord(path=rel_path, language="other", source_lines=text.splitlines())


_IMPACT_SKIPPED_SOURCE = ("Its definitions and calls are missing from the graph, and calls from "
                          "other files into it cannot be linked to it.")
_IMPACT_SKIPPED_CONFIG = ("Its contents were not read. The config check must treat this as UNKNOWN, "
                          "not as 'no matching host or key found'.")
_IMPACT_SKIPPED_OTHER = "The file was not read."


def parse_gaps(records) -> list:
    """Structured gaps derivable from the records alone (pure function):

      * every skipped record               -> kind = its skip_reason
      * every fallback-parsed source file  -> "parse_fallback" (no call sites)
      * every file with syntax errors      -> "syntax_errors" (partial extraction)

    Repo-level gaps that have no record (vendored directories, the file cap)
    are emitted by `discover_files` instead."""
    out = []
    for rec in records:
        if rec.skipped:
            if rec.is_config:
                impact = _IMPACT_SKIPPED_CONFIG
            elif rec.language in LANG_BY_EXT.values():
                impact = _IMPACT_SKIPPED_SOURCE
            else:
                impact = _IMPACT_SKIPPED_OTHER
            out.append(Gap(stage="parse", scope="file", kind=rec.skip_reason or "skipped",
                           path=rec.path, detail=rec.skip_detail, impact=impact))
        elif rec.parse_mode == "fallback":
            out.append(Gap(
                stage="parse", scope="file", kind="parse_fallback", path=rec.path,
                detail=("Tree-sitter could not be used for this file, so a minimal line-based "
                        "parse was used instead."),
                impact=("Call sites are missing (so it shows no outgoing calls) and definitions "
                        "and imports are approximate. This is NOT 'analysed, no calls found'.")))
        elif rec.has_syntax_errors:
            out.append(Gap(
                stage="parse", scope="file", kind="syntax_errors", path=rec.path,
                detail="The file contains syntax errors; tree-sitter extracted what the tree contains.",
                impact="Some definitions or calls in this file may be missing."))
    return out


def parse_repo(repo_root: Path, ctx=None, max_files: int = 400,
               max_source_bytes: int = MAX_SOURCE_BYTES,
               max_other_bytes: int = MAX_OTHER_BYTES, gaps=None):
    """Parse every discoverable file under `repo_root`.

    Returns one FileRecord per file that was parsed OR deliberately skipped
    (`rec.skipped`); filter on `rec.skipped` to separate them. A file whose
    parse raised is returned as a skipped record (`parse_exception`), not
    dropped. Vendored directories and files beyond the cap cannot be listed one
    by one; they are reported as repo-level Gap entries instead.

    Everything not analysed is appended to `gaps` (or `ctx.gaps`); see gaps.py."""
    sink = gap_sink(ctx, gaps)
    discovery_skips = []
    files = discover_files(repo_root, max_files=max_files, ctx=ctx, skipped=discovery_skips,
                           max_source_bytes=max_source_bytes, max_other_bytes=max_other_bytes,
                           gaps=sink)
    records = []
    for f in files:
        try:
            rec = parse_file(repo_root, f, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the run
            _bump(ctx, "parse.file_exception")
            log(LOG, "error", "parser raised on file", file=str(f),
                error=f"{type(exc).__name__}: {exc}")
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            records.append(_skipped_record(
                f.relative_to(repo_root).as_posix(), "parse_exception", size,
                f"the parser raised {type(exc).__name__}: {exc}"))
            continue
        records.append(rec)
        if rec.skipped:
            continue
        _bump(ctx, f"parse.language.{rec.language}")
        _bump(ctx, "parse.definitions", len(rec.definitions))
        _bump(ctx, "parse.calls", len(rec.calls))
        _bump(ctx, "parse.imports", len(rec.imports))
        if TRACE_EDGES:
            log(LOG, "debug", "parsed file", file=rec.path, language=rec.language,
                defs=len(rec.definitions), calls=len(rec.calls), imports=len(rec.imports))
    records.extend(discovery_skips)
    for g in parse_gaps(records):
        emit(sink, g)

    skipped_records = [r for r in records if r.skipped]
    if ctx is not None:
        if skipped_records:
            by_reason = {}
            for r in skipped_records:
                by_reason[r.skip_reason] = by_reason.get(r.skip_reason, 0) + 1
            shown = "; ".join(f"{r.path} ({r.skip_reason})" for r in skipped_records[:10])
            more = f" ... and {len(skipped_records) - 10} more" if len(skipped_records) > 10 else ""
            ctx.note("warning", "parse",
                     f"{len(skipped_records)} file(s) were not parsed "
                     f"({', '.join(f'{n} {k}' for k, n in sorted(by_reason.items()))}): "
                     f"{shown}{more}. Their definitions and calls are missing from the graph.")

        fell_back = ctx.count("parse.treesitter.fallback_used")
        used_ts = ctx.count("parse.treesitter.used")
        if fell_back:
            reason = _ts_load_errors.get("_module") or next(iter(_ts_load_errors.values()), None)
            log(LOG, "warning", "tree-sitter unavailable for some/all files; used line-based fallback",
                fallback_files=fell_back, treesitter_files=used_ts, reason=reason)
            ctx.note("warning", "parse",
                     f"{fell_back} file(s) were parsed with the minimal line-based fallback "
                     "instead of tree-sitter"
                     + (f" ({reason})" if reason else "")
                     + ". Their call sites are missing and definitions/imports are approximate.")

        broken = ctx.count("parse.treesitter.syntax_error")
        if broken:
            log(LOG, "warning", "some files contain syntax errors", files=broken)
            ctx.note("warning", "parse",
                     f"{broken} file(s) contain syntax errors; tree-sitter extracted what it "
                     "could, but some definitions or calls in them may be missing.")

        total_calls = ctx.count("parse.calls")
        log(LOG, "info", "parse complete",
            files=len(records),
            code_files=sum(1 for r in records if not r.is_config and not r.skipped
                           and r.language not in ("unknown", "other")),
            config_files=sum(1 for r in records if r.is_config and not r.skipped),
            skipped_files=len(skipped_records),
            definitions=ctx.count("parse.definitions"),
            calls=total_calls)
    return records