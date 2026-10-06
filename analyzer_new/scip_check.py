"""
Stage 2 (real): SCIP symbol resolution — indexing layer.

PEACH analyses Python, JavaScript and TypeScript (and the frameworks built
on them: Django/Flask/FastAPI/Streamlit, React/Next/Express/Nest/Angular...).
This module runs the real SCIP toolchain (https://github.com/sourcegraph/scip)
for exactly those languages and turns its output into one lookup table that
`resolver.py` uses to decide which calls resolve to code defined in the repo.

    scip-python       python               (wraps pyright)
    scip-typescript   javascript, typescript (wraps the TypeScript compiler;
                      one run indexes both languages)
    scip              the SCIP CLI, used only to dump an index as JSON

How it works
------------
1. For each indexer needed by the languages actually present in `records`,
   run it once against the repo and write a binary `.scip` index to a temp
   dir. (One run per indexer, not per language: JS and TS share a run.)
2. `scip print --json` converts the index to JSON. This stays a subprocess +
   `json.load` integration, with no generated protobuf bindings to vendor.
3. A symbol is "defined in the repo" if the index contains a *definition*
   occurrence for it. Every other occurrence of that symbol is a reference
   to repo code. References to anything else (third-party packages, the
   standard library, globals like `fetch`) have no definition occurrence in
   our index and therefore never resolve — that is how `requests.post` is
   told apart from a repo function that is also named `post`.
4. References are matched to our call sites by exact position — file, line
   AND column of the callee's final identifier (`CallSite.callee_line/col`
   from parser.py) — never by line alone, since many calls share a line
   (`requests.post(helper(1))` is two calls on one line, only one internal).

Symbol spans
------------
The same pass over each index that builds the reference map also records one
`ScipSymbolSpan` per definition occurrence (name position, enclosing range,
kind, display name). `run_scip` returns them in a `ScipResult` together with the
reference map and status, so SCIP runs once per job and everything it learned
can be reused; `build_reference_map` is a thin wrapper that returns only the
map and status.

Failure handling (nothing here may take the pipeline down)
----------------------------------------------------------
Every external step is wrapped: a missing binary, a non-zero exit, a timeout,
an unreadable index or any unexpected exception marks *that indexer's
languages* as skipped (with the reason) and the run carries on. The result
reports which files the index actually covers (`status["covered_files"]`), so
the resolver can use its fallback for exactly the files SCIP did not cover
and nothing else.

Verified against scip-python 0.6.6, scip-typescript 0.4.0 and scip CLI 0.10.0.
Facts learned from those tools and baked in below:
  * `scip print --json` uses snake_case keys (`relative_path`, `symbol_roles`);
    camelCase is accepted too in case a build emits protojson names.
  * scip-python crashes without a project version when the folder is not a
    git checkout, but still writes a tiny index file — so success requires
    exit code 0 AND a non-empty file, and we always pass `--project-version`.
  * scip-typescript fails on a repo with no tsconfig.json unless given
    `--infer-tsconfig`, which writes a tsconfig.json into the repo; that
    file is removed again afterwards.
  * scip-typescript skips files over 1 MB; those files simply show up as
    "not covered" and get the fallback.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .obs import get_logger, log
from .parser import SKIP_DIRS

LOG = get_logger("scip_check")

INDEX_TIMEOUT = int(os.environ.get("PEACH_SCIP_INDEX_TIMEOUT", "300"))
PRINT_TIMEOUT = int(os.environ.get("PEACH_SCIP_PRINT_TIMEOUT", "120"))

_ROLE_DEFINITION = 0x1          # scip.proto SymbolRole.Definition bit
_PROJECT_VERSION = "0.0.0"      # scip-python needs *a* version; its value is irrelevant to us

_bin_cache = {}
_lock = threading.Lock()


# ------------------------------------------------------------ binaries / subprocess ----

def _find_binary(name):
    """Locate an executable in $SCIP_HOME, then $PATH. `shutil.which` also
    applies PATHEXT, so `scip-python` finds `scip-python.cmd` on Windows.
    Misses are cached too (one probe per process)."""
    with _lock:
        if name in _bin_cache:
            return _bin_cache[name]
    found = None
    home = os.environ.get("SCIP_HOME")
    if home:
        found = shutil.which(name, path=home)
    if not found:
        found = shutil.which(name)
    with _lock:
        _bin_cache[name] = found
    return found


def _run(cmd, cwd, timeout, stdout=None):
    """Run a command. Returns (CompletedProcess, None) or (None, error text).
    Never raises: timeouts, launch failures and decode problems all come back
    as an error string. Output is decoded as UTF-8 with replacement, because
    the Windows locale codec would otherwise raise on tool output."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return proc, None
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - must never propagate
        return None, f"could not run: {type(exc).__name__}: {exc}"


def _error_detail(proc):
    """One useful line from a failed tool's output: the last line mentioning
    an error, else the last line."""
    lines = [ln.strip() for ln in ((proc.stderr or "") + "\n" + (proc.stdout or "")).splitlines()
             if ln.strip()]
    for line in reversed(lines):
        if "error" in line.lower():
            return line[:200]
    return lines[-1][:200] if lines else "no output"


# ------------------------------------------------------------------ indexer commands ----

def _noop():
    pass


def _remove_file(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


def _nested_tsconfig_dirs(root: Path):
    """Directories (relative, POSIX) below `root` that contain a tsconfig.json,
    skipping vendored/build directories. Used when the repo root has no
    tsconfig of its own (monorepos, `frontend/` + `backend/` layouts)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "tsconfig.json" in filenames and Path(dirpath) != root:
            found.append(Path(dirpath).relative_to(root).as_posix())
    return sorted(found)


def _python_command(binary, root: Path, out: Path):
    cmd = [binary, "index", "--cwd", str(root),
           "--project-name", root.name or "repo",
           "--project-version", _PROJECT_VERSION,
           "--output", str(out)]
    return cmd, _noop


def _typescript_command(binary, root: Path, out: Path):
    """scip-typescript needs a TypeScript project to index:
      * root tsconfig.json present  -> use it (it may reference sub-projects)
      * else nested tsconfig.json   -> index each of those projects
      * else (plain JS repo)        -> --infer-tsconfig, which writes a
        tsconfig.json into the repo; the returned cleanup removes it."""
    cmd = [binary, "index", "--cwd", str(root), "--output", str(out), "--no-progress-bar"]
    cleanup = _noop
    if not (root / "tsconfig.json").is_file():
        nested = _nested_tsconfig_dirs(root)
        if nested:
            cmd.extend(nested)
        else:
            cmd.append("--infer-tsconfig")
            cleanup = lambda: _remove_file(root / "tsconfig.json")  # noqa: E731
    return cmd, cleanup


# indexer binary -> (languages it covers, command builder)
_INDEXERS = {
    "scip-python": (("python",), _python_command),
    "scip-typescript": (("javascript", "typescript"), _typescript_command),
}
_LANGUAGE_TO_INDEXER = {lang: name for name, (langs, _) in _INDEXERS.items() for lang in langs}


def _build_index(indexer, repo_root: Path, workdir: Path):
    """Run one indexer. Returns (index_path, None) or (None, reason)."""
    languages, build_command = _INDEXERS[indexer]
    binary = _find_binary(indexer)
    if not binary:
        return None, f"{indexer} not found on $PATH/$SCIP_HOME"

    out_path = workdir / f"{indexer}.scip"
    cmd, cleanup = build_command(binary, repo_root, out_path)
    try:
        proc, err = _run(cmd, repo_root, INDEX_TIMEOUT)
    finally:
        cleanup()
    if err:
        return None, f"{indexer} {err}"
    # Exit code AND a non-empty file: scip-python writes a stub file even when
    # it crashes (exit 1), so the file's existence alone proves nothing.
    if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        return None, f"{indexer} failed (exit {proc.returncode}): {_error_detail(proc)}"
    return out_path, None


def _read_index(scip_path: Path, workdir: Path):
    """`scip print --json` -> parsed dict. The JSON goes to a file, not a
    pipe, so a large index is not held in memory twice. Returns (data, None)
    or (None, reason)."""
    json_path = workdir / (scip_path.name + ".json")
    try:
        with open(json_path, "wb") as fh:
            proc, err = _run([_find_binary("scip"), "print", "--json", str(scip_path)],
                             workdir, PRINT_TIMEOUT, stdout=fh)
    except OSError as exc:
        return None, f"`scip print` could not write its output: {exc}"
    if err:
        return None, f"`scip print` {err}"
    if proc.returncode != 0:
        return None, f"`scip print` failed (exit {proc.returncode}): {_error_detail(proc)}"
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, MemoryError) as exc:
        return None, f"could not parse `scip print --json` output: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "unexpected `scip print --json` output (not a JSON object)"
    return data, None


# ----------------------------------------------------------------------- result types ----

@dataclass
class ScipSymbolSpan:
    """One definition occurrence from a SCIP index, with the source span it covers.

    `name_line` / `name_col` locate the declared identifier with the same
    convention as the ref_map definition locations (1-based line, 0-based column).
    `start_line` / `end_line` are 1-based and inclusive and come from the
    occurrence's `enclosing_range` (the whole function / class body). Without an
    enclosing range both equal `name_line` and `has_enclosing_range` is False, so
    a consumer must not mistake that for a one-line definition. `kind` is one of
    function | method | class | variable | module | other; `display_name` is ""
    when the index carries no SymbolInformation for the symbol."""
    symbol: str
    display_name: str
    kind: str
    file: str                   # POSIX repo-relative
    name_line: int
    name_col: int
    start_line: int
    end_line: int
    has_enclosing_range: bool
    is_local: bool              # symbol starts with "local " (unique only within `file`)


def _empty_status():
    return {"indexed": [], "skipped": {}, "covered_files": set(), "scip_print_available": False}


@dataclass
class ScipResult:
    """Everything one SCIP run produced, so it can be indexed once and reused.

    ref_map : same as `build_reference_map`'s first return value.
    status  : same as `build_reference_map`'s second return value.
    spans   : {file: [ScipSymbolSpan]} sorted by (start_line, -end_line)."""
    ref_map: dict = field(default_factory=dict)
    status: dict = field(default_factory=_empty_status)
    spans: dict = field(default_factory=dict)

    def to_json_safe(self) -> dict:
        """Plain dicts/lists/strings/numbers only, ready for `json.dumps`.

        ref_map keys `(file, line, col)` become "file:line:col" (split with
        `rsplit(":", 2)`); definition locations become [file, line, col] lists;
        `status["covered_files"]` becomes a sorted list; spans become dicts."""
        return {
            "ref_map": {f"{f}:{ln}:{col}": [list(loc) for loc in locations]
                        for (f, ln, col), locations in sorted(self.ref_map.items())},
            "status": _json_safe(self.status),
            "spans": {path: [asdict(s) for s in spans]
                      for path, spans in sorted(self.spans.items())},
        }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {(":".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v)
                for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((_json_safe(v) for v in obj), key=str)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ------------------------------------------------------------------- symbol kinds ----

# scip.proto `SymbolInformation.Kind` (checked against sourcegraph/scip main):
# our coarse kind -> {enum name: enum number}. Names not listed here (Array, Boolean,
# Type, TypeAlias, Message, Operator, Event, ...) normalise to "other".
_KIND_GROUPS = {
    "function": {"Function": 17},
    "method": {
        "Method": 26, "AbstractMethod": 66, "Accessor": 72, "Constructor": 9, "Getter": 18,
        "MethodAlias": 74, "MethodSpecification": 67, "ProtocolMethod": 68,
        "PureVirtualMethod": 69, "Setter": 45, "SingletonMethod": 76, "StaticMethod": 80,
        "TraitMethod": 70, "TypeClassMethod": 71,
    },
    "class": {
        "Class": 7, "Contract": 62, "Enum": 11, "Interface": 21, "Mixin": 85, "Object": 33,
        "Protocol": 42, "SingletonClass": 75, "Struct": 49, "Trait": 53, "TypeClass": 56,
        "Union": 59,
    },
    "variable": {
        "Variable": 61, "Attribute": 4, "Constant": 8, "EnumMember": 12, "Field": 15,
        "Parameter": 37, "Property": 41, "SelfParameter": 44, "StaticDataMember": 77,
        "StaticField": 79, "StaticProperty": 81, "StaticVariable": 82, "ThisParameter": 52,
        "Value": 60,
    },
    "module": {
        "Module": 29, "Namespace": 30, "Package": 35, "PackageObject": 36, "File": 16,
        "Library": 64,
    },
}


def _kind_key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


_KIND_BY_NAME = {_kind_key(n): k for k, members in _KIND_GROUPS.items() for n in members}
_KIND_BY_INT = {num: k for k, members in _KIND_GROUPS.items() for num in members.values()}


def _normalise_kind(raw):
    """SymbolInformation.kind (enum NAME string, as `scip print --json` emits, or its
    integer value) -> function | method | class | variable | module | other."""
    if isinstance(raw, bool):
        return "other"
    if isinstance(raw, int):
        return _KIND_BY_INT.get(raw, "other")
    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            return _KIND_BY_INT.get(int(text), "other")
        return _KIND_BY_NAME.get(_kind_key(text), "other")
    return "other"


# ----------------------------------------------------------------------- index -> map ----

def _doc_path(doc):
    """Document path as our POSIX-style relative path (indexers on Windows
    may report backslashes)."""
    path = doc.get("relative_path") or doc.get("relativePath")
    return path.replace("\\", "/") if path else None


def _roles(occ):
    return occ.get("symbol_roles", occ.get("symbolRoles", 0)) or 0


def _parse_range(rng):
    """SCIP range -> (1-based line, 0-based start column), or None.

    Ranges are 0-based [line, start, end] (single line) or
    [start_line, start, end_line, end]. Zero-width ranges are dropped: the
    indexers emit one at (1, 0) for the file's own module symbol, which would
    otherwise collide with a real identifier at the very top-left of a file."""
    if not rng or len(rng) not in (3, 4):
        return None
    if len(rng) == 3:
        line, start, end = rng
        if start == end:
            return None
    else:
        line, start, end_line, end = rng
        if (line, start) == (end_line, end):
            return None
    return line + 1, start


def _symbol_key(path, symbol):
    """`local N` symbols are only unique within one document (every file
    numbers its own from 0), so they are scoped by file; all others are global."""
    return (path, symbol) if symbol.startswith("local ") else symbol


def _parse_span(rng):
    """SCIP (enclosing) range -> (start_line, start_col, end_line, end_col), or None.

    Lines are 1-based, columns 0-based (the `_parse_range` convention). `end_line`
    is the last line the span *covers*, inclusive. A multi-line range whose end
    column is 0 stops at the very start of its end line, so the span really ends
    on the previous line and `end_line` is reported as that line (`end_col` stays
    0). Zero-width, reversed or malformed ranges give None."""
    try:
        if not rng or len(rng) not in (3, 4):
            return None
        if len(rng) == 3:
            line, start, end = rng
            if end <= start:
                return None
            return line + 1, start, line + 1, end
        line, start, end_line, end = rng
        if (end_line, end) <= (line, start):
            return None
        last = end_line + 1
        if end == 0 and end_line > line:
            last = end_line
        return line + 1, start, last, end
    except (TypeError, ValueError):
        return None


def _symbol_table(doc):
    """{symbol: (kind, display_name)} from a document's SymbolInformation list."""
    table = {}
    for info in doc.get("symbols") or []:
        symbol = info.get("symbol")
        if symbol:
            table[symbol] = (info.get("kind"),
                             info.get("display_name") or info.get("displayName") or "")
    return table


def _make_span(symbol, path, pos, occ, symtab):
    name_line, name_col = pos
    enclosing = _parse_span(occ.get("enclosing_range") or occ.get("enclosingRange"))
    if enclosing:
        start_line, _, end_line, _ = enclosing
    else:
        start_line = end_line = name_line
    raw_kind, display_name = symtab.get(symbol, (None, ""))
    return ScipSymbolSpan(
        symbol=symbol, display_name=display_name, kind=_normalise_kind(raw_kind), file=path,
        name_line=name_line, name_col=name_col, start_line=start_line, end_line=end_line,
        has_enclosing_range=enclosing is not None, is_local=symbol.startswith("local "))


def _span_sort_key(span):
    return (span.start_line, -span.end_line, span.name_line, span.name_col, span.symbol)


def _merge_index_full(index_data, wanted):
    """Convert one index into `(ref_map, covered_files, spans)`.

    ref_map and covered_files are exactly what `_merge_index` returns. spans is
    {file: [ScipSymbolSpan]}: one entry per definition occurrence (NOT filtered
    by `wanted`), sorted by (start_line, -end_line). Only these three small
    structures are kept; the raw index JSON is not retained."""
    documents = index_data.get("documents") or []
    covered = set()
    defs = {}
    spans = {}

    # pass 1: every definition occurrence, grouped by symbol, plus its span
    for doc in documents:
        path = _doc_path(doc)
        if not path:
            continue
        covered.add(path)
        symtab = None
        for occ in doc.get("occurrences") or []:
            symbol = occ.get("symbol")
            if not symbol or not (_roles(occ) & _ROLE_DEFINITION):
                continue
            pos = _parse_range(occ.get("range"))
            if pos:
                defs.setdefault(_symbol_key(path, symbol), []).append((path, pos[0], pos[1]))
                if symtab is None:
                    symtab = _symbol_table(doc)
                spans.setdefault(path, []).append(_make_span(symbol, path, pos, occ, symtab))

    # pass 2: references at call-site positions whose symbol is defined here
    ref_map = {}
    for doc in documents:
        path = _doc_path(doc)
        if not path:
            continue
        for occ in doc.get("occurrences") or []:
            symbol = occ.get("symbol")
            if not symbol or (_roles(occ) & _ROLE_DEFINITION):
                continue
            pos = _parse_range(occ.get("range"))
            if not pos or (path, pos[0], pos[1]) not in wanted:
                continue
            locations = defs.get(_symbol_key(path, symbol))
            if locations:
                ref_map[(path, pos[0], pos[1])] = tuple(locations)

    for file_spans in spans.values():
        file_spans.sort(key=_span_sort_key)
    return ref_map, covered, spans


def _merge_index(index_data, wanted):
    """Convert one index into `(ref_map, covered_files)`.

    ref_map: {(file, line, col): (def_location, ...)} for references at a
    position in `wanted` (the call sites we actually have) whose symbol has
    at least one definition occurrence in the index. A def_location is
    (file, line, col) of the declared identifier. Filtering to `wanted`
    keeps memory proportional to the number of calls, not to every
    reference in the repo. (`_merge_index_full` also returns symbol spans.)"""
    ref_map, covered, _spans = _merge_index_full(index_data, wanted)
    return ref_map, covered


# ----------------------------------------------------------------------- entry point ----

def run_scip(records, repo_root, ctx=None):
    """Run whichever SCIP indexers apply to the languages present in `records`
    and return a `ScipResult`. This is the only place indexers run. Never raises.

    `ScipResult.ref_map` and `.status` have exactly the shape documented on
    `build_reference_map`; `.spans` maps each indexed file to the
    `ScipSymbolSpan`s of its definitions. A file in `status["covered_files"]` is
    authoritative: a call there that is not in ref_map is NOT repo code. A file
    outside it was not seen by SCIP at all."""
    status = {"indexed": [], "skipped": {}, "covered_files": set(),
              "scip_print_available": _find_binary("scip") is not None}
    ref_map = {}
    spans = {}
    try:
        _run_indexers(records, Path(repo_root), status, ref_map, spans)
    except Exception as exc:  # noqa: BLE001 - last resort: SCIP must never crash the run
        log(LOG, "error", "unexpected error in SCIP stage; falling back for everything",
            error=f"{type(exc).__name__}: {exc}")
        ref_map.clear()
        spans.clear()
        status["covered_files"].clear()
        status["indexed"].clear()
        for lang in {r.language for r in records} & set(_LANGUAGE_TO_INDEXER):
            status["skipped"][lang] = f"unexpected error: {type(exc).__name__}: {exc}"
    _report(ctx, status, ref_map)
    return ScipResult(ref_map=ref_map, status=status, spans=spans)


def build_reference_map(records, repo_root, ctx=None):
    """Run whichever SCIP indexers apply to the languages present in
    `records` and return `(ref_map, status)`. Never raises. Thin wrapper over
    `run_scip`, which also returns symbol spans.

    ref_map : {(file, line, col): ((def_file, def_line, def_col), ...)} — for
              every call site SCIP resolved to code defined in the repo.
              Keyed by the callee identifier's position (CallSite.callee_line
              / callee_col); values are Definition.name_line/name_col positions.
    status  : {
        "indexed":  languages covered by a successful index,
        "skipped":  {language: reason} for languages SCIP could not handle,
        "covered_files": set of repo-relative paths present in an index,
        "scip_print_available": whether the `scip` CLI was found,
    }
    A file in `covered_files` is authoritative: a call there that is not in
    ref_map is NOT repo code. A file outside it was not seen by SCIP at all.
    """
    result = run_scip(records, repo_root, ctx=ctx)
    return result.ref_map, result.status


def _run_indexers(records, repo_root: Path, status, ref_map, spans=None):
    """Run the indexers needed for `records`, filling `status`, `ref_map` and
    (if given) `spans` in place. A file present in two indexes keeps the span
    list of the first one merged."""
    if spans is None:
        spans = {}
    present = sorted({r.language for r in records
                      if not r.is_config and not r.skipped} & set(_LANGUAGE_TO_INDEXER))
    if not present:
        return
    needed = sorted({_LANGUAGE_TO_INDEXER[lang] for lang in present})

    if not status["scip_print_available"]:
        reason = ("`scip` CLI not found on $PATH/$SCIP_HOME (needed to read any index); "
                  "install it from https://github.com/sourcegraph/scip")
        for lang in present:
            status["skipped"][lang] = reason
        return

    wanted = {(c.file, c.callee_line, c.callee_col)
              for r in records for c in r.calls if c.callee_col >= 0}

    workdir = Path(tempfile.mkdtemp(prefix="peach_scip_"))
    try:
        for indexer in needed:
            languages = [l for l in _INDEXERS[indexer][0] if l in present]
            try:
                reason = None
                index_path, reason = _build_index(indexer, repo_root, workdir)
                if index_path:
                    data, reason = _read_index(index_path, workdir)
                    if data is not None:
                        refs, covered, new_spans = _merge_index_full(data, wanted)
                        data = None     # the raw index JSON is not kept
                        ref_map.update(refs)
                        for span_file, file_spans in new_spans.items():
                            spans.setdefault(span_file, file_spans)
                        status["covered_files"] |= covered
                        status["indexed"].extend(languages)
                        log(LOG, "info", "SCIP index merged", indexer=indexer,
                            languages=languages, files=len(covered), call_references=len(refs),
                            symbol_spans=sum(len(v) for v in new_spans.values()))
                        continue
            except Exception as exc:  # noqa: BLE001 - one indexer failing must not stop the other
                reason = f"{indexer} raised {type(exc).__name__}: {exc}"
            for lang in languages:
                status["skipped"][lang] = reason
            log(LOG, "warning", "SCIP unavailable for languages; fallback will be used",
                indexer=indexer, languages=languages, reason=reason)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _report(ctx, status, ref_map):
    if ctx is None:
        return
    ctx.bump("resolve.scip.languages_indexed", len(status["indexed"]))
    ctx.bump("resolve.scip.languages_skipped", len(status["skipped"]))
    ctx.bump("resolve.scip.files_covered", len(status["covered_files"]))
    ctx.bump("resolve.scip.references", len(ref_map))
    if status["skipped"]:
        ctx.note("warning", "resolve",
                 "SCIP resolution failed for: " +
                 "; ".join(f"{lang} ({reason})" for lang, reason in status["skipped"].items()) +
                 ". Calls in those files use the same-file fallback instead.")