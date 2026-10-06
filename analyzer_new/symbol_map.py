"""
Symbol map: the one file the vector branch needs from steps 1-2.

`build_symbol_map` joins the tree-sitter records (parser.py) with the SCIP
symbol spans (scip_check.ScipResult) into a plain dict; `write_symbol_map` puts
it in `workspaces/{job_id}/artifacts/symbol_map.json` atomically;
`load_symbol_map` reads it back and checks its shape. The vector branch talks to
the rest of Peach only through that file.

Schema (version 1; JSON with sorted keys, byte-identical for identical input)
-----------------------------------------------------------------------------
    { "schema_version": 1, "commit_sha": "...", "pipeline_version": "...",
      "file_sha_algorithm": "sha256", "not_listed_note": "...",
      "files": { "<repo-relative POSIX path>": {
          "language", "coverage", "skip_reason", "line_count", "file_sha",
          "symbols": [ {symbol_id, qualified_name, name, kind, start_line,
                        end_line, name_line, name_col, parent_id,
                        boundary_source} ] } } }

coverage: "scip" (the SCIP index covers the file) | "treesitter_only" (parsed by
tree-sitter, not in the index) | "fallback_lines" (line-based fallback parse) |
"skipped" (deliberately not parsed; no symbols) | "unparsed" (config / other).

How symbols are built
---------------------
Every `parser.Definition` is a symbol. It is joined to a SCIP span of the same
file by the exact (name_line, name_col) of its identifier - the join
`resolver.build_definition_location_index` uses. A match whose span has an
`enclosing_range` takes SCIP's start/end and its symbol string
(boundary_source "scip"); otherwise the tree-sitter start/end and
`ts:{file}:{qualified_name}` are used (boundary_source "treesitter"). SCIP
function/method/class spans with no Definition are added too, unless they are
local, unnamed, or have no enclosing range (a name-line-only span says nothing
about where the body ends, so it is dropped and reported, not guessed).
SCIP spans are only used for files in `status["covered_files"]`.

`parent_id` is the smallest symbol whose line span contains the symbol. Spans
are whole lines, so two symbols on one line look nested; that is a known limit.
If two spans partially overlap, the later one (in (start_line, -end_line)
order) falls back to its tree-sitter span, and a `boundary_conflict` gap is
emitted either way. symbol_id is unique per file: a repeat (overloads, a
function redefined) gets "~2", "~3", ... appended in sorted order.

line_count / file_sha come from the file bytes under `repo_root`. line_count is
the "\\n"-split count the parser uses (`count("\\n") + 1`; 0 for an empty file).
When they cannot be computed they stay 0 / "" and a gap says so.

Gaps (stage "vector", see gaps.py): boundary_conflict, scip_symbol_dropped,
file_unreadable, file_stats_unavailable. A gap means "unknown".
"""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .gaps import Gap, emit, gap_sink
from .obs import get_logger, log

if TYPE_CHECKING:  # duck-typed at runtime; imported for the hints only
    from .parser import FileRecord
    from .scip_check import ScipResult, ScipSymbolSpan

LOG = get_logger("symbol_map")

SCHEMA_VERSION = 1
PIPELINE_VERSION = "0.1.0"
SYMBOL_MAP_FILENAME = "symbol_map.json"
FILE_SHA_ALGORITHM = "sha256"
COVERAGE_VALUES = ("scip", "treesitter_only", "fallback_lines", "skipped", "unparsed")
NOT_LISTED_NOTE = (
    "Only files the parser produced a record for are listed. Files beyond the "
    "parser's max_files cap and files in vendored directories (SKIP_DIRS) are "
    "not listed: absence here means 'not analysed', never 'has no symbols'."
)

_SCIP_SYMBOL_KINDS = frozenset({"function", "method", "class"})
_CHUNK = 1 << 20
_FILE_KEYS = ("language", "coverage", "skip_reason", "line_count", "file_sha", "symbols")
_SYMBOL_KEYS = ("symbol_id", "qualified_name", "name", "kind", "start_line", "end_line",
                "name_line", "name_col", "parent_id", "boundary_source")


class SymbolMapError(ValueError):
    """symbol_map.json is unreadable, malformed, or of an unsupported schema."""


@dataclass
class _Sym:
    """Working form of one symbol while a file is being assembled."""
    symbol_id: str
    qualified_name: str
    name: str
    kind: str
    start_line: int
    end_line: int
    name_line: int
    name_col: int
    boundary_source: str
    ts_start: Optional[int] = None   # tree-sitter span, kept so a SCIP span can be swapped back
    ts_end: Optional[int] = None
    ts_id: str = ""


def _bump(ctx, key: str, n: int = 1) -> None:
    bump = getattr(ctx, "bump", None)
    if bump is not None:
        bump(key, n)


# ------------------------------------------------------------------ file bytes ----

def _file_stats(repo_root: Path, rel_path: str) -> Tuple[int, str]:
    """(line_count, sha256 hex) of `repo_root/rel_path`, read in chunks. Raises
    OSError if unreadable and ValueError if the path leaves `repo_root`."""
    root = repo_root.resolve()
    full = (root / rel_path).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes repo root: {rel_path}") from None
    digest = hashlib.sha256()
    newlines = size = 0
    with full.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            newlines += chunk.count(b"\n")
            size += len(chunk)
    return (newlines + 1 if size else 0), digest.hexdigest()


# --------------------------------------------------------------------- symbols ----

def _pick_span(candidates):
    """Best of the SCIP spans sharing one name position: one with an enclosing
    range, then a function/method/class, then a non-local one, then file order."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (not c[1].has_enclosing_range,
                                          c[1].kind not in _SCIP_SYMBOL_KINDS,
                                          c[1].is_local, c[0]))


def _collect_symbols(rec: "FileRecord", scip_spans: list) -> Tuple[List[_Sym], int, int]:
    """Definitions joined to SCIP spans, plus SCIP-only symbols.
    Returns (symbols, dropped_no_range, dropped_no_name)."""
    by_pos: Dict[Tuple[int, int], list] = {}
    for i, span in enumerate(scip_spans):
        by_pos.setdefault((span.name_line, span.name_col), []).append((i, span))
    consumed = set()
    out: List[_Sym] = []
    for d in rec.definitions:
        ts_id = f"ts:{rec.path}:{d.qualified_name}"
        # name_col -1 means "unknown", the same convention resolver.py uses
        match = _pick_span(by_pos.get((d.name_line, d.name_col))) if d.name_col >= 0 else None
        sym = _Sym(symbol_id=ts_id, qualified_name=d.qualified_name, name=d.name, kind=d.kind,
                   start_line=d.start_line, end_line=d.end_line, name_line=d.name_line,
                   name_col=d.name_col, boundary_source="treesitter",
                   ts_start=d.start_line, ts_end=d.end_line, ts_id=ts_id)
        if match is not None:
            consumed.add(match[0])
            span = match[1]
            if span.has_enclosing_range:
                sym.symbol_id, sym.boundary_source = span.symbol, "scip"
                sym.start_line, sym.end_line = span.start_line, span.end_line
        out.append(sym)

    no_range = no_name = 0
    for i, span in enumerate(scip_spans):
        if i in consumed or span.is_local or span.kind not in _SCIP_SYMBOL_KINDS:
            continue
        if not span.display_name:
            no_name += 1
        elif not span.has_enclosing_range:
            no_range += 1
        else:
            out.append(_Sym(symbol_id=span.symbol, qualified_name=span.display_name,
                            name=span.display_name, kind=span.kind,
                            start_line=span.start_line, end_line=span.end_line,
                            name_line=span.name_line, name_col=span.name_col,
                            boundary_source="scip"))
    return out, no_range, no_name


def _sort_key(s: _Sym):
    return (s.start_line, -s.end_line, s.name_line, s.name_col, s.qualified_name, s.symbol_id)


def _sweep(ordered: List[_Sym]) -> Tuple[List[Optional[int]], List[int]]:
    """For symbols sorted by `_sort_key`: (parent index or None per symbol,
    indices of symbols that partially overlap an earlier one)."""
    parents: List[Optional[int]] = [None] * len(ordered)
    conflicts: List[int] = []
    active: List[int] = []          # earlier symbols that reach the current start line
    for i, s in enumerate(ordered):
        active = [j for j in active if ordered[j].end_line >= s.start_line]
        best = None
        partial = False
        for j in active:
            e = ordered[j]
            if e.end_line >= s.end_line:        # e contains s (e.start_line <= s.start_line)
                rank = (e.end_line - e.start_line, -j)   # smallest, then latest
                if best is None or rank < best:
                    best, parents[i] = rank, j
            else:
                partial = True
        if partial:
            conflicts.append(i)
        active.append(i)
    return parents, conflicts


def _arrange(syms: List[_Sym]) -> Tuple[List[Optional[int]], List[_Sym], int]:
    """Sort, resolve partial overlaps, make symbol_ids unique. Mutates `syms`
    (sorted in place). Returns (parents, conflicted symbols, remaining conflicts)."""
    conflicted: List[_Sym] = []
    seen_ids = set()                # identity, not field equality: two symbols can look alike
    while True:
        syms.sort(key=_sort_key)
        parents, conflicts = _sweep(syms)
        flip = None
        for i in conflicts:
            s = syms[i]
            if id(s) not in seen_ids:
                seen_ids.add(id(s))
                conflicted.append(s)
            if s.boundary_source == "scip" and s.ts_start is not None:
                flip = s
                break           # re-sweep: this swap may already resolve the later conflicts
        if flip is None:
            break
        flip.start_line, flip.end_line = flip.ts_start, flip.ts_end
        flip.boundary_source, flip.symbol_id = "treesitter", flip.ts_id
    used = set()
    for s in syms:
        candidate, n = s.symbol_id, 1
        while candidate in used:
            n += 1
            candidate = f"{s.symbol_id}~{n}"
        s.symbol_id = candidate
        used.add(candidate)
    return parents, conflicted, len(conflicts)


def _symbol_dicts(rec, scip_spans, path, sink) -> List[dict]:
    syms, no_range, no_name = _collect_symbols(rec, scip_spans)
    parents, conflicted, remaining = _arrange(syms)
    if conflicted:
        names = ", ".join(s.qualified_name for s in conflicted[:3])
        more = "" if len(conflicted) <= 3 else f" and {len(conflicted) - 3} more"
        emit(sink, Gap(
            stage="vector", scope="file", kind="boundary_conflict", path=path,
            count=len(conflicted),
            detail=(f"{len(conflicted)} symbol span(s) partially overlap an earlier symbol "
                    f"({names}{more}); {remaining} still overlap after falling back to "
                    f"tree-sitter spans where one was available."),
            impact="Chunks for these symbols may overlap or be bounded less precisely."))
    if no_range or no_name:
        emit(sink, Gap(
            stage="vector", scope="file", kind="scip_symbol_dropped", path=path,
            count=no_range + no_name,
            detail=(f"SCIP reported {no_range + no_name} function/method/class with no matching "
                    f"parser definition that could not be used: {no_range} without an enclosing "
                    f"range, {no_name} without a name."),
            impact="These symbols are missing from the map; their extent is unknown."))
    return [{"symbol_id": s.symbol_id, "qualified_name": s.qualified_name, "name": s.name,
             "kind": s.kind, "start_line": s.start_line, "end_line": s.end_line,
             "name_line": s.name_line, "name_col": s.name_col,
             "parent_id": None if p is None else syms[p].symbol_id,
             "boundary_source": s.boundary_source}
            for s, p in zip(syms, parents)]


def _coverage(rec, covered: set) -> str:
    if rec.skipped:
        return "skipped"
    if rec.path in covered:
        return "scip"
    if rec.parse_mode == "treesitter":
        return "treesitter_only"
    if rec.parse_mode == "fallback":
        return "fallback_lines"
    return "unparsed"


# ------------------------------------------------------------------ public API ----

def build_symbol_map(records, scip: Optional["ScipResult"], commit_sha: str, gaps=None, ctx=None,
                     repo_root=None, pipeline_version: Optional[str] = None) -> dict:
    """Join parser `records` with the `ScipResult` into the symbol-map dict.

    `repo_root` (optional) lets line_count / file_sha be computed from file
    bytes. `gaps` / `ctx` follow gaps.py. Never raises on missing or odd SCIP
    data (`scip` may be None: no file is then "scip"-covered)."""
    sink = gap_sink(ctx=ctx, gaps=gaps)
    status = getattr(scip, "status", None) or {}
    covered = set(status.get("covered_files") or ())
    spans_by_file = getattr(scip, "spans", None) or {}
    root = Path(repo_root) if repo_root is not None else None

    files: Dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: r.path):
        if rec.path in files:
            continue
        coverage = _coverage(rec, covered)
        line_count, file_sha = 0, ""
        if root is not None:
            try:
                line_count, file_sha = _file_stats(root, rec.path)
            except (OSError, ValueError) as exc:
                log(LOG, "warning", "could not read file for symbol map", file=rec.path,
                    error=f"{type(exc).__name__}: {exc}")
                emit(sink, Gap(stage="vector", scope="file", kind="file_unreadable", path=rec.path,
                               detail=f"{type(exc).__name__}: {exc}",
                               impact="line_count and file_sha are unknown (left as 0 / '')."))
        symbols: List[dict] = []
        if not rec.skipped:
            scip_spans = list(spans_by_file.get(rec.path) or []) if coverage == "scip" else []
            symbols = _symbol_dicts(rec, scip_spans, rec.path, sink)
        files[rec.path] = {"language": rec.language, "coverage": coverage,
                           "skip_reason": rec.skip_reason if rec.skipped else "",
                           "line_count": line_count, "file_sha": file_sha, "symbols": symbols}

    if root is None and files:
        emit(sink, Gap(stage="vector", scope="repo", kind="file_stats_unavailable",
                       count=len(files),
                       detail="No repo_root was given, so line_count and file_sha were not computed.",
                       impact="line_count is 0 and file_sha is '' for every file: unknown, not empty."))

    _bump(ctx, "vector.symbol_map.files", len(files))
    _bump(ctx, "vector.symbol_map.symbols", sum(len(f["symbols"]) for f in files.values()))
    return {"schema_version": SCHEMA_VERSION, "commit_sha": commit_sha or "",
            "pipeline_version": pipeline_version or PIPELINE_VERSION,
            "file_sha_algorithm": FILE_SHA_ALGORITHM, "not_listed_note": NOT_LISTED_NOTE,
            "files": files}


def _dumps(symbol_map: Any) -> str:
    # ensure_ascii keeps the output valid even for paths with undecodable bytes
    return json.dumps(symbol_map, sort_keys=True, indent=2) + "\n"


def write_symbol_map(symbol_map: dict, artifacts_dir) -> Path:
    """Write `symbol_map.json` into `artifacts_dir` (created if needed) via a
    temp file in the same directory plus `os.replace`, so a reader sees the old
    file or the new one, never a partial one. Returns the final path. Raises
    OSError if it cannot write: the vector branch has no other input."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / SYMBOL_MAP_FILENAME
    payload = _dumps(symbol_map).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".symbol_map.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _validate(data: Any) -> None:
    if not isinstance(data, dict):
        raise SymbolMapError("symbol map is not a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SymbolMapError(f"unsupported schema_version {data.get('schema_version')!r} "
                             f"(expected {SCHEMA_VERSION})")
    files = data.get("files")
    if not isinstance(files, dict) or "commit_sha" not in data or "pipeline_version" not in data:
        raise SymbolMapError("symbol map is missing commit_sha, pipeline_version or files")
    for path, entry in files.items():
        if not isinstance(entry, dict) or any(k not in entry for k in _FILE_KEYS):
            raise SymbolMapError(f"file entry {path!r} is malformed")
        if entry["coverage"] not in COVERAGE_VALUES:
            raise SymbolMapError(f"file entry {path!r} has unknown coverage {entry['coverage']!r}")
        if not isinstance(entry["symbols"], list):
            raise SymbolMapError(f"file entry {path!r}: symbols is not a list")
        for sym in entry["symbols"]:
            if not isinstance(sym, dict) or any(k not in sym for k in _SYMBOL_KEYS):
                raise SymbolMapError(f"file entry {path!r} has a malformed symbol")


def load_symbol_map(path) -> dict:
    """Read and validate a symbol map. Raises FileNotFoundError / OSError if it
    cannot be read and `SymbolMapError` if it is not a valid version-1 map."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, UnicodeDecodeError) as exc:    # JSONDecodeError is a ValueError
        raise SymbolMapError(f"symbol map is not valid JSON: {exc}") from exc
    _validate(data)
    return data