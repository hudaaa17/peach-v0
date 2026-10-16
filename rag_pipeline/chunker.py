"""Semantic chunking for the RAG index.

Unlike length-based vector chunking, every chunk here is bounded by an
existing code unit: a function, a method, a class, a file header (imports +
module docstring), or — for files parser.py didn't extract definitions from
(config, markup, tiny scripts) — the whole file. This keeps each chunk
meaningful on its own, which matters for both retrieval quality and for the
SLM being able to reason about "what breaks if I change X" without a chunk
boundary having sliced X in half.

This module deliberately does NOT re-parse the repo. It reuses the
`code_records` / `config_records` that `parser.py` already produced during
the graph stage, so the two views of the repo (graph + RAG) stay in sync and
we don't pay for a second parse pass. It only reads raw file bytes off disk
(still available at hook time, before pipeline.py's cleanup) to pull the
source text for each definition's line range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 8000  # soft ceiling; oversized chunks are flagged, not dropped


@dataclass
class Chunk:
    id: str
    repo_path: str                 # path relative to repo root
    language: str
    kind: str                      # "function" | "method" | "class" | "file_header" | "file"
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    code: str
    parent_qualified_name: Optional[str] = None
    imports: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def chunk_repo(repo_root, code_records: Iterable, config_records: Iterable = None,
               max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Build semantic chunks for every code record, plus config records."""
    repo_root = Path(repo_root)
    code_records = list(code_records)
    config_records = list(config_records) if config_records else []
    logger.info("chunking started: %d code files, %d config files",
                len(code_records), len(config_records))

    chunks: list[Chunk] = []

    for rec in code_records:
        rec_chunks = _chunk_code_record(repo_root, rec, max_chars)
        if not rec_chunks:
            logger.debug("no chunks produced for %s (unreadable or empty)", rec.path)
        chunks.extend(rec_chunks)

    for rec in config_records:
        chunks.extend(_chunk_config_record(repo_root, rec, max_chars))

    by_kind: dict = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    logger.info("chunking finished: %d chunks total (%s)", len(chunks),
                ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    return chunks


def _read_lines(repo_root: Path, rel_path: str) -> Optional[list[str]]:
    try:
        text = (repo_root / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text.splitlines()


def _chunk_code_record(repo_root: Path, rec, max_chars: int) -> list[Chunk]:
    lines = _read_lines(repo_root, rec.path)
    if lines is None:
        return []
    total_lines = len(lines)
    imports = [getattr(imp, "raw", str(imp)) for imp in getattr(rec, "imports", [])]

    defs = sorted(getattr(rec, "definitions", []), key=lambda d: d.start_line)
    out: list[Chunk] = []

    if not defs:
        # No extracted definitions (script with only top-level code, or a
        # language parser.py doesn't extract defs for) — chunk the whole file.
        out.extend(_split_oversized(
            repo_path=rec.path, language=rec.language, kind="file",
            name=Path(rec.path).name, qualified_name=f"{rec.path}::<file>",
            start_line=1, end_line=total_lines,
            code="\n".join(lines), imports=imports, max_chars=max_chars,
        ))
        return out

    # File header chunk: everything before the first definition (imports,
    # module docstring, module-level constants) — useful retrieval context
    # even though it isn't itself a function/class.
    header_end = min(max(defs[0].start_line - 1, 0), total_lines)
    header_code = "\n".join(lines[:header_end]).strip()
    if header_code:
        out.append(Chunk(
            id=f"{rec.path}::<header>",
            repo_path=rec.path, language=rec.language, kind="file_header",
            name=Path(rec.path).name, qualified_name=f"{rec.path}::<header>",
            start_line=1, end_line=header_end, code=header_code, imports=imports,
        ))

    for i, d in enumerate(defs):
        start = max(d.start_line, 1)
        explicit_end = getattr(d, "end_line", None)
        if explicit_end:
            end = explicit_end
        else:
            # Fallback: a definition's body runs until the next definition
            # starts (this also correctly gives a class chunk just its
            # header/docstring/class-level attrs, since its methods are the
            # "next" definitions and become their own chunks).
            nxt_start = defs[i + 1].start_line if i + 1 < len(defs) else total_lines + 1
            end = max(start, nxt_start - 1)
        end = min(end, total_lines)

        code = "\n".join(lines[start - 1:end]).strip()
        if not code:
            continue

        kind = getattr(d, "kind", "function")
        qn = getattr(d, "qualified_name", f"{rec.path}::{d.name}")
        out.extend(_split_oversized(
            repo_path=rec.path, language=rec.language, kind=kind,
            name=d.name, qualified_name=qn, start_line=start, end_line=end,
            code=code, imports=imports, max_chars=max_chars,
            parent_qualified_name=_infer_parent(qn),
        ))

    return out


def _chunk_config_record(repo_root: Path, rec, max_chars: int) -> list[Chunk]:
    lines = _read_lines(repo_root, rec.path)
    if lines is None:
        return []
    code = "\n".join(lines).strip()
    if not code:
        return []
    return _split_oversized(
        repo_path=rec.path, language=getattr(rec, "language", "config"), kind="config",
        name=Path(rec.path).name, qualified_name=f"{rec.path}::<config>",
        start_line=1, end_line=len(lines), code=code, imports=[], max_chars=max_chars,
    )


def _infer_parent(qualified_name: str) -> Optional[str]:
    """Best-effort: 'module.ClassName.method' -> 'module.ClassName'."""
    sep = "::" if "::" in qualified_name else "."
    parts = qualified_name.split(sep)
    return sep.join(parts[:-1]) if len(parts) > 1 else None


def _split_oversized(*, repo_path, language, kind, name, qualified_name,
                      start_line, end_line, code, imports, max_chars,
                      parent_qualified_name=None) -> list[Chunk]:
    """Keep a definition as one chunk whenever possible (that's the whole
    point of semantic chunking). Only split when it exceeds max_chars, and
    even then split on line boundaries and flag the pieces as partial so a
    downstream retriever/SLM knows to fetch siblings for full context."""
    if len(code) <= max_chars:
        return [Chunk(
            id=f"{repo_path}::{qualified_name}::{start_line}-{end_line}",
            repo_path=repo_path, language=language, kind=kind, name=name,
            qualified_name=qualified_name, start_line=start_line, end_line=end_line,
            code=code, imports=imports, parent_qualified_name=parent_qualified_name,
        )]

    code_lines = code.splitlines()
    chunks = []
    part = 1
    buf: list[str] = []
    buf_start = start_line
    cur = start_line
    for line in code_lines:
        buf.append(line)
        if sum(len(l) + 1 for l in buf) >= max_chars:
            chunks.append(Chunk(
                id=f"{repo_path}::{qualified_name}::part{part}::{buf_start}-{cur}",
                repo_path=repo_path, language=language, kind=kind, name=name,
                qualified_name=qualified_name, start_line=buf_start, end_line=cur,
                code="\n".join(buf), imports=imports,
                parent_qualified_name=parent_qualified_name,
                metadata={"oversized": True, "part": part},
            ))
            part += 1
            buf = []
            buf_start = cur + 1
        cur += 1
    if buf:
        chunks.append(Chunk(
            id=f"{repo_path}::{qualified_name}::part{part}::{buf_start}-{cur - 1}",
            repo_path=repo_path, language=language, kind=kind, name=name,
            qualified_name=qualified_name, start_line=buf_start, end_line=cur - 1,
            code="\n".join(buf), imports=imports,
            parent_qualified_name=parent_qualified_name,
            metadata={"oversized": True, "part": part},
        ))
    return chunks