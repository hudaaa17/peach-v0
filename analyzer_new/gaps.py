"""
Gap report: a structured, pipeline-wide list of "what was NOT analysed, and why".

Every stage appends `Gap` entries here instead of (only) writing free-text
log lines or `ctx.note` sentences. The graph builder and the Peach UI read the
one list, so a user can see exactly which parts of the repo are missing from the
graph and what the consequence is.

A gap never means "nothing found". It means "unknown": a file that was skipped
or only roughly parsed must never be reported as "analysed, no calls found".

How the list travels
--------------------
Every stage that can produce gaps takes an optional `gaps=` keyword (a plain
list). If it is omitted, the stage uses `ctx.gaps` (created on first use), so a
pipeline that already passes `ctx` everywhere needs no further wiring. If both
are absent the gaps are dropped (they are still logged by the stage).

    gaps = []
    records = parse_repo(root, ctx=ctx, gaps=gaps)
    edges   = resolve_calls(records, ctx=ctx, repo_root=root, gaps=gaps)
    ui_payload = gaps_to_dicts(gaps)

The vector branch runs separately and talks to the main pipeline only through
files in workspaces/{job_id}/artifacts/. Its gaps travel the same way:

    write_gaps(vector_gaps, artifacts_dir / "vector_gaps.json")   # vector branch
    all_gaps = merge_gaps(gaps, read_gaps(artifacts_dir / "vector_gaps.json"))

Vocabulary (keep these stable - the UI groups on `kind`)
--------------------------------------------------------
stage  : "parse" | "scip" | "ast_grep" | "config" | "joern" | "llm" | "vector" | "graph"
scope  : "file"     one file                         (`path` set)
         "language" every file of one language      (`path` empty)
         "repo"     part of the repo as a whole      (`path` empty)
kind   : parse : too_large, minified, read_error, stat_error, parse_exception,
                 parse_fallback, syntax_errors, file_cap, vendored_dir
         scip  : scip_language_failed, scip_not_covered, internal_unmapped
         vector: boundary_conflict, scip_symbol_dropped, file_unreadable,
                 file_stats_unavailable,
                 chunk_fallback_boundaries, chunk_oversize_split, chunk_skipped,
                 chunk_size_split_only, embed_failed, embed_partial,
                 store_failed, branch_failed
         graph : branch_failed   (the whole graph branch raised; pipeline_job.py)
"""
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class Gap:
    stage: str                  # which stage found it
    scope: str                  # "file" | "language" | "repo"
    kind: str                   # category, see module docstring
    path: str = ""              # repo-relative POSIX path; "" for language/repo scope
    detail: str = ""            # human-readable explanation, shown in the UI
    impact: str = ""            # what the user loses because of this gap
    count: int = 1              # how many items this entry stands for (calls, files...)
    samples: Optional[list] = None   # a few example paths for aggregated entries


def gap_sink(ctx=None, gaps=None):
    """The list gaps should be appended to, or None if there is nowhere to put
    them. An explicit `gaps` list wins; otherwise `ctx.gaps` is used and is
    created on first use. Never raises."""
    if gaps is not None:
        return gaps
    if ctx is None:
        return None
    existing = getattr(ctx, "gaps", None)
    if existing is not None:
        return existing
    try:
        ctx.gaps = []
    except Exception:  # noqa: BLE001 - a ctx that refuses attributes just gets no gaps
        return None
    return ctx.gaps


def emit(sink, gap: Gap) -> None:
    """Append one gap if there is somewhere to put it."""
    if sink is not None:
        sink.append(gap)


def gaps_to_dicts(gaps) -> list:
    """JSON-ready form for the API/UI layer."""
    return [asdict(g) for g in (gaps or [])]


def summarize_gaps(gaps) -> dict:
    """{kind: total count} - a one-glance summary, e.g. for a UI header badge."""
    out = {}
    for g in gaps or []:
        out[g.kind] = out.get(g.kind, 0) + g.count
    return out


# ---------------------------------------------------------------------------
# Merging and persistence (used to carry gaps across the file boundary between
# the main pipeline and the vector branch)
# ---------------------------------------------------------------------------

_GAP_FIELDS = {f.name for f in fields(Gap)}
_GAP_REQUIRED = {"stage", "scope", "kind"}


def merge_gaps(*gap_lists) -> list:
    """Concatenate gap lists in order, dropping exact duplicates.

    Two gaps are duplicates when stage, scope, kind, path and detail are all
    equal; the first one seen wins (its impact/count/samples are kept, later
    ones are not summed in). `None` arguments are treated as empty lists.
    Returns a new list; the inputs are not modified."""
    seen = set()
    merged = []
    for gap_list in gap_lists:
        for g in gap_list or []:
            key = (g.stage, g.scope, g.kind, g.path, g.detail)
            if key in seen:
                continue
            seen.add(key)
            merged.append(g)
    return merged


def gaps_from_dicts(list_of_dicts) -> list:
    """Inverse of `gaps_to_dicts`. Unknown keys are ignored so a newer writer
    never breaks an older reader.

    Raises ValueError for an entry that is not a dict or lacks stage/scope/kind:
    guessing those would turn a gap into a wrong statement, so the caller
    decides what to do with a damaged report."""
    out = []
    for i, d in enumerate(list_of_dicts or []):
        if not isinstance(d, dict):
            raise ValueError(f"gap entry {i} is not an object: {type(d).__name__}")
        missing = _GAP_REQUIRED - d.keys()
        if missing:
            raise ValueError(f"gap entry {i} is missing {sorted(missing)}")
        out.append(Gap(**{k: v for k, v in d.items() if k in _GAP_FIELDS}))
    return out


def write_gaps(gaps, path) -> None:
    """Write gaps to `path` as UTF-8 JSON (a list of dicts, same shape as
    `gaps_to_dicts`), atomically: the data goes to a temp file in the same
    directory and is renamed over `path`. A failure at any point leaves any
    existing file at `path` untouched, never a partial one, and removes the
    temp file. Creates missing parent directories. Re-raises the original
    error; callers that must not raise (stages) should catch it and record a
    Gap themselves."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = gaps_to_dicts(gaps)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_gaps(path) -> list:
    """Read a file written by `write_gaps` and return list[Gap].

    Raises FileNotFoundError if the file is missing and ValueError if it is
    not valid JSON, not a list, or holds malformed entries. A missing or
    damaged report means "unknown", so it is never silently turned into an
    empty list."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gap file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"gap file {path} must hold a JSON list")
    return gaps_from_dicts(data)