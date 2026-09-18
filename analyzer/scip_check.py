"""
Stage 2 (real): SCIP symbol resolution.

`resolve_calls` in resolver.py used to decide "does this call resolve to
something defined in the repo?" by matching the call's bare tail name
(the last dotted segment) against every definition in the repo with that
name — no scopes, no imports, no type information. That's a workaround
with a real, named failure mode the resolver's own instrumentation
already flagged (`resolve.internal.matched_on_tail_of_dotted`): a
repo-local function named `post` silently swallows every `requests.post`
call in the codebase as "internal", because the two share nothing but a
bare name.

This module shells out to the real SCIP toolchain
(https://github.com/sourcegraph/scip) — the actual code-intelligence
indexers the "SCIP symbol resolution" stage in the project write-up is
named after — to get real, scope- and type-aware resolution instead:

  1. A per-language SCIP indexer (`scip-python`, `scip-typescript`,
     `scip-go`, `scip-java`, or `rust-analyzer scip`, whichever applies
     and is installed) builds a `.scip` index for the repo. These wrap a
     real language server/compiler frontend (pyright, tsserver, etc.), so
     "internal" here means "the language's own tooling resolved this
     call to a definition", not "these two strings match".
  2. The `scip` CLI (`scip print --json`) converts that binary protobuf
     index to JSON so this stays a subprocess + `json.loads` integration
     — the same shape as the Joern stage — rather than vendoring
     generated protobuf bindings for a schema that isn't ours.
  3. Every occurrence that *references* a symbol which has a
     *definition*-role occurrence somewhere else in the same index is a
     real, resolved internal call.

SCIP is real code intelligence with a real cost: indexers need to be
installed, and (at the time of writing) there's no mainstream indexer for
every language this pipeline parses — Ruby, PHP and C# have none. So this
degrades per call site rather than per run: `resolve_calls` tries SCIP
first and only drops to the bare tail-name heuristic for a given call
when SCIP doesn't cover it (unsupported language, indexer not installed,
indexing failed, or that specific reference just isn't in the index) —
the same "degrade a level, don't go silent" pattern the rest of this
pipeline follows.
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from .obs import get_logger, log

LOG = get_logger("scip_check")

INDEX_TIMEOUT = int(os.environ.get("PEACH_SCIP_INDEX_TIMEOUT", "300"))
PRINT_TIMEOUT = int(os.environ.get("PEACH_SCIP_PRINT_TIMEOUT", "60"))

_ROLE_DEFINITION = 0x1  # scip.proto SymbolRole.Definition bit

# Real, named SCIP indexers, one per language we know an indexer for.
# Command shapes are best-effort — flags differ across tool versions —
# and every invocation is wrapped in _build_index so a wrong flag or a
# missing binary degrades *that one language* to the fallback rather than
# aborting the run.
_SCIP_INDEXERS = {
    "python": ("scip-python", lambda root, out: [
        "scip-python", "index", str(root), "--output", str(out),
    ]),
    "javascript": ("scip-typescript", lambda root, out: [
        "scip-typescript", "index", str(root), "--output", str(out),
    ]),
    "typescript": ("scip-typescript", lambda root, out: [
        "scip-typescript", "index", str(root), "--output", str(out),
    ]),
    "go": ("scip-go", lambda root, out: [
        "scip-go", "--output", str(out), str(root),
    ]),
    "rust": ("rust-analyzer", lambda root, out: [
        "rust-analyzer", "scip", str(root), "--output", str(out),
    ]),
    "java": ("scip-java", lambda root, out: [
        "scip-java", "index", "--output", str(out), "--cwd", str(root),
    ]),
}

_lock = threading.Lock()
_bin_cache = {}
_scip_print_available = None  # None = unchecked, True/False once probed


def _find_binary(name):
    if name in _bin_cache:
        return _bin_cache[name]
    scip_home = os.environ.get("SCIP_HOME")
    found = None
    if scip_home:
        candidate = Path(scip_home) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found = str(candidate)
    if not found:
        found = shutil.which(name)
    with _lock:
        _bin_cache[name] = found
    return found


def _check_scip_print():
    """Memoized: `scip print` is what every language's index gets read
    through, so it's worth probing once and remembering rather than
    re-`which`-ing it per language per run."""
    global _scip_print_available
    if _scip_print_available is None:
        _scip_print_available = _find_binary("scip") is not None
    return _scip_print_available


def _build_index(language, repo_root: Path, workdir: Path):
    spec = _SCIP_INDEXERS.get(language)
    if not spec:
        return None, f"no SCIP indexer wired for language {language!r}"
    bin_name, build_cmd = spec
    if not _find_binary(bin_name):
        return None, f"{bin_name} not found on $PATH/$SCIP_HOME"

    out_path = workdir / f"index-{language}.scip"
    cmd = build_cmd(repo_root, out_path)
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=INDEX_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"{bin_name} timed out after {INDEX_TIMEOUT}s"
    except OSError as exc:
        return None, f"failed to launch {bin_name}: {exc}"
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1][:200] if tail else "no output"
        return None, f"{bin_name} failed (exit {proc.returncode}): {detail}"
    return out_path, None


def _print_index_json(scip_path: Path):
    """`scip print --json` on a binary index. Tolerates either a single
    JSON `Index` object (`{"documents": [...]}`) or NDJSON of `Document`
    objects, since which one a given `scip` build emits isn't something
    this integration controls."""
    scip_bin = _find_binary("scip")
    try:
        proc = subprocess.run(
            [scip_bin, "print", "--json", str(scip_path)],
            capture_output=True, text=True, timeout=PRINT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"`scip print` timed out after {PRINT_TIMEOUT}s"
    except OSError as exc:
        return None, f"failed to launch scip: {exc}"
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = (proc.stderr or "").strip().splitlines()
        detail = tail[-1][:200] if tail else "no output"
        return None, f"`scip print` failed (exit {proc.returncode}): {detail}"

    text = proc.stdout.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "documents" in data:
            return data, None
        if isinstance(data, list):
            return {"documents": data}, None
    except json.JSONDecodeError:
        pass

    docs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError:
            return None, "could not parse `scip print --json` output as JSON or NDJSON"
    if not docs:
        return None, "`scip print --json` produced no parseable documents"
    return {"documents": docs}, None


def _occurrence_line(rng):
    """SCIP ranges are 0-indexed [startLine, startChar, endLine, endChar]
    or the compact same-line form [startLine, startChar, endChar]. Only
    the start line is needed to line up with our 1-indexed
    CallSite/Definition line numbers, so +1 is all the conversion is."""
    if not rng:
        return None
    return rng[0] + 1


def _merge_reference_map(index_data, ref_map, def_locations):
    documents = index_data.get("documents") or []

    # pass 1: any symbol with a Definition-role occurrence anywhere in
    # this index is "local" — defined inside the repo we indexed, as
    # opposed to a symbol the index only *references* from an external
    # dependency (those never get a Definition occurrence in our index).
    for doc in documents:
        rel_path = doc.get("relativePath") or doc.get("relative_path")
        if not rel_path:
            continue
        for occ in doc.get("occurrences") or []:
            roles = occ.get("symbolRoles") or occ.get("symbol_roles") or 0
            if roles & _ROLE_DEFINITION:
                line = _occurrence_line(occ.get("range"))
                symbol = occ.get("symbol")
                if symbol and line is not None:
                    def_locations[symbol] = (rel_path, line)

    # pass 2: every reference occurrence whose symbol got a definition in
    # pass 1 is a resolved internal call site.
    for doc in documents:
        rel_path = doc.get("relativePath") or doc.get("relative_path")
        if not rel_path:
            continue
        for occ in doc.get("occurrences") or []:
            roles = occ.get("symbolRoles") or occ.get("symbol_roles") or 0
            if roles & _ROLE_DEFINITION:
                continue
            symbol = occ.get("symbol")
            if symbol in def_locations:
                line = _occurrence_line(occ.get("range"))
                if line is not None:
                    ref_map[(rel_path, line)] = def_locations[symbol]


def build_reference_map(records, repo_root, ctx=None):
    """Runs whichever real SCIP indexers apply to the languages actually
    present in `records`, merges their output, and returns
    `(ref_map, status)`:

    - `ref_map`: `{(file, line): (def_file, def_line)}` for every call
      site SCIP resolved to a repo-local definition.
    - `status`: which languages got indexed and which were skipped (and
      why), so the pipeline can say *why* SCIP resolved nothing instead
      of quietly falling back to the tail-name heuristic for everything.
    """
    languages = sorted({r.language for r in records if not r.is_config})
    status = {"indexed": [], "skipped": {}, "scip_print_available": _check_scip_print()}
    ref_map = {}

    def _finish():
        # Single exit point so every return path — including the two
        # early-outs below — reports through `ctx` the same way. This
        # used to be duplicated at the bottom only, which meant "no scip
        # binary" and "no indexer for any present language" silently
        # skipped the ctx.bump/ctx.note reporting entirely.
        if ctx is not None:
            ctx.bump("resolve.scip.languages_indexed", len(status["indexed"]))
            ctx.bump("resolve.scip.languages_skipped", len(status["skipped"]))
            ctx.bump("resolve.scip.references", len(ref_map))
            if status["skipped"]:
                ctx.note("info", "resolve",
                         "SCIP resolution unavailable for: " +
                         ", ".join(f"{lang} ({reason})" for lang, reason in status["skipped"].items()) +
                         ". Those calls use the bare tail-name fallback instead.")
        return ref_map, status

    if not status["scip_print_available"]:
        status["skipped"]["*"] = (
            "`scip` CLI not found on $PATH/$SCIP_HOME (needed to read any "
            "index); install it from https://github.com/sourcegraph/scip"
        )
        return _finish()

    indexable = [l for l in languages if l in _SCIP_INDEXERS]
    if not indexable:
        status["skipped"]["*"] = (
            f"no SCIP indexer wired for any language present ({', '.join(languages) or 'none'})"
        )
        return _finish()

    workdir = Path(tempfile.mkdtemp(prefix="peach_scip_"))
    def_locations = {}
    try:
        for language in indexable:
            scip_path, err = _build_index(language, Path(repo_root), workdir)
            if err:
                status["skipped"][language] = err
                log(LOG, "warning", "SCIP indexing unavailable for language",
                    language=language, reason=err)
                continue

            index_data, err = _print_index_json(scip_path)
            if err:
                status["skipped"][language] = err
                log(LOG, "warning", "could not read SCIP index", language=language, reason=err)
                continue

            before = len(ref_map)
            _merge_reference_map(index_data, ref_map, def_locations)
            status["indexed"].append(language)
            log(LOG, "info", "SCIP index merged", language=language,
                new_references=len(ref_map) - before)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return _finish()