"""Stage 0: Connect GitHub repo — shallow clone into a scratch directory or a per-job workspace.

Two entry points:

* ``clone_repo(url, ctx)`` — legacy: clone into a fresh temp dir, return its ``Path``.
* ``clone_repo_to_workspace(url, job_id, workspaces_root, ctx)`` — multi-user jobs: clone into
  ``{workspaces_root}/{job_id}/repo``, create ``{job_id}/artifacts/`` and write
  ``artifacts/repo_handle.json`` (a serialised :class:`RepoHandle`) for downstream stages and
  the vector branch.

Both share the same hardening: https://github.com only, locked-down git invocation, a
post-clone size limit, and removal of symlinks that point outside the repository.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .obs import get_logger, log

LOG = get_logger("clone")

CLONE_TIMEOUT_SECONDS = 120
GIT_QUERY_TIMEOUT_SECONDS = 30
DEFAULT_MAX_REPO_BYTES = 500 * 1024 * 1024
MAX_REPO_BYTES_ENV = "PEACH_MAX_REPO_BYTES"

JOB_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
_JOB_ID_RE = re.compile(JOB_ID_PATTERN)
REPO_DIRNAME = "repo"
ARTIFACTS_DIRNAME = "artifacts"
HANDLE_FILENAME = "repo_handle.json"

_SHORTHAND_RE = re.compile(r"[\w.-]+/[\w.-]+")
_SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_BAD_URL_CHARS_RE = re.compile(r"[\x00-\x20\x7f\\]")


class CloneError(Exception):
    pass


# --------------------------------------------------------------------------- #
# RepoHandle
# --------------------------------------------------------------------------- #
@dataclass
class RepoHandle:
    """Everything a later stage (or the vector branch) needs to find a cloned repo.

    ``root`` is the absolute path of the checkout, ``commit_sha`` the commit HEAD pointed at
    after cloning, ``url`` the normalised clone URL, ``is_shallow`` whether history is
    truncated, ``default_branch`` the checked-out branch ("" if unknown / detached) and
    ``job_id`` the workspace job this handle belongs to.
    """

    root: Path
    commit_sha: str
    url: str
    is_shallow: bool
    default_branch: str
    job_id: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def to_json(self) -> str:
        """Serialise to a JSON string (``root`` becomes a plain string)."""
        return json.dumps(
            {
                "root": str(self.root),
                "commit_sha": self.commit_sha,
                "url": self.url,
                "is_shallow": self.is_shallow,
                "default_branch": self.default_branch,
                "job_id": self.job_id,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: "str | bytes") -> "RepoHandle":
        """Rebuild a handle from :meth:`to_json` output. Raises ``ValueError`` if malformed."""
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RepoHandle JSON is not valid: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("RepoHandle JSON must be an object")
        for key in ("root", "commit_sha", "url", "job_id"):
            if not isinstance(data.get(key), str):
                raise ValueError(f"RepoHandle JSON needs a string field {key!r}")
        if not isinstance(data.get("is_shallow"), bool):
            raise ValueError("RepoHandle JSON needs a boolean field 'is_shallow'")
        branch = data.get("default_branch", "")
        if not isinstance(branch, str):
            raise ValueError("RepoHandle field 'default_branch' must be a string")
        return cls(
            root=Path(data["root"]),
            commit_sha=data["commit_sha"],
            url=data["url"],
            is_shallow=data["is_shallow"],
            default_branch=branch,
            job_id=data["job_id"],
        )


# --------------------------------------------------------------------------- #
# URL handling
# --------------------------------------------------------------------------- #
def _require_github_https(url: str) -> None:
    """Raise ``CloneError`` unless ``url`` is plain https on host github.com."""
    msg = "Only https://github.com repositories are supported."
    if _BAD_URL_CHARS_RE.search(url):
        raise CloneError(msg)
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise CloneError(msg) from None
    if (
        parts.scheme.lower() != "https"
        or host != "github.com"
        or parts.username is not None
        or parts.password is not None
        or port not in (None, 443)
    ):
        raise CloneError(msg)


def normalize_repo_url(url: str) -> str:
    """Normalise ``url`` to ``https://github.com/...git``; reject anything else.

    Accepts ``owner/repo`` shorthand, ``git@github.com:owner/repo`` and https URLs.
    Any other scheme or host (``file:``, ``ext::``, ``ssh:``, other hosts, userinfo,
    non-default ports) raises ``CloneError``.
    """
    url = url.strip()
    if not url:
        raise CloneError("Please provide a GitHub repo URL.")
    # allow "owner/repo" shorthand
    if _SHORTHAND_RE.fullmatch(url):
        owner, repo = url.split("/")
        if owner in (".", "..") or repo in (".", ".."):
            raise CloneError("Only https://github.com repositories are supported.")
        return f"https://github.com/{url}.git"
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split("git@github.com:", 1)[1]
    if not url.endswith(".git"):
        url = url.rstrip("/") + ".git"
    _require_github_https(url)
    return url


# --------------------------------------------------------------------------- #
# git invocation
# --------------------------------------------------------------------------- #
def _hardening() -> "tuple[list[str], dict[str, str]]":
    """Return ``(git -c options, environment)`` used for every git call."""
    config = ["-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
    env = dict(os.environ)
    env["GIT_ALLOW_PROTOCOL"] = "https"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return config, env


def _clone_source(repo_url: str) -> str:
    """Map the validated URL to what is passed to ``git clone`` (identity; a test seam)."""
    return repo_url


def _run_git(args: "list[str]", *, cwd: "Path | None" = None, timeout: int):
    config, env = _hardening()
    return subprocess.run(
        ["git", *config, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def _clone_into(repo_url: str, dest: Path) -> float:
    """Run the shallow clone into ``dest``. Returns elapsed seconds. Does not clean up."""
    log(LOG, "info", "cloning", url=repo_url, dest=str(dest))
    t0 = time.time()
    try:
        result = _run_git(
            ["clone", "--depth", "1", "--single-branch", "--",
             _clone_source(repo_url), str(dest)],
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log(LOG, "error", "clone timed out", url=repo_url,
            timeout_seconds=CLONE_TIMEOUT_SECONDS)
        raise CloneError("Cloning timed out. Is the repo public and reachable?") from None
    except OSError as exc:
        log(LOG, "error", "could not run git", error=str(exc))
        raise CloneError(f"Could not run git: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
        log(LOG, "error", "git clone failed", url=repo_url,
            returncode=result.returncode, stderr=stderr)
        raise CloneError(f"git clone failed: {stderr}")
    return round(time.time() - t0, 2)


def _git_head_sha(repo: Path) -> str:
    """Return the full commit SHA of HEAD; any failure is a ``CloneError``."""
    try:
        result = _run_git(["rev-parse", "HEAD"], cwd=repo, timeout=GIT_QUERY_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log(LOG, "error", "could not read commit sha", error=str(exc))
        raise CloneError(f"Could not determine the cloned commit SHA: {exc}") from exc
    sha = (result.stdout or "").strip()
    if result.returncode != 0 or not _SHA_RE.fullmatch(sha):
        detail = (result.stderr or "").strip().splitlines()[-1:] or ["unexpected output"]
        log(LOG, "error", "could not read commit sha", returncode=result.returncode,
            detail=detail[0])
        raise CloneError(f"Could not determine the cloned commit SHA: {detail[0]}")
    return sha


def _git_default_branch(repo: Path) -> str:
    """Branch HEAD points at, or "" if unknown (detached HEAD, git failure)."""
    try:
        result = _run_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd=repo,
                          timeout=GIT_QUERY_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _is_shallow(repo: Path) -> bool:
    return (repo / ".git" / "shallow").is_file()


# --------------------------------------------------------------------------- #
# Post-clone checks
# --------------------------------------------------------------------------- #
def _max_repo_bytes() -> int:
    """Size limit from ``PEACH_MAX_REPO_BYTES`` (default 500 MB; bad values fall back)."""
    raw = os.environ.get(MAX_REPO_BYTES_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_REPO_BYTES
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        log(LOG, "warning", "ignoring invalid size limit", env=MAX_REPO_BYTES_ENV, value=raw)
        return DEFAULT_MAX_REPO_BYTES
    return value


def _link_stays_inside(link: str, real_root: str) -> bool:
    target = os.path.realpath(link)
    try:
        return os.path.commonpath([real_root, target]) == real_root
    except ValueError:
        return False


def _scan_tree(root: Path) -> "tuple[int, int, int]":
    """Walk ``root`` without following symlinks.

    Deletes symlinks whose resolved target lies outside ``root`` and counts the regular
    files and bytes that remain. Returns ``(file_count, total_bytes, removed_symlinks)``.
    """
    real_root = os.path.realpath(root)
    files = total = removed = 0

    def drop_if_escaping(full: str) -> bool:
        """True if ``full`` is a symlink (removed when it escapes the repo)."""
        nonlocal removed
        if not os.path.islink(full):
            return False
        if not _link_stays_inside(full, real_root):
            try:
                os.unlink(full)
            except OSError as exc:
                raise CloneError(f"Could not remove unsafe symlink {full}: {exc}") from exc
            removed += 1
        return True

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            drop_if_escaping(os.path.join(dirpath, name))
        for name in filenames:
            full = os.path.join(dirpath, name)
            if drop_if_escaping(full):
                continue
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                files += 1
                total += st.st_size
    return files, total, removed


def _finalize_clone(dest: Path, ctx, elapsed: float) -> None:
    """Sweep symlinks, enforce the size limit, log and bump counters."""
    file_count, bytes_on_disk, removed = _scan_tree(dest)
    if removed:
        log(LOG, "warning", "removed symlinks pointing outside the repo", count=removed)
    limit = _max_repo_bytes()
    if bytes_on_disk > limit:
        log(LOG, "error", "repo too large", size_bytes=bytes_on_disk, limit_bytes=limit)
        raise CloneError(
            f"Repository is too large: {bytes_on_disk} bytes on disk exceeds the limit of "
            f"{limit} bytes (set {MAX_REPO_BYTES_ENV} to change it)."
        )
    log(LOG, "info", "clone complete", seconds=elapsed, files=file_count,
        size_kb=bytes_on_disk // 1024)
    if ctx is not None:
        ctx.bump("clone.files_on_disk", max(file_count, 0))


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _rmtree(path: Path) -> None:
    """Remove ``path`` if present; never raises. Safe to call repeatedly."""
    path = Path(path)
    if not os.path.lexists(path):
        return
    try:
        if path.is_symlink():
            path.unlink()
            return
    except OSError:
        return

    def _retry(func, target, _exc) -> None:
        try:
            os.chmod(target, 0o700)
            func(target)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry)
    else:
        shutil.rmtree(path, onerror=_retry)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise CloneError(f"Invalid job_id: must match {JOB_ID_PATTERN}")
    return job_id


def _workspace_paths(workspaces_root, job_id: str) -> "tuple[Path, Path, Path]":
    """Return ``(job_dir, repo_dir, artifacts_dir)``; refuse a job dir that escapes the root."""
    root = Path(workspaces_root)
    job_dir = root / job_id
    if job_dir.resolve().parent != root.resolve():
        raise CloneError("Invalid workspace: job directory resolves outside the workspaces root.")
    return job_dir, job_dir / REPO_DIRNAME, job_dir / ARTIFACTS_DIRNAME


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def clone_repo(url: str, ctx=None) -> Path:
    """Shallow-clone a public repo into a fresh temp dir. Returns the path."""
    repo_url = normalize_repo_url(url)
    dest = Path(tempfile.mkdtemp(prefix="peach_repo_"))
    try:
        elapsed = _clone_into(repo_url, dest)
        _finalize_clone(dest, ctx, elapsed)
    except BaseException:
        _rmtree(dest)
        raise
    return dest


def clone_repo_to_workspace(url: str, job_id: str, workspaces_root, ctx=None) -> RepoHandle:
    """Clone into ``{workspaces_root}/{job_id}/repo`` and describe it with a :class:`RepoHandle`.

    Creates ``{job_id}/artifacts/`` and writes ``artifacts/repo_handle.json``. ``job_id`` must
    match ``JOB_ID_PATTERN``. If a ``repo/`` already exists for the job a ``CloneError`` is
    raised and nothing is touched. On any failure what this call created is removed (the whole
    job dir if this call created it; otherwise only ``repo/`` and the handle file, so artifacts
    from earlier stages are not destroyed).
    """
    job_id = _validate_job_id(job_id)
    repo_url = normalize_repo_url(url)
    job_dir, repo_dir, artifacts_dir = _workspace_paths(workspaces_root, job_id)

    created_job_dir = False
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
        created_job_dir = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise CloneError(f"Could not create workspace: {exc}") from exc
    if os.path.lexists(repo_dir):
        raise CloneError(f"Workspace for job {job_id} already contains a cloned repo.")

    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        elapsed = _clone_into(repo_url, repo_dir)
        _finalize_clone(repo_dir, ctx, elapsed)
        handle = RepoHandle(
            root=repo_dir.resolve(),
            commit_sha=_git_head_sha(repo_dir),
            url=repo_url,
            is_shallow=_is_shallow(repo_dir),
            default_branch=_git_default_branch(repo_dir),
            job_id=job_id,
        )
        _write_text_atomic(artifacts_dir / HANDLE_FILENAME, handle.to_json())
    except BaseException as exc:
        if created_job_dir:
            _rmtree(job_dir)
        else:
            _rmtree(repo_dir)
            try:
                (artifacts_dir / HANDLE_FILENAME).unlink()
            except OSError:
                pass
        if isinstance(exc, OSError):
            raise CloneError(f"Workspace error: {exc}") from exc
        raise
    log(LOG, "info", "workspace ready", job_id=job_id, commit=handle.commit_sha,
        shallow=handle.is_shallow, branch=handle.default_branch)
    return handle


def cleanup_workspace(workspaces_root, job_id: str, keep_artifacts: bool = True) -> None:
    """Delete ``repo/`` for the job; also delete ``artifacts/`` if ``keep_artifacts`` is False.

    Idempotent: calling it again, or for a job that never existed, is a no-op.
    Raises ``CloneError`` only for an invalid ``job_id``.
    """
    job_id = _validate_job_id(job_id)
    job_dir, repo_dir, artifacts_dir = _workspace_paths(workspaces_root, job_id)
    _rmtree(repo_dir)
    if not keep_artifacts:
        _rmtree(artifacts_dir)
        try:
            job_dir.rmdir()
        except OSError:
            pass
    log(LOG, "info", "workspace cleaned", job_id=job_id, kept_artifacts=keep_artifacts)