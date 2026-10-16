"""Stage 0: Connect GitHub repo — shallow clone into a scratch directory."""
import subprocess
import tempfile
import shutil
import re
import time
from pathlib import Path

from .obs import get_logger, log

LOG = get_logger("clone")


class CloneError(Exception):
    pass


def normalize_repo_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise CloneError("Please provide a GitHub repo URL.")
    # allow "owner/repo" shorthand
    if re.match(r"^[\w.-]+/[\w.-]+$", url):
        return f"https://github.com/{url}.git"
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split("git@github.com:", 1)[1]
    if not url.endswith(".git"):
        url = url.rstrip("/") + ".git"
    return url


def clone_repo(url: str, ctx=None) -> Path:
    """Shallow-clone a public repo into a fresh temp dir. Returns the path."""
    repo_url = normalize_repo_url(url)
    dest = Path(tempfile.mkdtemp(prefix="peach_repo_"))
    log(LOG, "info", "cloning", url=repo_url, dest=str(dest))
    t0 = time.time()
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", repo_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        log(LOG, "error", "clone timed out", url=repo_url, timeout_seconds=120)
        raise CloneError("Cloning timed out. Is the repo public and reachable?")
    except OSError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        log(LOG, "error", "could not run git", error=str(exc))
        raise CloneError(f"Could not run git: {exc}")

    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        stderr = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
        log(LOG, "error", "git clone failed", url=repo_url,
            returncode=result.returncode, stderr=stderr)
        raise CloneError(f"git clone failed: {stderr}")

    elapsed = round(time.time() - t0, 2)
    try:
        file_count = sum(1 for p in dest.rglob("*") if p.is_file())
        bytes_on_disk = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    except OSError:
        file_count, bytes_on_disk = -1, -1
    log(LOG, "info", "clone complete", seconds=elapsed, files=file_count,
        size_kb=bytes_on_disk // 1024 if bytes_on_disk >= 0 else -1)
    if ctx is not None:
        ctx.bump("clone.files_on_disk", max(file_count, 0))
    return dest
