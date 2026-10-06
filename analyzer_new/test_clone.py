"""Tests for Stage 0 (clone.py). No network: a local bare git repo stands in for GitHub.

Run from the project root with:  python -m pytest analyzer_new/test_clone.py
(``analyzer_new`` needs an ``__init__.py`` so the relative imports in clone.py resolve).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from analyzer_new import clone
from analyzer_new.clone import (
    CloneError,
    RepoHandle,
    cleanup_workspace,
    clone_repo,
    clone_repo_to_workspace,
    normalize_repo_url,
)

REAL_RUN = subprocess.run


def _can_symlink() -> bool:
    d = Path(tempfile.mkdtemp())
    try:
        os.symlink(str(d), str(d / "l"))
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


needs_symlinks = pytest.mark.skipif(not _can_symlink(), reason="symlinks not permitted here")
JOB = "job_12345678"
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(*args, cwd=None):
    out = REAL_RUN(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, env=GIT_ENV,
                   capture_output=True, text=True, check=True)
    return out.stdout.strip()


def make_bare(tmp_path, name, populate):
    work = tmp_path / f"{name}_work"
    work.mkdir()
    git("init", "-q", "-b", "main", cwd=work)
    populate(work)
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "init", cwd=work)
    bare = tmp_path / f"{name}.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    return bare, git("rev-parse", "HEAD", cwd=work)


def _plain(work):
    (work / "README.md").write_text("hello\n")
    (work / "src").mkdir()
    (work / "src" / "a.py").write_text("print('a')\n")


@pytest.fixture
def source(tmp_path):
    bare, sha = make_bare(tmp_path, "plain", _plain)
    return bare, sha


@pytest.fixture
def local_git(monkeypatch, source):
    """Point the clone at the local bare repo and allow the file protocol for it."""
    bare, _ = source
    monkeypatch.setattr(clone, "_clone_source", lambda url: bare.as_uri())

    def hardening():
        env = dict(os.environ)
        env["GIT_ALLOW_PROTOCOL"] = "file"
        env["GIT_TERMINAL_PROMPT"] = "0"
        return ["-c", "core.hooksPath=/dev/null"], env

    monkeypatch.setattr(clone, "_hardening", hardening)
    return bare


class FakeCtx:
    def __init__(self):
        self.counts = {}

    def bump(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n


# ----------------------------------------------------------------- RepoHandle
def test_repohandle_json_roundtrip(tmp_path):
    h = RepoHandle(root=tmp_path / "repo", commit_sha="a" * 40, url="https://github.com/o/r.git",
                   is_shallow=True, default_branch="main", job_id=JOB)
    again = RepoHandle.from_json(h.to_json())
    assert again == h
    assert isinstance(again.root, Path)


def test_repohandle_unknown_branch_defaults_empty():
    text = ('{"root": "/x", "commit_sha": "abc", "url": "u", "is_shallow": false, '
            f'"job_id": "{JOB}"}}')
    assert RepoHandle.from_json(text).default_branch == ""


@pytest.mark.parametrize("bad", [
    "not json", "[]", '{"root": "/x"}',
    '{"root": "/x", "commit_sha": "a", "url": "u", "is_shallow": "yes", "job_id": "j"}',
])
def test_repohandle_from_json_rejects_malformed(bad):
    with pytest.raises(ValueError):
        RepoHandle.from_json(bad)


# ------------------------------------------------------- workspace clone
def test_workspace_layout_sha_and_handle_file(tmp_path, source, local_git):
    _, sha = source
    root = tmp_path / "workspaces"
    ctx = FakeCtx()
    handle = clone_repo_to_workspace("owner/repo", JOB, root, ctx=ctx)

    repo = root / JOB / "repo"
    assert (repo / "README.md").read_text() == "hello\n"
    assert (root / JOB / "artifacts").is_dir()
    assert handle.commit_sha == sha
    assert handle.root == repo.resolve()
    assert handle.url == "https://github.com/owner/repo.git"
    assert handle.job_id == JOB
    assert handle.is_shallow is True
    assert handle.default_branch == "main"
    assert ctx.counts["clone.files_on_disk"] > 0

    saved = RepoHandle.from_json((root / JOB / "artifacts" / "repo_handle.json").read_text())
    assert saved == handle


@pytest.mark.parametrize("bad", [
    "", "short", "../../etc/passwd", "a/b/c/d/e/f/g/h", "has space 12", "x" * 65,
    "valid_id_1\n", "..\\..\\evil1", "job_id_é_123", "/abs/path/12345",
])
def test_job_id_validation_rejects(tmp_path, local_git, bad):
    root = tmp_path / "workspaces"
    with pytest.raises(CloneError):
        clone_repo_to_workspace("owner/repo", bad, root)
    with pytest.raises(CloneError):
        cleanup_workspace(root, bad)
    assert not root.exists()


@pytest.mark.parametrize("good", ["abcdefgh", "A_b-1234", "x" * 64, "--------"])
def test_job_id_validation_accepts(good):
    assert clone._validate_job_id(good) == good


def test_existing_repo_dir_is_refused_and_untouched(tmp_path, local_git):
    root = tmp_path / "workspaces"
    (root / JOB / "repo").mkdir(parents=True)
    (root / JOB / "repo" / "keep.txt").write_text("mine")
    with pytest.raises(CloneError):
        clone_repo_to_workspace("owner/repo", JOB, root)
    assert (root / JOB / "repo" / "keep.txt").read_text() == "mine"


# ---------------------------------------------------------------- hosts
@pytest.mark.parametrize("bad", [
    "https://evil.com/x/y", "file:///etc", "ext::sh", "http://github.com/o/r",
    "https://github.com@evil.com/o/r", "https://evil.com@github.com/o/r",
    "ssh://git@github.com/o/r", "git@evil.com:o/r", "https://github.com.evil.com/o/r",
    "https://github.com:8443/o/r", "--upload-pack=touch /tmp/x", "../x", "./y",
    "https://github.com\\@evil.com/o/r", "https://github.com/o/r with space",
])
def test_non_github_https_rejected(tmp_path, monkeypatch, bad):
    def boom(*a, **k):
        raise AssertionError("git must not run for a rejected URL")

    monkeypatch.setattr(clone.subprocess, "run", boom)
    with pytest.raises(CloneError):
        normalize_repo_url(bad)
    with pytest.raises(CloneError):
        clone_repo(bad)
    root = tmp_path / "workspaces"
    with pytest.raises(CloneError):
        clone_repo_to_workspace(bad, JOB, root)
    assert not root.exists()


@pytest.mark.parametrize("given, expected", [
    ("owner/repo", "https://github.com/owner/repo.git"),
    ("https://github.com/o/r", "https://github.com/o/r.git"),
    ("https://github.com/o/r.git", "https://github.com/o/r.git"),
    ("https://github.com/o/r/", "https://github.com/o/r.git"),
    ("git@github.com:o/r.git", "https://github.com/o/r.git"),
    ("  git@github.com:o/r  ", "https://github.com/o/r.git"),
])
def test_valid_urls_normalised(given, expected):
    assert normalize_repo_url(given) == expected


def test_empty_url_rejected():
    with pytest.raises(CloneError):
        normalize_repo_url("   ")


# ---------------------------------------------------- git hardening
def test_git_invocation_is_hardened(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: nope")

    monkeypatch.setattr(clone.subprocess, "run", fake_run)
    with pytest.raises(CloneError, match="git clone failed: fatal: nope"):
        clone_repo("owner/repo")

    cmd, kw = calls[0]
    assert cmd[0] == "git"
    assert cmd[1:5] == ["-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
    assert cmd.index("clone") > 4
    assert "--depth" in cmd and cmd[cmd.index("--depth") + 1] == "1"
    assert cmd[cmd.index("--") + 1] == "https://github.com/owner/repo.git"
    assert kw["env"]["GIT_ALLOW_PROTOCOL"] == "https"
    assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert kw["timeout"] == 120
    assert not Path(cmd[-1]).exists()  # temp dir removed after failure


def test_timeout_cleans_job_dir(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(clone.subprocess, "run", fake_run)
    root = tmp_path / "workspaces"
    with pytest.raises(CloneError, match="timed out"):
        clone_repo_to_workspace("owner/repo", JOB, root)
    assert not (root / JOB).exists()
    assert root.exists()


def test_clone_failure_preserves_preexisting_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "workspaces"
    (root / JOB / "artifacts").mkdir(parents=True)
    other = root / JOB / "artifacts" / "other_stage.json"
    other.write_text("{}")
    monkeypatch.setattr(
        clone.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, "", "fatal: nope"))
    with pytest.raises(CloneError):
        clone_repo_to_workspace("owner/repo", JOB, root)
    assert other.read_text() == "{}"
    assert not (root / JOB / "repo").exists()


# --------------------------------------------------------------- SHA
def test_sha_failure_is_clone_error_and_cleans_up(tmp_path, local_git, monkeypatch):
    def fake_run(cmd, **kw):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: bad")
        return REAL_RUN(cmd, **kw)

    monkeypatch.setattr(clone.subprocess, "run", fake_run)
    root = tmp_path / "workspaces"
    with pytest.raises(CloneError, match="SHA"):
        clone_repo_to_workspace("owner/repo", JOB, root)
    assert not (root / JOB).exists()


def test_malformed_sha_output_is_clone_error(tmp_path, local_git, monkeypatch):
    def fake_run(cmd, **kw):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "not-a-sha\n", "")
        return REAL_RUN(cmd, **kw)

    monkeypatch.setattr(clone.subprocess, "run", fake_run)
    with pytest.raises(CloneError):
        clone_repo_to_workspace("owner/repo", JOB, tmp_path / "workspaces")


# ---------------------------------------------------------- symlinks
def _with_links(work):
    _plain(work)
    outside_file = work.parent / "secret.txt"
    outside_dir = work.parent / "outside_dir"
    outside_file.write_text("secret")
    outside_dir.mkdir()
    os.symlink(str(outside_file), work / "abs_file_escape")
    os.symlink(str(outside_dir), work / "abs_dir_escape")
    os.symlink("chain_b", work / "chain_a")
    os.symlink(str(outside_file), work / "chain_b")
    os.symlink("../..", work / "src" / "rel_escape")
    os.symlink("README.md", work / "inside_file_link")
    os.symlink("src", work / "inside_dir_link")


@needs_symlinks
def test_escaping_symlinks_removed_inside_ones_kept(tmp_path, monkeypatch):
    bare, _ = make_bare(tmp_path, "links", _with_links)
    monkeypatch.setattr(clone, "_clone_source", lambda url: bare.as_uri())
    monkeypatch.setattr(
        clone, "_hardening",
        lambda: (["-c", "core.hooksPath=/dev/null"],
                 {**os.environ, "GIT_ALLOW_PROTOCOL": "file", "GIT_TERMINAL_PROMPT": "0"}))

    root = tmp_path / "workspaces"
    handle = clone_repo_to_workspace("owner/repo", JOB, root)
    repo = handle.root
    for gone in ("abs_file_escape", "abs_dir_escape", "chain_a", "chain_b", "src/rel_escape"):
        assert not os.path.lexists(repo / gone), gone
    for kept in ("inside_file_link", "inside_dir_link", "README.md"):
        assert os.path.lexists(repo / kept), kept
    assert (tmp_path / "secret.txt").read_text() == "secret"  # targets untouched


@needs_symlinks
def test_scan_tree_does_not_count_symlinks(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "f.txt").write_text("12345")
    os.symlink("f.txt", root / "link")
    files, size, removed = clone._scan_tree(root)
    assert (files, size, removed) == (1, 5, 0)


# -------------------------------------------------------- size limit
def test_size_limit_deletes_clone(tmp_path, local_git, monkeypatch):
    monkeypatch.setenv("PEACH_MAX_REPO_BYTES", "10")
    root = tmp_path / "workspaces"
    with pytest.raises(CloneError, match="PEACH_MAX_REPO_BYTES"):
        clone_repo_to_workspace("owner/repo", JOB, root)
    assert not (root / JOB).exists()


def test_size_limit_applies_to_clone_repo(local_git, monkeypatch):
    monkeypatch.setenv("PEACH_MAX_REPO_BYTES", "10")
    made = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*a, **k):
        made.append(real_mkdtemp(*a, **k))
        return made[-1]

    monkeypatch.setattr(clone.tempfile, "mkdtemp", spy)
    with pytest.raises(CloneError, match="too large"):
        clone_repo("owner/repo")
    assert made and not Path(made[0]).exists()


def test_max_bytes_default_and_invalid_values(monkeypatch):
    monkeypatch.delenv("PEACH_MAX_REPO_BYTES", raising=False)
    assert clone._max_repo_bytes() == 500 * 1024 * 1024
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv("PEACH_MAX_REPO_BYTES", bad)
        assert clone._max_repo_bytes() == 500 * 1024 * 1024
    monkeypatch.setenv("PEACH_MAX_REPO_BYTES", "1234")
    assert clone._max_repo_bytes() == 1234


# ----------------------------------------------------------- cleanup
def test_cleanup_idempotent_and_keeps_artifacts(tmp_path, local_git):
    root = tmp_path / "workspaces"
    clone_repo_to_workspace("owner/repo", JOB, root)
    clone_repo_to_workspace("owner/repo", "other_job_1", root)

    cleanup_workspace(root, JOB)
    assert not (root / JOB / "repo").exists()
    assert (root / JOB / "artifacts" / "repo_handle.json").is_file()
    cleanup_workspace(root, JOB)  # second call is a no-op
    assert (root / JOB / "artifacts" / "repo_handle.json").is_file()

    cleanup_workspace(root, JOB, keep_artifacts=False)
    assert not (root / JOB).exists()
    cleanup_workspace(root, JOB, keep_artifacts=False)  # again, still fine

    assert (root / "other_job_1" / "repo" / "README.md").is_file()  # sibling untouched


def test_cleanup_unknown_job_is_noop(tmp_path):
    cleanup_workspace(tmp_path / "nope", "never_ran_1")
    cleanup_workspace(tmp_path / "nope", "never_ran_1", keep_artifacts=False)


def test_cleanup_handles_readonly_files(tmp_path, local_git):
    root = tmp_path / "workspaces"
    clone_repo_to_workspace("owner/repo", JOB, root)
    # git object files are read-only; removal must still succeed
    cleanup_workspace(root, JOB, keep_artifacts=False)
    assert not (root / JOB).exists()


def test_rerun_after_cleanup_works(tmp_path, local_git):
    root = tmp_path / "workspaces"
    clone_repo_to_workspace("owner/repo", JOB, root)
    cleanup_workspace(root, JOB)
    handle = clone_repo_to_workspace("owner/repo", JOB, root)
    assert (handle.root / "README.md").is_file()


# ------------------------------------------------- legacy clone_repo
def test_clone_repo_still_returns_fresh_temp_path(local_git):
    ctx = FakeCtx()
    dest = clone_repo("owner/repo", ctx)
    try:
        assert isinstance(dest, Path)
        assert dest.name.startswith("peach_repo_")
        assert (dest / "README.md").read_text() == "hello\n"
        assert ctx.counts["clone.files_on_disk"] > 0
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_clone_repo_returns_distinct_dirs(local_git):
    a, b = clone_repo("owner/repo"), clone_repo("owner/repo")
    try:
        assert a != b
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)