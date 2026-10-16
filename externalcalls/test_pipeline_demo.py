"""Structural subprocess tests for the real pipeline demo (no stage doubles).

Run these in the provisioned container. Host compilation alone does not verify
Tree-sitter, SCIP, ast-grep, Joern, or any pipeline result values.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOERN = Path("/opt/joern/joern-cli/joern")
STAGE_NAMES = (
    "clone", "parse_repo", "scip", "resolve_calls", "ast_grep",
    "literal", "constprop", "config", "joern_tasks", "slices",
    "joern_run", "joern_apply", "gaps",
)


@pytest.fixture
def source_folder(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "constants.py").write_text(
        'IMPORTED_URL = "https://imported.example/api"\n',
        encoding="utf-8", newline="\n")
    (source / "client.py").write_text(
        "import requests\n"
        "from constants import IMPORTED_URL\n"
        "\n"
        'requests.get(url="https://literal.example/api")\n'
        "requests.get(url=IMPORTED_URL)\n"
        "\n"
        "def send(url):\n"
        "    return requests.get(url=url)\n",
        encoding="utf-8", newline="\n")
    return source


def _run_demo(*arguments):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "externalcalls.pipeline_demo", *map(str, arguments)],
        cwd=PROJECT_ROOT, env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=2400)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic(result, log=""):
    return (f"subprocess exit={result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\npipeline.log:\n{log}")


def test_local_fixture_all_stages_and_artifacts(source_folder, tmp_path):
    """T1: stage order, readable artifacts, funnel/trace, and call/edge alignment."""
    out = tmp_path / "demo-output"
    result = _run_demo("--source", source_folder, "--out", out, "--max-joern-groups", 0)
    assert result.returncode == 0, _diagnostic(result)
    logfile = out / "pipeline.log"
    assert logfile.is_file(), _diagnostic(result)
    log = logfile.read_text(encoding="utf-8")
    diagnostic = _diagnostic(result, log)
    expected_headers = [(str(number), name) for number, name in enumerate(STAGE_NAMES, 1)]
    for text in (log, result.stdout):
        assert re.findall(r"^=== STAGE (\d+)/13 (\w+) ===$", text, re.MULTILINE) == expected_headers, diagnostic
        assert not re.search(r"^STATUS:\s*FAILED\b", text, re.MULTILINE), diagnostic
    stages = {}
    for number, name in enumerate(STAGE_NAMES, 1):
        path = out / f"stage{number:02d}_{name}.json"
        assert path.is_file(), diagnostic
        stages[number] = _load(path)
        assert isinstance(stages[number], dict), diagnostic
        assert stages[number]["stage"] == number, diagnostic
        assert stages[number]["status"] in {"OK", "SKIPPED"}, diagnostic
    assert (out / "summary.json").is_file(), diagnostic
    summary = _load(out / "summary.json")
    assert isinstance(summary["funnel"], dict), diagnostic
    assert isinstance(summary["timings"], list), diagnostic
    assert "funnel check" in log, diagnostic
    assert "Edge trace:" in log, diagnostic
    assert isinstance(_load(out / "edge_trace.json"), list), diagnostic
    assert isinstance(_load(out / "gaps.json"), list), diagnostic
    # Compare this run's actual parser output and resolver output. No host,
    # source, or measured call-count value is a golden expectation.
    calls_found = sum(len(record["calls"]) for record in stages[2]["data"]["records"])
    edges_found = len(stages[4]["edges"])
    assert stages[2]["outputs"]["calls"] == calls_found, diagnostic
    assert stages[4]["outputs"]["edges"] == edges_found == calls_found, diagnostic


@pytest.mark.skipif(not JOERN.is_file(), reason="real Joern CLI is missing: /opt/joern/joern-cli/joern")
def test_joern_and_keyword_probe_structure(source_folder, tmp_path):
    """T2: Joern/apply blocks and diagnostic probe artifact, with real tools."""
    out = tmp_path / "joern-output"
    result = _run_demo("--source", source_folder, "--out", out,
                       "--max-joern-groups", 1, "--probe-keyword")
    assert result.returncode == 0, _diagnostic(result)
    log = (out / "pipeline.log").read_text(encoding="utf-8")
    diagnostic = _diagnostic(result, log)
    assert "=== STAGE 11/13 joern_run ===" in log, diagnostic
    assert "=== STAGE 12/13 joern_apply ===" in log, diagnostic
    assert not re.search(r"^STATUS:\s*FAILED\b", log, re.MULTILINE), diagnostic
    stage11 = _load(out / "stage11_joern_run.json")
    stage12 = _load(out / "stage12_joern_apply.json")
    assert isinstance(stage11, dict) and isinstance(stage12, dict), diagnostic
    assert stage11["status"] in {"OK", "SKIPPED"}, diagnostic
    assert stage12["status"] in {"OK", "SKIPPED"}, diagnostic
    results = stage11["data"].get("results", [])
    assert isinstance(results, list), diagnostic
    if any(item["status"] == "no_sources_found" for item in results):
        assert (out / "probe.json").is_file(), diagnostic
        assert isinstance(_load(out / "probe.json"), list), diagnostic


def test_missing_out_argument_exits_two(source_folder):
    """T3: argparse rejects a missing required --out argument."""
    result = _run_demo("--source", source_folder)
    assert result.returncode == 2, _diagnostic(result)
    assert "--out" in result.stderr, _diagnostic(result)


def test_nonexistent_source_exits_two_with_clear_message(tmp_path):
    """T4: an absent local source is reported clearly with exit code 2."""
    missing = tmp_path / "source-does-not-exist"
    result = _run_demo("--source", missing, "--out", tmp_path / "output")
    diagnostic = _diagnostic(result)
    assert result.returncode == 2, diagnostic
    assert "source folder does not exist" in result.stderr.lower(), diagnostic
    assert str(missing) in result.stderr, diagnostic
