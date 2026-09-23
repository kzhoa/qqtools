"""Exercise opt-in selection through real pytest collection and execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], "1 passed, 1 deselected"),
        (["-m", "stress"], "2 deselected"),
        (["--run-stress", "-m", "stress"], "1 passed, 1 deselected"),
    ],
)
def test_stress_requires_explicit_opt_in(tmp_path, arguments, expected):
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "conftest.py").write_text((root / "tests/conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    stress: optional qualification\n")
    (tmp_path / "test_sample.py").write_text(
        "import pytest\ndef test_regular(): pass\n@pytest.mark.stress\ndef test_load(): pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (5 if arguments == ["-m", "stress"] else 0), result.stdout + result.stderr
    assert expected in result.stdout


def test_release_selects_every_slow_storage_matrix():
    from scripts.ci.run_preflight import RELEASE_SLOW_INTEGRATION_NODES

    root = Path(__file__).resolve().parents[3]
    path = "tests/integration/test_local_responsibility_storage.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "--collect-only", "-m", "slow", "-q"],
        cwd=root,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = {line.split("[", 1)[0] for line in result.stdout.splitlines() if line.startswith(path + "::")}
    selected = {node for node in RELEASE_SLOW_INTEGRATION_NODES if node.startswith(path + "::")}
    assert collected
    assert collected == selected
