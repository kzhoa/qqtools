"""Check resolved tox isolation without installing or running product suites."""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_resolved_tox_gates_preserve_interpreters_and_artifact_isolation():
    environments = [
        "artifact-e2e",
        "release-e2e",
        *(f"py{minor}-artifact-smoke" for minor in (311, 312, 313, 314)),
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tox",
            "config",
            "--no-provision",
            "-e",
            ",".join(environments),
            "-k",
            "base_python",
            "commands",
            "deps",
            "package",
            "set_env",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(result.stdout)
    artifact = config["testenv:artifact-e2e"]
    release = config["testenv:release-e2e"]
    for key in ("base_python", "commands", "deps", "package"):
        assert release[key] == artifact[key]
    assert "--installed-lifecycle-gate" in release["commands"]
    for name in environments:
        environment = config[f"testenv:{name}"]
        assert environment["package"] == "wheel"
        assert "PYTHONPATH=" in environment["set_env"].splitlines()
        assert str(ROOT / "src") not in environment["set_env"]
    for minor in (311, 312, 313, 314):
        smoke = config[f"testenv:py{minor}-artifact-smoke"]
        assert smoke["base_python"] == f"py{minor}"
        assert not smoke["deps"].strip()
        assert "python -E" in smoke["commands"]
        assert "site-packages" in smoke["commands"]
        assert "qexp --help" in smoke["commands"]


def test_default_tox_gate_does_not_repeat_suites_or_erase_diagnostics():
    result = subprocess.run(
        [sys.executable, "-m", "tox", "list", "--no-provision", "-d", "--no-desc"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["preflight"]
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    assert not config.getboolean("tox", "skip_missing_interpreters")
    assert not config.has_section("testenv:cleanup")
