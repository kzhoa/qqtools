from __future__ import annotations

import configparser
import importlib.machinery
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import run_preflight

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    loader = importlib.machinery.SourceFileLoader("qqtools_dev_entry", str(ROOT / "scripts/dev"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_tool_pins_match_script_tox_and_ci():
    script = (ROOT / "scripts/dev").read_text()
    metadata = script.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    config = tomllib.loads("\n".join(line.removeprefix("# ") for line in metadata.splitlines()))
    pins = set(config["dependencies"])
    assert {pin.split("==")[0] for pin in pins} == {"tox", "tox-uv", "uv"}
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for name in ("dev", "ci-preflight"):
        assert {dep for dep in extras[name] if dep.split("==")[0] in {"tox", "tox-uv", "uv"}} == pins
    tox_config = configparser.ConfigParser(interpolation=None)
    tox_config.read(ROOT / "tox.ini")
    assert set(tox_config["tox"]["requires"].split()) == pins


@pytest.mark.parametrize("args", [["-q"], ["-k", "retry"], ["tests/unit/example.py::test_retry", "-q"]])
def test_entry_forwards_pytest_arguments_and_exit_code(entry, monkeypatch, args):
    calls = []

    def run(command):
        calls.append(command)
        return 5

    monkeypatch.setattr(entry, "run", run)
    assert entry.main(["test", *args]) == 5
    assert calls == [[sys.executable, "-m", "tox", "run", "-e", "unit", "--", *args]]


@pytest.mark.parametrize("command", ["env", "preflight"])
def test_gate_and_env_reject_extra_arguments(entry, command):
    with pytest.raises(SystemExit) as exc:
        entry.main([command, "-k", "retry"])
    assert exc.value.code == 2


def test_prerequisite_failure_prevents_tox_setup(entry, monkeypatch):
    monkeypatch.setattr(entry, "check_prerequisites", lambda: "missing tmux")
    monkeypatch.setattr(entry, "run", lambda _: pytest.fail("tox must not start"))
    assert entry.main(["preflight"]) == 1


def test_non_313_runner_fails_before_commands(monkeypatch, capsys):
    # Preserve the attributes supplied by the real version_info object.
    class Version(tuple):
        major = 3
        minor = 12

    monkeypatch.setattr(run_preflight, "sys", SimpleNamespace(version_info=Version((3, 12)), stderr=sys.stderr))
    monkeypatch.setattr(run_preflight, "COMMANDS", (("must-not-run",),))
    assert run_preflight.main([]) == 1
    assert "./scripts/dev preflight" in capsys.readouterr().err


@pytest.mark.parametrize("platform,tmux,expected", [("darwin", "/bin/tmux", "Linux"), ("linux", None, "tmux")])
def test_platform_prerequisites(monkeypatch, platform, tmux, expected):
    monkeypatch.setattr(run_preflight, "sys", SimpleNamespace(version_info=(3, 13), platform=platform))
    monkeypatch.setattr(run_preflight.shutil, "which", lambda _: tmux)
    assert expected in run_preflight.check_prerequisites()


def test_release_profile_uses_exact_release_validator_and_excludes_large_observation_scale(monkeypatch):
    commands = []
    monkeypatch.setattr(run_preflight, "check_prerequisites", lambda: None)
    monkeypatch.setattr(
        run_preflight.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert (
        run_preflight.main(
            [
                "--profile",
                "release",
                "--release-base",
                "base",
                "--release-head",
                "head",
                "--release-actor",
                "kzhoa",
            ]
        )
        == 0
    )

    rendered = [" ".join(command) for command in commands]
    validator = next(index for index, command in enumerate(rendered) if "check_release_commit.py validate" in command)
    common = next(index for index, command in enumerate(rendered) if "ruff check" in command)
    full_qexp = next(index for index, command in enumerate(rendered) if "qexp_integration_gate.py" in command)
    assert validator < common < full_qexp
    assert "--budget-seconds 600" in rendered[full_qexp]
    assert commands[full_qexp][-1] == f"--deselect={run_preflight.RELEASE_EXCLUDED_OBSERVATION_SCALE_NODE}"
    assert sum(argument.startswith("--deselect=") for argument in commands[full_qexp]) == 1
    assert not any("--lifecycle-gate=representative" in command for command in rendered)


def test_release_profile_requires_release_identity(monkeypatch):
    monkeypatch.setattr(run_preflight, "check_prerequisites", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        run_preflight.main(["--profile", "release"])

    assert exc_info.value.code == 2


def test_existing_incompatible_ide_environment_is_preserved(entry, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    venv = tmp_path / ".venv"
    venv.mkdir()
    sentinel = venv / "keep.txt"
    sentinel.write_text("existing environment")
    assert entry.main(["env"]) == 1
    assert sentinel.read_text() == "existing environment"
    assert "Remove it" in capsys.readouterr().err


def test_existing_ide_environment_is_reused_and_explicitly_targeted(entry, monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    python = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "unrelated"))
    monkeypatch.setattr(entry.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    calls = []
    monkeypatch.setattr(entry, "run", lambda command: calls.append(command) or 0)
    assert entry.main(["env"]) == 0
    assert calls == [["uv", "pip", "install", "--python", str(python), "-e", ".[full]", "pytest-xdist"]]
