from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_e2e_helpers():
    helper_path = Path(__file__).parents[2] / "e2e" / "qexp" / "qexp_e2e.py"
    spec = importlib.util.spec_from_file_location("qexp_e2e_isolation_test", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_env_replaces_host_resource_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helpers = _load_e2e_helpers()
    monkeypatch.setenv("QEXP_MACHINE_RUNTIME_ROOT", "/host/qexp-machine")
    monkeypatch.setenv("TMPDIR", "/host/tmp")
    monkeypatch.setenv("TMUX", "/host/tmux")
    monkeypatch.setenv("TMUX_PANE", "%99")
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")

    base = tmp_path / "case"
    env = helpers.make_env(base)

    assert env["QEXP_MACHINE_RUNTIME_ROOT"] == str(base / "machine-runtime")
    assert env["TMPDIR"] == str(base / "tmp")
    assert env["TMP"] == str(base / "tmp")
    assert env["TEMP"] == str(base / "tmp")
    assert env["TMUX_TMPDIR"] == str(base / "tmux")
    assert "TMUX" not in env
    assert "TMUX_PANE" not in env
    assert "PYTHONPATH" not in env


def test_cleanup_rejects_machine_runtime_outside_test_root(tmp_path: Path) -> None:
    helpers = _load_e2e_helpers()
    env = helpers.make_env(tmp_path / "case")
    common = ["qexp", "--machine-runtime-root", str(tmp_path / "outside")]

    with pytest.raises(RuntimeError, match="refusing to clean non-test machine runtime"):
        helpers._require_test_owned_cleanup(common, env)


def test_cleanup_accepts_resources_owned_by_test_root(tmp_path: Path) -> None:
    helpers = _load_e2e_helpers()
    base = tmp_path / "case"
    env = helpers.make_env(base)
    common = ["qexp", "--machine-runtime-root", str(base / "alternate-machine-runtime")]

    helpers._require_test_owned_cleanup(common, env)


def test_cleanup_accepts_realistic_common_command(tmp_path: Path) -> None:
    helpers = _load_e2e_helpers()
    base = tmp_path / "case"
    env = helpers.make_env(base)
    common = [
        "qexp",
        "--project",
        str(base / ".qexp"),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(base / "runtime"),
        "--machine-runtime-root",
        str(base / "machine-runtime"),
    ]

    helpers._require_test_owned_cleanup(common, env)


def test_cleanup_rejects_duplicate_machine_runtime_arguments(tmp_path: Path) -> None:
    helpers = _load_e2e_helpers()
    base = tmp_path / "case"
    env = helpers.make_env(base)
    common = [
        "qexp",
        "--machine-runtime-root",
        str(base / "owned"),
        "--machine-runtime-root",
        str(tmp_path / "outside"),
    ]

    with pytest.raises(RuntimeError, match="duplicate --machine-runtime-root"):
        helpers._require_test_owned_cleanup(common, env)


@pytest.mark.parametrize(
    "argument",
    [
        ["--machine-runtime-root"],
        ["--machine-runtime-root", ""],
        ["--machine-runtime-root="],
    ],
)
def test_cleanup_rejects_missing_machine_runtime_value(tmp_path: Path, argument: list[str]) -> None:
    helpers = _load_e2e_helpers()
    env = helpers.make_env(tmp_path / "case")

    with pytest.raises(RuntimeError, match="missing --machine-runtime-root value"):
        helpers._require_test_owned_cleanup(["qexp", *argument], env)


@pytest.mark.parametrize(
    "argument",
    [
        ["--machine-runtime-r", "/var/tmp/non-test-qexp"],
        ["--machine-runtime-r=/var/tmp/non-test-qexp"],
    ],
)
def test_cleanup_rejects_abbreviated_machine_runtime_argument(tmp_path: Path, argument: list[str]) -> None:
    helpers = _load_e2e_helpers()
    env = helpers.make_env(tmp_path / "case")

    with pytest.raises(RuntimeError, match="abbreviated --machine-runtime-root"):
        helpers._require_test_owned_cleanup(["qexp", *argument], env)
