from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.cli.entrypoint import main

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _gpu_command(runtime_root, *args: str) -> list[str]:
    return ["--machine-runtime-root", str(runtime_root), "agent", "config", "gpus", *args, "--format", "json"]


def _initialize(runtime_root, capsys) -> None:
    assert main(["init", "--machine", "gpu-1", "--machine-runtime-root", str(runtime_root), "--format", "json"]) == 0
    capsys.readouterr()


def test_gpu_policy_accepts_expect_machine_and_legacy_machine_alias(tmp_path, capsys) -> None:
    runtime_root = tmp_path / "machine"
    _initialize(runtime_root, capsys)

    assert main([*_gpu_command(runtime_root, "set", "--visible", "0", "--expect-machine", "gpu-1")]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["current_revision"] == 1

    assert main([*_gpu_command(runtime_root, "set", "--visible", "1", "--machine", "gpu-1")]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["current_revision"] == 2
    assert second["configured_gpu_ids"] == [1]


def test_machine_assertion_mismatch_fails_before_gpu_policy_mutation(tmp_path, capsys) -> None:
    runtime_root = tmp_path / "machine"
    _initialize(runtime_root, capsys)
    assert main([*_gpu_command(runtime_root, "set", "--visible", "0")]) == 0
    capsys.readouterr()

    assert main([*_gpu_command(runtime_root, "set", "--visible", "1", "--expect-machine", "wrong")]) == 2
    mismatch = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "expected 'wrong'" in mismatch
    assert "actual local agent identity is 'gpu-1'" in mismatch
    assert "assertion" in mismatch

    assert main([*_gpu_command(runtime_root, "show", "--format", "json")]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["revision"] == 1
    assert shown["configured_gpu_ids"] == [0]


@pytest.mark.parametrize(
    "assertions",
    [
        ("--machine", "gpu-1", "--expect-machine", "gpu-2"),
        ("--expect-machine", "gpu-1", "--machine", "gpu-2"),
    ],
)
def test_conflicting_machine_aliases_fail_before_dispatch(tmp_path, capsys, assertions) -> None:
    _initialize(tmp_path / "machine", capsys)
    command = _gpu_command(tmp_path / "machine", "set", "--visible", "0", *assertions)
    assert main(command) == 2
    captured = capsys.readouterr()
    assert "conflicting values" in json.loads(captured.out)["error"]["message"]


def test_equal_machine_aliases_are_accepted_across_parser_levels(tmp_path, capsys) -> None:
    runtime_root = tmp_path / "machine"
    _initialize(runtime_root, capsys)
    command = [
        "--machine",
        "gpu-1",
        *_gpu_command(runtime_root, "set", "--visible", "0"),
        "--expect-machine",
        "gpu-1",
    ]
    assert main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["current_revision"] == 1


def test_qexp_machine_remains_an_assertion_source(tmp_path, monkeypatch, capsys) -> None:
    runtime_root = tmp_path / "machine"
    _initialize(runtime_root, capsys)
    monkeypatch.setenv("QEXP_MACHINE", "wrong")
    assert main([*_gpu_command(runtime_root, "set", "--visible", "0", "--expect-machine", "gpu-1")]) == 2
    assert "QEXP_MACHINE" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_help_distinguishes_assertion_setup_and_placement(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    root_help = " ".join(capsys.readouterr().out.split())
    assert "--expect-machine" in root_help
    assert "--home-machine" in root_help
    assert "identity assertion" in root_help

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "--help"])
    assert exc_info.value.code == 0
    init_help = " ".join(capsys.readouterr().out.split())
    assert "identity" in init_help
    assert "--expect-machine" not in init_help


def test_expect_machine_is_not_a_setup_identity_option(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["init", "--expect-machine", "gpu-1", "--machine-runtime-root", str(tmp_path / "machine")])
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_expect_machine_is_not_a_legacy_migration_identity_option(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["admin", "migrate", "agent", "--expect-machine", "gpu-1"])
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_unbound_observation_remains_available_without_new_assertion(tmp_path, capsys) -> None:
    shared_root = tmp_path / ".qexp"
    init_shared_root(shared_root, "project-gpu", runtime_root=tmp_path / "legacy-runtime")
    base = [
        "--project",
        str(shared_root),
        "--machine-runtime-root",
        str(tmp_path / "unbound-machine-runtime"),
        "task",
        "list",
        "--format",
        "json",
    ]

    assert main(base) == 0
    capsys.readouterr()
    assert main([*base, "--expect-machine", "project-gpu"]) == 2
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "cannot verify a Project identity without a local binding" in message
