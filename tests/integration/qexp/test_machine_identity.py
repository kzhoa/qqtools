"""Identity failures must never authorize fresh machine setup."""

import json
import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime, MachineRuntimeUninitializedError
from qqtools.plugins.qexp.agent.identity import MachineRuntimeIdentityError
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None for path in root.rglob("*")}


def test_fresh_runtime_advises_init_without_creating_files(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    with pytest.raises(MachineRuntimeUninitializedError, match="qexp init --machine NAME"):
        runtime.require_initialized()
    assert not runtime.root.exists()
    initialize_machine(runtime, "gpu-1")
    runtime.require_initialized()


@pytest.mark.parametrize(
    "contents",
    [
        b"{",
        b"\xff",
        b"[]",
        b"{}",
        b'{"machine_runtime": []}',
        b'{"machine_runtime": {"instance_id": ""}}',
        b'{"machine_runtime": {"instance_id": "seed", "runtime_id": "invalid"}}',
    ],
)
def test_malformed_identity_is_not_uninitialized(tmp_path: Path, contents: bytes) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    runtime.paths["identity"].write_bytes(contents)
    before = _snapshot(runtime.root)
    for action in (runtime.require_initialized, lambda: initialize_machine(runtime, "gpu-1", confirmed=True)):
        with pytest.raises(MachineRuntimeIdentityError, match="malformed") as captured:
            action()
        assert not isinstance(captured.value, MachineRuntimeUninitializedError)
        assert str(runtime.paths["identity"]) in str(captured.value)
        assert "qexp init" not in str(captured.value)
        assert _snapshot(runtime.root) == before


@pytest.mark.parametrize(
    "relative_path",
    [
        "projects/project-a/runtime/agent/processes/task.json",
        "agent/machine-agent.pid",
        "reservations/active/task.json",
        "inventory.json",
        "current-generation.json",
        "registry.json",
        "config.json",
    ],
)
def test_missing_identity_with_evidence_blocks_operations_and_initialization(
    tmp_path: Path, relative_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    evidence = runtime.root / relative_path
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    before = _snapshot(runtime.root)
    with pytest.raises(MachineRuntimeIdentityError, match="existing runtime data"):
        runtime.require_initialized()
    with pytest.raises(MachineRuntimeIdentityError, match="existing runtime data"):
        initialize_machine(runtime, "gpu-1", confirmed=True)
    for command in (["init", "--machine", "gpu-1", "--yes"], ["agent", "name"], ["project", "list"]):
        assert main(["--machine-runtime-root", str(runtime.root), *command, "--format=json"]) == 1
        error = json.loads(capsys.readouterr().out)["error"]
        assert "existing runtime data" in error["message"]
        assert "qexp init" not in error["message"]
        assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_unreadable_identity_reports_access_failure_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[OSError]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    before = _snapshot(runtime.root)
    original_open = Path.open

    def fail_identity_open(path: Path, *args, **kwargs):
        if path == runtime.paths["identity"]:
            raise error_type("injected read failure")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_identity_open)
        for action in (runtime.require_initialized, lambda: initialize_machine(runtime, "gpu-1", confirmed=True)):
            with pytest.raises(MachineRuntimeIdentityError, match="unreadable") as captured:
                action()
            assert "qexp init" not in str(captured.value)
            assert "access permissions" in str(captured.value)
    assert _snapshot(runtime.root) == before


def test_unreadable_evidence_is_not_an_empty_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    projects = runtime.paths["projects"]
    projects.mkdir(parents=True)
    original_scandir = os.scandir

    def fail_projects_scan(path):
        if Path(path) == projects:
            raise PermissionError("injected evidence access failure")
        return original_scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", fail_projects_scan)
        with pytest.raises(MachineRuntimeIdentityError, match="cannot inspect"):
            runtime.require_initialized()
    assert not runtime.paths["identity"].exists()


def test_dangling_identity_link_does_not_authorize_initialization(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.root.mkdir()
    runtime.paths["identity"].symlink_to(tmp_path / "missing-identity")
    for action in (runtime.require_initialized, lambda: initialize_machine(runtime, "gpu-1")):
        with pytest.raises(MachineRuntimeIdentityError, match="inaccessible"):
            action()
    assert runtime.paths["identity"].is_symlink()
    assert not (tmp_path / "missing-identity").exists()


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_cli_renders_corrupt_identity_as_operational_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    runtime.paths["identity"].write_text("{", encoding="utf-8")
    before = _snapshot(runtime.root)
    for command in (["init", "--machine", "gpu-1"], ["agent", "name"], ["agent", "start"]):
        assert main(["--machine-runtime-root", str(runtime.root), *command, "--format", output_format]) == 1
        output = capsys.readouterr()
        message = json.loads(output.out)["error"]["message"] if output_format == "json" else output.err
        assert "malformed" in message
        assert str(runtime.paths["identity"]) in message
        assert "qexp init" not in message
        assert _snapshot(runtime.root) == before


def test_empty_layout_and_lock_files_allow_initialization(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    with runtime.agent_lifecycle_guard():
        pass
    with pytest.raises(MachineRuntimeUninitializedError):
        runtime.require_initialized()
    initialize_machine(runtime, "gpu-1")
    runtime.require_initialized()


def test_archived_replacement_with_missing_identity_can_resume(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    old_id = runtime.instance_id
    new_id = "a" * 64
    archive = runtime.paths["archives"] / f"{old_id}-{new_id}"
    atomic_replace(archive / "manifest.json", {"archive": {"old_runtime_id": old_id}})
    atomic_replace(
        runtime.paths["replacement_transaction"],
        {
            "replacement": {
                "version": 1,
                "phase": "archived",
                "old_runtime_id": old_id,
                "new_runtime_id": new_id,
                "target_name": "gpu-1",
                "policy": "daemon",
                "detach_old_runtime": False,
                "archive_path": str(archive),
                "obligations": [],
            }
        },
    )
    runtime.paths["identity"].unlink()
    result = initialize_machine(runtime, "gpu-1", confirmed=True)
    assert result["action"] == "reinitialized"
    assert result["old_runtime_id"] == old_id
    assert runtime.instance_id == new_id
    runtime.require_initialized()
