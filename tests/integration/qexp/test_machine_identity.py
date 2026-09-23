"""Identity failures must never authorize fresh machine setup."""

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime, MachineRuntimeUninitializedError
from qqtools.plugins.qexp.agent.identity import MachineRuntimeIdentityError
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.infrastructure.host import host_instance_id
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
        assert "qexp admin repair identity --dry-run" in str(captured.value)
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
        assert "qexp admin repair identity --dry-run" in error["message"]
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
            assert "qexp admin repair identity --dry-run" in str(captured.value)
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


def _diagnose(runtime: MachineRuntime, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, object]]:
    exit_code = main(
        ["--machine-runtime-root", str(runtime.root), "admin", "repair", "identity", "--dry-run", "--format=json"]
    )
    return exit_code, json.loads(capsys.readouterr().out)


def test_identity_diagnosis_healthy_and_config_outside_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    runtime.paths["config"].write_text("{", encoding="utf-8")
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 0
    assert result["outcome"] == "healthy"
    assert result["runtime_root"] == str(runtime.root)
    assert result["selection_source"] == "--machine-runtime-root"
    assert result["configuration_scope"] == "not_checked"
    assert result["planned_changes"] == []
    assert any(check["name"] == "host_continuity" and check["status"] == "passed" for check in result["checks"])
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_does_not_trust_explicit_runtime_id_without_host_proof(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    record = json.loads(runtime.paths["identity"].read_text(encoding="utf-8"))
    record["machine_runtime"]["instance_id"] = "independently-staged-seed"
    runtime.paths["identity"].write_text(json.dumps(record), encoding="utf-8")
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == "host_continuity_unverified"
    assert any(check["name"] == "generation" and check["status"] == "passed" for check in result["checks"])
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_detects_host_mismatch_in_seed_only_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    record = json.loads(runtime.paths["identity"].read_text(encoding="utf-8"))
    seed = record["machine_runtime"]["instance_id"]
    record["machine_runtime"].pop("runtime_id")
    runtime.paths["identity"].write_text(json.dumps(record), encoding="utf-8")
    other_runtime_id = sha256(f"{seed}\0different-host".encode()).hexdigest()
    assert other_runtime_id != sha256(f"{seed}\0{host_instance_id()}".encode()).hexdigest()
    atomic_replace(
        runtime.paths["current_generation"],
        {"current_generation": {"version": 1, "runtime_id": other_runtime_id, "published_at": 1.0}},
    )
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == "host_mismatch"
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("state", ["fresh", "missing_with_data", "malformed"])
def test_identity_diagnosis_classifies_missing_and_corrupt_state_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], state: str
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    if state == "missing_with_data":
        runtime.root.mkdir()
        runtime.paths["inventory"].write_text("{}", encoding="utf-8")
    elif state == "malformed":
        runtime.root.mkdir()
        runtime.paths["identity"].write_text("{", encoding="utf-8")
    before = _snapshot(runtime.root) if runtime.root.exists() else None

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == ("uninitialized" if state == "fresh" else "insufficient_evidence")
    assert (_snapshot(runtime.root) if runtime.root.exists() else None) == before


def test_identity_diagnosis_rejects_mutating_spelling_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    exit_code = main(["--machine-runtime-root", str(runtime.root), "admin", "repair", "identity", "--format=json"])
    output = capsys.readouterr()

    assert exit_code == 2
    assert "automatic restoration is unavailable" in output.out + output.err
    assert not runtime.root.exists()


def test_identity_diagnosis_reports_pending_replacement_without_restoring_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    archive = runtime.paths["archives"] / "old-new"
    atomic_replace(
        runtime.paths["replacement_transaction"],
        {
            "replacement": {
                "version": 1,
                "phase": "archived",
                "old_runtime_id": runtime.instance_id,
                "new_runtime_id": "a" * 64,
                "target_name": "gpu-2",
                "policy": "daemon",
                "detach_old_runtime": False,
                "archive_path": str(archive),
                "obligations": [],
            }
        },
    )
    runtime.paths["identity"].unlink()
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["reason"] == "replacement_pending"
    assert result["replacement_phase"] == "archived"
    assert result["replacement_target"] == "gpu-2"
    assert _snapshot(runtime.root) == before

    human_exit = main(["--machine-runtime-root", str(runtime.root), "admin", "repair", "identity", "--dry-run"])
    human = capsys.readouterr().out
    assert human_exit == 1
    assert "archived" in human
    assert "gpu-2" in human
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_blocks_malformed_replacement_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    atomic_replace(
        runtime.paths["replacement_transaction"],
        {
            "replacement": {
                "version": 1,
                "phase": "archived",
                "old_runtime_id": runtime.instance_id,
                "new_runtime_id": "a" * 64,
                "target_name": "gpu-2",
                "policy": "daemon",
                "detach_old_runtime": False,
                "archive_path": 7,
                "obligations": [],
            }
        },
    )
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == "replacement_pending"
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_distinguishes_unreadable_evidence_from_blocked_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    before = _snapshot(runtime.root)
    original_open = Path.open

    def fail_identity_open(path: Path, *args, **kwargs):
        if path == runtime.paths["identity"]:
            raise PermissionError("injected read failure")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_identity_open)
        exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "failed"
    assert result["reason"] == "access_failed"
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_blocks_inconsistent_binding(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    atomic_replace(
        runtime.paths["registry"],
        {
            "registry": {
                "version": 1,
                "revision": 1,
                "bindings": [
                    {
                        "project_id": "project-1",
                        "shared_root": str(tmp_path / "project"),
                        "machine_name": "gpu-1",
                        "enabled": True,
                        "registration_generation": "generation-1",
                        "runtime_instance_id": "a" * 64,
                        "runtime_root": str(runtime.root),
                    }
                ],
            }
        },
    )
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == "inconsistent_generation"
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_blocks_binding_without_ownership_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    atomic_replace(
        runtime.paths["registry"],
        {
            "registry": {
                "version": 1,
                "revision": 1,
                "bindings": [
                    {
                        "project_id": "project-1",
                        "shared_root": str(tmp_path / "project"),
                        "machine_name": "gpu-1",
                        "enabled": True,
                        "registration_generation": None,
                        "runtime_instance_id": None,
                        "runtime_root": str(runtime.root),
                    }
                ],
            }
        },
    )
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["reason"] == "insufficient_evidence"
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_blocks_foreign_binding_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    atomic_replace(
        runtime.paths["registry"],
        {
            "registry": {
                "version": 1,
                "revision": 1,
                "bindings": [
                    {
                        "project_id": "project-1",
                        "shared_root": str(tmp_path / "project"),
                        "machine_name": "gpu-1",
                        "enabled": True,
                        "registration_generation": "generation-1",
                        "runtime_instance_id": runtime.instance_id,
                        "runtime_root": str(tmp_path / "foreign-machine"),
                    }
                ],
            }
        },
    )
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "blocked"
    assert result["reason"] == "inconsistent_generation"
    assert _snapshot(runtime.root) == before


def test_identity_diagnosis_dangling_replacement_link_is_inspection_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    runtime.paths["replacement_transaction"].symlink_to(tmp_path / "missing-transaction")
    before = _snapshot(runtime.root)

    exit_code, result = _diagnose(runtime, capsys)

    assert exit_code == 1
    assert result["outcome"] == "failed"
    assert result["reason"] == "access_failed"
    assert _snapshot(runtime.root) == before
