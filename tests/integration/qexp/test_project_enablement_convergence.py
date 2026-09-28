from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status
from qqtools.plugins.qexp.agent.setup import set_project_enablement
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _args(runtime_root: Path, *values: str) -> list[str]:
    return ["--machine-runtime-root", str(runtime_root), *values]


def _json(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def _registered_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path, Path, dict]:
    runtime_root = tmp_path / "machine"
    project_root = tmp_path / "project"
    assert main(_args(runtime_root, "init", "--machine", "gpu-1", "--format=json")) == 0
    capsys.readouterr()
    assert main(_args(runtime_root, "project", "init", str(project_root), "--format=json")) == 0
    capsys.readouterr()
    assert main(_args(runtime_root, "project", "register", str(project_root), "--format=json")) == 0
    registered = _json(capsys)["projects"][0]
    return runtime_root, project_root, registered


def test_enablement_commit_and_idempotent_retry_expose_independent_revisions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)

    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 0
    disabled = _json(capsys)
    assert disabled == {
        **disabled,
        "action": "project_disabled",
        "status": "committed",
        "requested_enabled": False,
        "effective_enabled": False,
        "inventory_converged": True,
        "reason": None,
    }
    assert type(disabled["registry_revision"]) is int
    assert type(disabled["inventory_revision"]) is int

    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 0
    repeated = _json(capsys)
    assert repeated["status"] == "committed"
    assert repeated["registry_revision"] > disabled["registry_revision"]
    assert repeated["inventory_revision"] == disabled["inventory_revision"]

    assert main(_args(runtime_root, "project", "list", "--format=json")) == 0
    listing = _json(capsys)
    project = listing["projects"][0]
    assert listing["revision"] == disabled["inventory_revision"]
    assert listing["inventory_revision"] == disabled["inventory_revision"]
    assert listing["registry_revision"] == repeated["registry_revision"]
    assert project["enabled"] is False
    assert project["inventory_enabled"] is False
    assert project["effective_enabled"] is False
    assert project["inventory_converged"] is True
    assert project["inventory_revision"] == disabled["inventory_revision"]
    assert project["registry_revision"] == repeated["registry_revision"]


def test_inventory_failure_reports_partial_commit_and_retry_repairs_without_inverse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)
    second_root = tmp_path / "second-project"
    assert main(_args(runtime_root, "project", "init", str(second_root), "--format=json")) == 0
    capsys.readouterr()
    assert main(_args(runtime_root, "project", "register", str(second_root), "--format=json")) == 0
    second = _json(capsys)["projects"][0]
    runtime = MachineRuntime(runtime_root)
    inventory_before = read_json(runtime.paths["inventory"])
    unrelated_before = next(
        item for item in inventory_before["inventory"]["entries"] if item["project_id"] == second["project_id"]
    )
    real_save = __import__("qqtools.plugins.qexp.agent.setup", fromlist=["save_inventory_locked"]).save_inventory_locked

    def fail_inventory(*_args, **_kwargs) -> None:
        raise OSError("private inventory failure detail")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.setup.save_inventory_locked", fail_inventory)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 1
    partial = _json(capsys)
    assert partial["status"] == "partially_committed"
    assert partial["reason"] == "inventory_mirror_incomplete"
    assert partial["requested_enabled"] is False
    assert partial["effective_enabled"] is False
    assert partial["inventory_converged"] is False
    assert "private inventory failure detail" not in json.dumps(partial)
    assert (
        next(item for item in runtime.load_registry()[1] if item.project_id == registered["project_id"]).enabled
        is False
    )
    assert read_json(runtime.paths["inventory"]) == inventory_before
    status = get_machine_agent_status(runtime)
    assert status["inventory_converged"] is False
    assert "enablement_mirror_diverged" in status["enablement_blockers"]
    assert read_json(runtime.paths["inventory"]) == inventory_before

    monkeypatch.setattr("qqtools.plugins.qexp.agent.inventory.save_inventory_locked", fail_inventory)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"])) == 1
    human = capsys.readouterr().out.lower()
    assert "partially_committed" in human
    assert "effective" in human
    assert "retry" in human

    monkeypatch.undo()
    monkeypatch.setattr("qqtools.plugins.qexp.agent.setup.save_inventory_locked", real_save)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 0
    repaired = _json(capsys)
    assert repaired["status"] == "committed"
    assert repaired["registry_revision"] > partial["registry_revision"]
    assert repaired["inventory_revision"] > partial["inventory_revision"]
    inventory_after = read_json(runtime.paths["inventory"])
    assert (
        next(item for item in inventory_after["inventory"]["entries"] if item["project_id"] == second["project_id"])
        == unrelated_before
    )


def test_failure_before_registry_replace_is_not_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)
    runtime = MachineRuntime(runtime_root)
    revision_before, bindings_before = runtime.load_registry()
    real_fsync = __import__("qqtools.plugins.qexp.runtime.store", fromlist=["os"]).os.fsync
    calls = 0

    def fail_first_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("private pre-replace detail")
        real_fsync(fd)

    monkeypatch.setattr("qqtools.plugins.qexp.runtime.store.os.fsync", fail_first_fsync)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 1
    result = _json(capsys)
    assert result["status"] == "not_committed"
    assert result["reason"] == "registry_not_committed"
    assert result["effective_enabled"] is True
    assert result["registry_revision"] == revision_before
    assert "private pre-replace detail" not in json.dumps(result)
    assert runtime.load_registry() == (revision_before, bindings_before)


def test_directory_fsync_failure_is_unknown_even_when_new_value_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)
    runtime = MachineRuntime(runtime_root)
    real_fsync = __import__("qqtools.plugins.qexp.runtime.store", fromlist=["os"]).os.fsync
    calls = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("private post-replace detail")
        real_fsync(fd)

    monkeypatch.setattr("qqtools.plugins.qexp.runtime.store.os.fsync", fail_directory_fsync)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 1
    result = _json(capsys)
    assert result["status"] == "outcome_unknown"
    assert result["reason"] == "registry_outcome_unknown"
    assert result["effective_enabled"] is None
    assert result["inventory_converged"] is False
    assert type(result["registry_revision"]) is int
    assert "private post-replace detail" not in json.dumps(result)
    assert runtime.load_registry()[1][0].enabled is False

    monkeypatch.setattr("qqtools.plugins.qexp.runtime.store.os.fsync", real_fsync)
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 0
    retried = _json(capsys)
    assert retried["status"] == "committed"
    assert retried["registry_revision"] > result["registry_revision"]
    assert retried["inventory_converged"] is True


def test_disable_uses_local_stable_identity_when_shared_project_is_unreadable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, project_root, registered = _registered_project(tmp_path, capsys)
    shared_root = project_root / ".qexp"
    hidden_root = project_root / ".qexp-unavailable"
    shared_root.rename(hidden_root)

    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 0
    disabled = _json(capsys)
    assert disabled["status"] == "committed"
    assert disabled["effective_enabled"] is False

    assert main(_args(runtime_root, "project", "enable", registered["project_id"], "--format=json")) == 1
    failed_enable = _json(capsys)["error"]
    assert failed_enable["code"] == "operational_failure"


def test_project_list_human_output_labels_divergent_effective_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.setup.save_inventory_locked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mirror unavailable")),
    )
    assert main(_args(runtime_root, "project", "disable", registered["project_id"], "--format=json")) == 1
    capsys.readouterr()

    assert main(_args(runtime_root, "project", "list")) == 0
    human = capsys.readouterr().out.lower()
    assert "effective" in human
    assert "inventory" in human
    assert "warning" in human


def test_project_list_does_not_fall_back_to_inventory_when_registry_is_malformed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, _registered = _registered_project(tmp_path, capsys)
    runtime = MachineRuntime(runtime_root)
    runtime.paths["registry"].write_text("{not-json", encoding="utf-8")

    assert main(_args(runtime_root, "project", "list", "--format=json")) == 1

    result = _json(capsys)
    assert result["error"]["code"] == "operational_failure"
    assert "projects" not in result


def test_missing_registry_blocks_inventory_fallback_pool_replay_and_agent_start(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, _registered = _registered_project(tmp_path, capsys)
    runtime = MachineRuntime(runtime_root)
    runtime.paths["registry"].unlink()

    for command in (
        ("project", "list", "--format=json"),
        ("project", "register", "--from-pool", "--format=json"),
        ("agent", "start", "--timeout", "0.1", "--format=json"),
    ):
        assert main(_args(runtime_root, *command)) == 1
        result = _json(capsys)
        assert result["error"]["code"] == "operational_failure"
        assert "projects" not in result

    assert not runtime.paths["registry"].exists()


def test_disable_commit_orders_after_inflight_claim_guard_and_fences_later_claim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root, _project_root, registered = _registered_project(tmp_path, capsys)
    runtime = MachineRuntime(runtime_root)
    binding = runtime.load_registry()[1][0]
    claim_entered = Event()
    release_claim = Event()
    disable_done = Event()
    results: list[dict] = []

    def hold_committing_claim() -> None:
        with runtime.enabled_claim_guard(binding) as is_eligible:
            assert is_eligible
            claim_entered.set()
            assert release_claim.wait(timeout=5)

    def disable() -> None:
        results.append(set_project_enablement(runtime, registered["project_id"], False))
        disable_done.set()

    claim_thread = Thread(target=hold_committing_claim)
    disable_thread = Thread(target=disable)
    claim_thread.start()
    assert claim_entered.wait(timeout=5)
    disable_thread.start()
    assert not disable_done.wait(timeout=0.1)
    release_claim.set()
    claim_thread.join(timeout=5)
    disable_thread.join(timeout=5)

    assert not claim_thread.is_alive()
    assert not disable_thread.is_alive()
    assert results[0]["status"] == "committed"
    with runtime.enabled_claim_guard(binding) as is_eligible:
        assert not is_eligible
