import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import diagnostic_recovery
from qqtools.plugins.qexp.agent import scheduler_diagnostics as diagnostics
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _pid_start_time_ticks, _publish_process_status
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from tests.helpers.qexp.scheduler_recovery import (
    diagnostic_tree,
    first_project_history,
    overdue_identity,
    seed_over_budget_store,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic_ns=lambda: 0))
    return SchedulerDiagnosticStore(MachineRuntime(tmp_path / "runtime"))


def _observe(store, identity):
    return store.observe_finding(
        identity=identity, severity="fault", source_revision={"registry_revision": 1}, details={"progress": "started"}
    )


def test_full_store_recovery_exceeds_ordinary_archive_budget_and_is_restartable(store, monkeypatch, capsys):
    runtime = MachineRuntime(store.runtime_root)
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    monkeypatch.setattr(diagnostic_recovery, "_verified_agent", lambda *_args: (123, "successor", 456))
    command = ["--machine-runtime-root", str(runtime.root), "agent", "diagnostics", "repair", "--format=json"]
    seed_over_budget_store(runtime, "selected", "generation")
    assert store.paths["scheduler_diagnostics_history_index"].stat().st_size > diagnostics.MAX_PUBLICATION_BYTES

    finding = store.active_view(project_id="selected", limit=1)["items"][0]
    with pytest.raises(OSError, match="publication_byte_limit"):
        store.resolve_finding(
            identity=finding["identity"],
            episode=finding["episode"],
            observation_revision=finding["observation_revision"],
            source_revision=finding["source_revision"],
        )
    assert store.active_view()["total"] == diagnostics.MAX_ACTIVE_RECORDS

    before = diagnostic_tree(store.paths["scheduler_diagnostics"])
    assert main([*command, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["eligible"] == diagnostics.MAX_ACTIVE_RECORDS
    assert preview["pending_transition"] is True
    assert preview["apply_needed"] is True
    assert diagnostic_tree(store.paths["scheduler_diagnostics"]) == before

    class InterruptedAfterBackup(BaseException):
        pass

    captured_backups = []
    create_backup = diagnostics.create_scheduler_diagnostic_backup

    def interrupt_after_backup(*args, **kwargs):
        captured_backups.append(create_backup(*args, **kwargs))
        raise InterruptedAfterBackup

    with monkeypatch.context() as interrupted:
        interrupted.setattr(diagnostics, "create_scheduler_diagnostic_backup", interrupt_after_backup)
        with pytest.raises(InterruptedAfterBackup):
            main([*command, "--apply"])
    assert len(captured_backups) == 1
    assert diagnostic_tree(captured_backups[0]) == before
    assert diagnostic_tree(store.paths["scheduler_diagnostics"]) == before

    first_turn = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True, automatic=True)
    assert first_turn["state"] == "progressing"
    assert 0 < first_turn["retired"] <= 4
    assert store.summary_view(project_id="selected")["coverage"] == "incomplete"

    resumed = SchedulerDiagnosticStore(runtime)
    concurrent = {**overdue_identity(runtime, "current-project", "generation-2", 1), "target_id": "successor"}
    _observe(resumed, concurrent)
    assert main([*command, "--apply"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "awaiting_observation"
    assert result["apply_needed"] is False
    reported = diagnostic_recovery.read_scheduler_recovery(runtime, instance_id="successor")
    assert reported["retired"] == result["retired"]
    assert reported["last_progress_at"] is not None
    assert result["retired"] + first_turn["retired"] == diagnostics.MAX_ACTIVE_RECORDS
    assert result["backup_verified"] is True
    assert Path(result["backup_path"]).is_dir()
    assert resumed.active_view()["total"] == 1
    assert resumed.active_view()["items"][0]["identity"] == concurrent
    assert resumed.summary_view(project_id="selected")["coverage"] == "incomplete"
    history = first_project_history(resumed, "selected")
    assert history["disposition"] == "superseded"
    assert history["observation_count"] == 1
    assert history["details"] == {"progress": "started"}
    assert resumed.history_view()["captured_max_sequence"] == 501 + diagnostics.MAX_ACTIVE_RECORDS
    assert main([*command, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["retired"] == 0
    assert resumed.history_view()["captured_max_sequence"] == 501 + diagnostics.MAX_ACTIVE_RECORDS


def test_repair_preserves_current_surviving_and_ambiguous_findings(store):
    runtime = MachineRuntime(store.runtime_root)
    former = overdue_identity(runtime, "old-project", "generation", 0)
    current = {**former, "project_id": "new-project", "target_id": "successor"}
    ambiguous = dict(former)
    del ambiguous["target_id"]
    persistent = {**former, "producer": "primary_probe", "reason_code": "ready_index_unreadable"}
    for identity in (former, current, ambiguous, persistent):
        _observe(store, identity)
    unchanged = diagnostic_tree(store.paths["scheduler_diagnostics"])
    blocked = store.repair(runtime_id=runtime.instance_id, owner=lambda: None, apply=True)
    assert blocked["state"] == "blocked"
    assert blocked["reason"] == "ownership_unverified"
    assert diagnostic_tree(store.paths["scheduler_diagnostics"]) == unchanged

    result = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True)
    assert result["retired"] == 1
    assert result["retained_current"] == 1
    assert result["retained_uncertain"] == 2
    assert {item["identity"].get("target_id") for item in store.active_view()["items"]} == {
        "predecessor",
        "successor",
        None,
    }
    assert diagnostic_tree(Path(result["backup_path"])) == unchanged


def test_manual_repair_revalidates_owner_after_verified_backup(store):
    runtime = MachineRuntime(store.runtime_root)
    _observe(store, overdue_identity(runtime, "selected", "generation", 0))
    original = diagnostic_tree(store.paths["scheduler_diagnostics"])

    def owner():
        if list(store.paths["scheduler_diagnostics"].parent.glob("scheduler-v1-backup-*")):
            return None
        return "successor"

    deferred = store.repair(runtime_id=runtime.instance_id, owner=owner, apply=True)
    assert deferred["reason"] == "ownership_unverified"
    assert deferred["backup_verified"] is True
    assert deferred["retired"] == 0
    assert diagnostic_tree(store.paths["scheduler_diagnostics"]) == original

    current = {**overdue_identity(runtime, "selected", "new-generation", 1), "target_id": "successor"}
    _observe(store, current)
    result = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True)
    assert result["retired"] == 1
    assert store.active_view()["items"][0]["identity"] == current
    check = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True)
    assert check["retired"] == 0
    assert len(store.history_view()["items"]) == 1


def test_backup_failure_leaves_live_store_byte_for_byte_intact(store, monkeypatch):
    runtime = MachineRuntime(store.runtime_root)
    _observe(store, overdue_identity(runtime, "selected", "generation", 1))
    before = diagnostic_tree(store.paths["scheduler_diagnostics"])

    def fail_verification(*args, **kwargs):
        raise OSError("backup verification failure")

    monkeypatch.setattr(diagnostics, "create_scheduler_diagnostic_backup", fail_verification)
    result = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True)
    assert result["state"] == "blocked"
    assert result["reason"] == "backup_failed"
    assert result["backup_path"] is None
    assert result["backup_verified"] is False
    assert diagnostic_tree(store.paths["scheduler_diagnostics"]) == before


def test_replay_rebuilds_summary_if_a_new_finding_committed_before_retry(store, monkeypatch):
    runtime = MachineRuntime(store.runtime_root)
    former = overdue_identity(runtime, "former-project", "generation", 1)
    _observe(store, former)
    write = SchedulerDiagnosticStore._write

    def lose_summary(self, path, value, budget, *, final=False):
        if path == self.paths["scheduler_diagnostics_summary"]:
            raise OSError("interrupted after durable metadata write")
        return write(self, path, value, budget, final=final)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(SchedulerDiagnosticStore, "_write", lose_summary)
        failed = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True, automatic=True)
    assert failed["pending_transition"] is True
    assert store.active_view(project_id="former-project")["total"] == 0

    current = {**former, "project_id": "current-project", "target_id": "successor"}
    _observe(store, current)
    assert store.summary_view(project_id="former-project")["active_count"] == 1

    replay_started = datetime.now(timezone.utc)
    resumed = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True, automatic=True)
    assert resumed["pending_transition"] is False
    former_summary = store.summary_view(project_id="former-project")
    assert datetime.fromisoformat(former_summary["last_retirement_at"]) >= replay_started
    assert former_summary["active_count"] == 0
    assert store.summary_view(project_id="current-project")["active_count"] == 1
    assert store.history_view()["items"][0]["disposition"] == "superseded"


def test_repair_backs_up_then_cleans_abandoned_atomic_temporary_files(store):
    runtime = MachineRuntime(store.runtime_root)
    _observe(store, overdue_identity(runtime, "selected", "generation", 1))
    root = store.paths["scheduler_diagnostics"]
    partial_paths = (root / "history/.index.json.abcdefgh", root / f"pending/.{'a' * 32}.json.abcd0123")
    for partial in partial_paths:
        partial.write_bytes(b'{"diagnostics":')
    before = diagnostic_tree(root)
    preview = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor")
    assert preview["state"] == "pending"
    assert preview["pending_transition"] is True
    assert all(path.is_file() for path in partial_paths)

    repaired = store.repair(runtime_id=runtime.instance_id, owner=lambda: "successor", apply=True)
    assert repaired["state"] == "awaiting_observation"
    assert repaired["backup_verified"] is True
    assert diagnostic_tree(Path(repaired["backup_path"])) == before
    assert all(not path.exists() for path in partial_paths)
    assert len(store.history_view()["items"]) == 1


def test_recovery_shutdown_fences_a_pending_atomic_publication(store, monkeypatch):
    runtime = MachineRuntime(store.runtime_root)
    _observe(store, overdue_identity(runtime, "selected", "generation", 1))
    replace = diagnostics.atomic_replace
    allowed = True

    def stop_before_replace(path, value, *, before_replace):
        def stop_then_check(witness):
            nonlocal allowed
            allowed = False
            before_replace(witness)

        return replace(path, value, before_replace=stop_then_check)

    with monkeypatch.context() as stopping:
        stopping.setattr(diagnostics, "atomic_replace", stop_before_replace)
        stopped = store.repair(
            runtime_id=runtime.instance_id,
            owner=lambda: "successor",
            apply=True,
            automatic=True,
            mutation_allowed=lambda: allowed,
        )
    assert stopped["reason"] == "ownership_unverified"
    assert store.active_view()["total"] == 1
    assert store.history_view()["items"] == []


def test_large_agent_roster_keeps_repair_bounded_and_invalidates_stale_identity(store):
    runtime = MachineRuntime(store.runtime_root)
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    _observe(store, overdue_identity(runtime, "selected", "generation", 1))
    pid = os.getpid()
    runtime.paths["pid"].write_text(str(pid))
    roster = [f"{index:032x}" for index in range(2000)]

    def publish(instance_id):
        _publish_process_status(
            runtime,
            instance_id=instance_id,
            pid=pid,
            start_ticks=_pid_start_time_ticks(pid),
            waiting_for_first_registration=False,
            runtime_id=runtime.instance_id,
            reconciled_project_ids=roster,
        )

    publish("successor")
    status = runtime.paths["agent"] / "status.json"
    assert status.stat().st_size > 64 * 1024
    assert diagnostic_recovery.repair_scheduler_diagnostics(runtime, apply=False)["eligible"] == 1

    # Replacing status invalidates the projection until its writer republishes it.
    replacement = status.with_suffix(".replacement")
    replacement.write_bytes(status.read_bytes())
    replacement.replace(status)
    blocked = diagnostic_recovery.repair_scheduler_diagnostics(runtime, apply=True)
    assert blocked["reason"] == "ownership_unverified"
    assert blocked["retired"] == 0
    publish("successor")
    result = diagnostic_recovery.repair_scheduler_diagnostics(runtime, apply=True)
    assert result["backup_verified"] is True
    assert result["retired"] == 1
