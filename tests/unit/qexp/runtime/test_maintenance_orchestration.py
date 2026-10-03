from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parents[4]
RUNTIME_ROOT = PROJECT_ROOT / "src/qqtools/plugins/qexp/runtime"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            else:
                imports.update(alias.name for alias in node.names)
    return imports


def test_shared_maintenance_steps_have_narrow_named_contracts() -> None:
    from qqtools.plugins.qexp.runtime import maintenance_steps

    assert tuple(inspect.signature(maintenance_steps.advance_ready_index_step).parameters) == (
        "cfg",
        "cursor",
        "prior_degraded_reasons",
    )
    assert tuple(inspect.signature(maintenance_steps.advance_group_ready_members_step).parameters) == (
        "cfg",
        "cursor",
    )
    assert tuple(inspect.signature(maintenance_steps.advance_task_observation_step).parameters) == (
        "cfg",
        "cursor",
    )
    assert tuple(inspect.signature(maintenance_steps.advance_submission_control_step).parameters) == (
        "cfg",
        "cursor",
    )
    assert tuple(inspect.signature(maintenance_steps.advance_orphan_recovery_step).parameters) == (
        "cfg",
        "cursor",
        "reservation_runtime_root",
    )

    assert tuple(maintenance_steps.ReadyIndexStepResult.__dataclass_fields__) == (
        "cursor",
        "completed",
        "repaired",
        "blocked",
        "failure",
        "prior_degraded_reasons",
        "build_identity",
    )
    assert tuple(maintenance_steps.GroupReadyMembersStepResult.__dataclass_fields__) == (
        "cursor",
        "completed",
        "repaired",
        "blocked",
        "failure",
        "build_identity",
    )
    assert tuple(maintenance_steps.TaskObservationStepResult.__dataclass_fields__) == (
        "cursor",
        "completed",
        "detail",
        "failure",
        "meaningful_progress",
    )
    assert tuple(maintenance_steps.SubmissionControlStepResult.__dataclass_fields__) == (
        "cursor",
        "completed",
        "detail",
        "failure",
        "meaningful_progress",
    )
    assert tuple(maintenance_steps.OrphanRecoveryStepResult.__dataclass_fields__) == (
        "cursor",
        "completed",
        "repaired",
        "blocked",
        "failure",
        "meaningful_progress",
    )


def test_maintenance_facade_exposes_the_distinct_orchestration_owners() -> None:
    from qqtools.plugins.qexp.runtime import maintenance, maintenance_full_audit, maintenance_service

    assert maintenance.advance_full_audit is maintenance_full_audit.advance_full_audit
    assert maintenance.create_invocation_ledger is maintenance_full_audit.create_invocation_ledger
    assert maintenance.advance_maintenance_work is maintenance_service.advance_maintenance_work
    assert set(maintenance.__all__) == {
        "CONTEXT_RESOLUTION_OPERATIONS",
        "PHASES",
        "PHASE_OPERATION_RESERVATIONS",
        "advance_full_audit",
        "advance_maintenance_work",
        "create_invocation_ledger",
    }


def test_maintenance_orchestration_dependency_direction_is_acyclic() -> None:
    facade_imports = _imports(RUNTIME_ROOT / "maintenance.py")
    audit_imports = _imports(RUNTIME_ROOT / "maintenance_full_audit.py")
    service_imports = _imports(RUNTIME_ROOT / "maintenance_service.py")

    assert {"maintenance_full_audit", "maintenance_service"} <= facade_imports
    assert "maintenance_service" not in audit_imports
    assert "maintenance" not in audit_imports
    assert "maintenance_full_audit" in service_imports
    assert "maintenance" not in service_imports


def test_maintenance_service_uses_only_narrow_full_audit_interfaces() -> None:
    service_path = RUNTIME_ROOT / "maintenance_service.py"
    tree = ast.parse(service_path.read_text(encoding="utf-8"), filename=str(service_path))

    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "maintenance_full_audit"
    }

    assert referenced == {
        "PHASE_OPERATION_RESERVATIONS",
        "advance_full_audit",
        "create_invocation_ledger",
        "ensure_full_audit_outbox",
    }


def test_outbox_full_audit_delegates_with_the_service_invocation_ledger(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_full_audit, maintenance_service

    descriptor = {
        "identity": {
            "project_id": "project",
            "kind": "full_audit",
            "target_id": "project",
            "work_generation": "generation",
        },
        "state": "running",
        "phase": "submission",
        "cursor": {},
        "due_at": "2000-01-01T00:00:00+00:00",
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(maintenance_full_audit, "ensure_full_audit_outbox", lambda _cfg: None)
    monkeypatch.setattr(
        maintenance_service,
        "select_due_work",
        lambda _cfg, *, max_scan: {"descriptor": descriptor, "more": False, "next_due_at": None},
    )

    def advance(_cfg, **kwargs):
        captured.update(kwargs)
        return {"maintenance_state": "running", "phase": "cleanup", "cursor": {"offset": 1}}

    monkeypatch.setattr(maintenance_full_audit, "advance_full_audit", advance)
    monkeypatch.setattr(maintenance_service, "update_work", lambda *_args, **_kwargs: None)

    result = maintenance_service.advance_maintenance_work(
        SimpleNamespace(),
        reservation_runtime_root=None,
    )

    ledger = captured["ledger"]
    assert ledger.semantic_item_limit == 1
    assert captured == {
        "reservation_runtime_root": None,
        "max_work_items": 1,
        "ledger": ledger,
        "create_if_missing": False,
        "create_successor": False,
        "create_successor_on_change": True,
    }
    assert result["maintenance_state"] == "running"


def test_ready_step_copies_caller_owned_continuation(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_steps

    cursor = {"mode": "audit", "offset": 7}
    degraded_reasons = ["existing"]
    monkeypatch.setattr(maintenance_steps, "read_directory_entry", lambda *_args, **_kwargs: (None, 9))
    monkeypatch.setattr(maintenance_steps, "read_ready_index_state", lambda _cfg: "active")

    result = maintenance_steps.advance_ready_index_step(
        SimpleNamespace(shared_root=Path("/unused")),
        cursor=cursor,
        prior_degraded_reasons=degraded_reasons,
    )

    assert result.completed is True
    assert result.cursor == {}
    assert result.prior_degraded_reasons == ("existing",)
    assert cursor == {"mode": "audit", "offset": 7}
    assert degraded_reasons == ["existing"]


def test_orphan_step_copies_caller_owned_nested_continuation(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_steps

    cursor = {"orphan": {"stage": "attempt", "task_id": "task", "attempt_number": 1}}
    monkeypatch.setattr(Path, "exists", lambda _path: False)

    result = maintenance_steps.advance_orphan_recovery_step(
        SimpleNamespace(shared_root=Path("/unused")),
        cursor=cursor,
        reservation_runtime_root=None,
    )

    assert result.failure == {
        "code": "attempt_truth_missing",
        "phase": "orphan_recovery",
        "type": "Intervention",
    }
    assert result.meaningful_progress is False
    assert cursor == {"orphan": {"stage": "attempt", "task_id": "task", "attempt_number": 1}}


def test_orphan_attempt_step_preserves_baseline_progress_semantics(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_steps

    cursor = {"orphan": {"stage": "attempt", "task_id": "task", "attempt_number": 1}}
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr(maintenance_steps, "read_json_limited", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        maintenance_steps.AttemptRecord,
        "from_dict",
        lambda _value: SimpleNamespace(
            task_id="task",
            attempt_number=1,
            machine_name="machine",
            attempt_id="attempt",
            current_fencing_token=3,
            process={"process_group_id": 12, "process_group_start_time_ticks": 34},
        ),
    )

    result = maintenance_steps.advance_orphan_recovery_step(
        SimpleNamespace(shared_root=Path("/unused"), machine_name="machine"),
        cursor=cursor,
        reservation_runtime_root=None,
    )

    assert result.cursor["orphan"]["stage"] == "process"
    assert result.meaningful_progress is False
    assert cursor == {"orphan": {"stage": "attempt", "task_id": "task", "attempt_number": 1}}


def test_orphan_progress_override_survives_checkpoint_failure_path() -> None:
    from qqtools.plugins.qexp.runtime import maintenance_full_audit

    step = maintenance_full_audit.FullAuditStepResult(
        cursor={"orphan": {"stage": "process"}},
        completed=False,
        repaired=(),
        blocked=(),
        failure=None,
        detail=None,
        meaningful_progress=False,
    )

    assert not maintenance_full_audit._cursor_made_meaningful_progress(
        step,
        current_cursor=step.cursor,
        previous_cursor={"orphan": {"stage": "attempt"}},
    )


def test_observation_step_reports_progress_without_cursor_change(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_steps

    class Maintenance:
        def __init__(self, _cfg) -> None:
            self.closed = False

        def advance(self) -> dict[str, object]:
            return {"state": "building", "processed": 1}

        def close(self) -> None:
            self.closed = True

    cursor = {"requested": True}
    monkeypatch.setattr(maintenance_steps, "inspect_observation", lambda _cfg: {"state": "building"})
    monkeypatch.setattr(maintenance_steps, "ObservationMaintenance", Maintenance)

    result = maintenance_steps.advance_task_observation_step(object(), cursor=cursor)

    assert result.cursor == cursor
    assert result.completed is False
    assert result.meaningful_progress is True
    assert result.detail["processed"] == 1
    assert cursor == {"requested": True}


def test_ready_step_preserves_detected_reason_when_degradation_write_fails(monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import maintenance_full_audit, maintenance_steps

    diagnostic = object()
    issue = "observed-projection-damage"
    cfg = SimpleNamespace(shared_root=Path("/unused"))
    record = {"cursor": {"mode": "audit", "offset": 0}, "prior_degraded_reasons": []}
    monkeypatch.setattr(maintenance_steps, "read_directory_entry", lambda *_args, **_kwargs: ("task.json", 1))
    monkeypatch.setattr(maintenance_steps, "read_ready_index_state", lambda _cfg: "active")
    monkeypatch.setattr(maintenance_steps.TaskRecord, "from_dict", lambda _value: SimpleNamespace(task_id="task"))
    monkeypatch.setattr(maintenance_steps, "read_json_limited", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(maintenance_steps, "ready_task_projection_issue", lambda *_args: issue)
    monkeypatch.setattr(
        maintenance_steps,
        "parse_ready_reason",
        lambda _value: SimpleNamespace(diagnostic=diagnostic),
    )
    monkeypatch.setattr(
        maintenance_steps,
        "mark_ready_index_degraded",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("queue reconstruction in progress")),
    )

    with pytest.raises(RuntimeError, match="queue reconstruction in progress"):
        maintenance_full_audit._advance_ready_phase(cfg, record)

    assert record["prior_degraded_reasons"] == [issue]


def test_maintenance_domain_owners_do_not_import_command_or_repair_orchestration() -> None:
    step_owner_forbidden = {
        "doctor",
        "commands",
        "commands.cleanup",
        "qqtools.plugins.qexp.doctor",
        "qqtools.plugins.qexp.commands",
        "qqtools.plugins.qexp.commands.cleanup",
    }

    for filename in ("maintenance_steps.py", "cleanup_maintenance.py"):
        imports = _imports(RUNTIME_ROOT / filename)
        assert not imports & step_owner_forbidden, (
            f"{filename} reverses into command/repair orchestration: {sorted(imports & step_owner_forbidden)}"
        )

    facade_forbidden = {
        "doctor",
        "commands.cleanup",
        "qqtools.plugins.qexp.doctor",
        "qqtools.plugins.qexp.commands.cleanup",
    }
    imports = _imports(RUNTIME_ROOT / "maintenance.py")
    assert not imports & facade_forbidden, (
        f"maintenance.py reverses into repair/cleanup orchestration: {sorted(imports & facade_forbidden)}"
    )
