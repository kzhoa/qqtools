import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.doctor import repair_metadata
from qqtools.plugins.qexp.runtime.maintenance import advance_maintenance_work
from qqtools.plugins.qexp.runtime.maintenance_outbox import (
    activate_work,
    prepare_work,
    read_work,
    retire_work,
    select_due_work,
    update_work,
)
from qqtools.plugins.qexp.runtime.project_activation import read_project_activation
from qqtools.plugins.qexp.runtime.ready import rebuild as ready_rebuild
from qqtools.plugins.qexp.runtime.ready import state as ready_state
from qqtools.plugins.qexp.runtime.ready.diagnostics import build_diagnostic
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _base_args(cfg) -> list[str]:
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


def _maintenance_descriptor(cfg) -> tuple[Path, dict]:
    root = cfg.shared_root / "operations" / "maintenance-v1"
    pointer = read_json(root / "full-audit.current.json")
    path = root / pointer["descriptor_file"]
    return path, read_json(path)


def test_full_repair_resumes_one_shared_budget_item_at_a_time(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    results = []
    for _ in range(64):
        result = repair_metadata(
            cfg,
            reservation_runtime_root=cfg.runtime_root,
            max_work_items=1,
        )
        results.append(result)
        budget = result["budget"]
        assert budget["semantic_items_consumed"] <= 1
        assert budget["operations_consumed"] <= 256
        assert budget["semantic_items_remaining"] >= 0
        assert budget["operations_remaining"] >= 0
        assert result["scope"]["kind"] == "full_audit"
        assert result["scope"]["capture_id"] == result["scope"]["work_generation"]
        assert result["remaining_work"] is None
        if result["complete"]:
            break

    assert results[-1]["complete"] is True
    assert results[-1]["rerun_required"] is False
    assert results[-1]["outcome"] in {"no_change", "repaired"}
    generation = results[0]["scope"]["work_generation"]
    assert all(item["scope"]["work_generation"] == generation for item in results)
    assert len({item["phase"] for item in results}) > 1

    successor = repair_metadata(
        cfg,
        reservation_runtime_root=cfg.runtime_root,
        max_work_items=1,
    )
    assert successor["scope"]["work_generation"] != generation


def test_partial_repair_json_and_strict_exit_contract(tmp_path: Path, capsys) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    base = _base_args(cfg)

    assert main([*base, "admin", "repair", "--max-work-items", "1", "--format=json"]) == 0
    partial = json.loads(capsys.readouterr().out)
    assert partial["complete"] is False
    assert partial["outcome"] == "partial"
    assert partial["rerun_required"] is True
    assert partial["budget"]["semantic_items_consumed"] == 1
    assert partial["budget"]["exhaustion_reason"] == "semantic_items"
    assert "incomplete" in partial["message"].lower()
    assert partial["next_action"]

    assert (
        main(
            [
                *base,
                "admin",
                "repair",
                "--max-work-items",
                "1",
                "--strict",
                "--format=json",
            ]
        )
        == 1
    )
    strict_partial = json.loads(capsys.readouterr().out)
    assert strict_partial["complete"] is False
    assert strict_partial["outcome"] == "partial"
    assert strict_partial["scope"]["work_generation"] == partial["scope"]["work_generation"]


def test_full_repair_report_bounds_deferred_phase_details(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    result = repair_metadata(
        cfg,
        reservation_runtime_root=cfg.runtime_root,
        max_work_items=1,
    )

    assert len(result["deferred_phases"]) <= 16
    assert result["deferred_phase_count"] >= len(result["deferred_phases"])
    assert isinstance(result["cursor"], dict)
    assert result["next_due_at"] is None or isinstance(result["next_due_at"], str)


def test_full_audit_cursor_resumes_across_runtime_roots(tmp_path: Path) -> None:
    cfg_a = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt-a")
    cfg_b = RootConfig(cfg_a.shared_root, cfg_a.project_root, cfg_a.machine_name, tmp_path / "rt-b")

    results = []
    for index in range(128):
        cfg = cfg_a if index % 2 == 0 else cfg_b
        result = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root, max_work_items=1)
        results.append(result)
        assert result["intervention"] is None
        if result["complete"]:
            break

    assert results[-1]["complete"] is True
    generation = results[0]["scope"]["work_generation"]
    assert all(result["scope"]["work_generation"] == generation for result in results)


def test_stage1_descriptor_migration_fails_closed_without_capture_identity(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root, max_work_items=1)
    descriptor_path, descriptor = _maintenance_descriptor(cfg)
    stage2_fields = {
        "source_capture",
        "source_revision",
        "build_identity",
        "meaningful_progress_at",
        "due_at",
        "retry_count",
    }
    for field in stage2_fields:
        descriptor.pop(field)
    descriptor["schema_version"] = 1
    atomic_replace(descriptor_path, descriptor)
    pointer_path = descriptor_path.parent / "full-audit.current.json"
    pointer = read_json(pointer_path)
    pointer["schema_version"] = 1
    atomic_replace(pointer_path, pointer)

    migrated = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root, max_work_items=1)

    assert migrated["outcome"] == "blocked"
    assert migrated["intervention"]["code"] == "legacy_capture_identity_unknown"
    _, persisted = _maintenance_descriptor(cfg)
    assert persisted["schema_version"] == 2
    assert persisted["state"] == "intervention"


def test_maintenance_outbox_selects_queue_order_not_directory_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    records = [
        activate_work(
            cfg,
            prepare_work(cfg, kind="ready", target_id=f"target-{index}", work_generation=f"work-{index}"),
        )
        for index in range(3)
    ]

    def reverse_directory_entry(path: Path, offset: int) -> tuple[str | None, int]:
        names = sorted((item.name for item in path.iterdir()), reverse=True)
        return (names[offset], offset + 1) if offset < len(names) else (None, offset)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.maintenance_outbox.read_directory_entry",
        reverse_directory_entry,
    )

    selected = select_due_work(cfg)

    assert selected["descriptor"]["identity"] == records[0]["identity"]


def test_maintenance_outbox_restart_preserves_rotation_and_new_arrivals_join_tail(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    first = activate_work(cfg, prepare_work(cfg, kind="ready", target_id="first", work_generation="first"))
    second = activate_work(cfg, prepare_work(cfg, kind="ready", target_id="second", work_generation="second"))

    assert select_due_work(cfg)["descriptor"]["identity"] == first["identity"]
    retire_work(cfg, kind="ready", target_id="first", work_generation="first", proof={"test": True})
    third = activate_work(cfg, prepare_work(cfg, kind="ready", target_id="third", work_generation="third"))

    assert select_due_work(cfg)["descriptor"]["identity"] == second["identity"]
    retire_work(cfg, kind="ready", target_id="second", work_generation="second", proof={"test": True})
    assert select_due_work(cfg)["descriptor"]["identity"] == third["identity"]


def test_maintenance_selection_is_independent_of_retired_queue_history(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    for index in range(70):
        record = activate_work(
            cfg,
            prepare_work(cfg, kind="ready", target_id=f"old-{index}", work_generation=f"old-{index}"),
        )
        retire_work(
            cfg,
            kind="ready",
            target_id=record["identity"]["target_id"],
            work_generation=record["identity"]["work_generation"],
            proof={"test": True},
        )
    current = activate_work(cfg, prepare_work(cfg, kind="ready", target_id="current", work_generation="current"))

    selected = select_due_work(cfg, max_scan=1)

    assert selected["descriptor"]["identity"] == current["identity"]


def test_future_due_cycle_yields_with_earliest_retry(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    due_times = ["2099-01-02T00:00:00+00:00", "2099-01-01T00:00:00+00:00"]
    for index, due_at in enumerate(due_times):
        activate_work(cfg, prepare_work(cfg, kind="ready", target_id=f"future-{index}", work_generation=f"w-{index}"))
        update_work(
            cfg,
            kind="ready",
            target_id=f"future-{index}",
            work_generation=f"w-{index}",
            state="waiting",
            due_at=due_at,
            publish_activation=False,
        )

    first = select_due_work(cfg, max_scan=1)
    second = select_due_work(cfg, max_scan=1)

    assert first["descriptor"] is None and first["more"] is True
    assert second == {"descriptor": None, "next_due_at": due_times[1], "more": False}


def test_prepared_operation_waits_for_producer_handoff(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    prepared = prepare_work(
        cfg,
        kind="availability",
        target_id="missing-operation",
        work_generation="missing-operation",
    )

    result = advance_maintenance_work(cfg, reservation_runtime_root=cfg.runtime_root)
    persisted = read_work(
        cfg,
        kind="availability",
        target_id="missing-operation",
        work_generation="missing-operation",
    )

    assert result["maintenance_state"] == "waiting"
    assert persisted is not None
    assert persisted["state"] == "prepared"
    assert persisted["failure"]["code"] == "producer_handoff_pending"
    assert persisted["identity"] == prepared["identity"]


def test_outbox_fails_closed_when_committed_descriptor_progress_is_lost(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    prepared = prepare_work(
        cfg,
        kind="availability",
        target_id="crashed-prepare",
        work_generation="crashed-prepare",
        phase="operation",
        cursor={"operation_key": "crashed-prepare"},
    )
    active = cfg.shared_root / "operations" / "maintenance-v1" / "active"
    [descriptor_path] = list(active.glob("*.json"))
    descriptor_path.unlink()

    selected = select_due_work(cfg)
    persisted = read_work(
        cfg,
        kind="availability",
        target_id="crashed-prepare",
        work_generation="crashed-prepare",
    )

    assert selected["descriptor"] is None
    assert persisted is not None
    assert persisted["identity"] == prepared["identity"]
    assert persisted["state"] == "intervention"
    assert persisted["retirement_proof"]["code"] == "maintenance_progress_lost"


def test_queue_reconstruction_does_not_reset_lost_progress(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    active = activate_work(
        cfg,
        prepare_work(
            cfg,
            kind="ready",
            target_id="reconstruct-progress",
            work_generation="generation",
            cursor={"step": 0},
        ),
    )
    progressed = update_work(
        cfg,
        kind="ready",
        target_id="reconstruct-progress",
        work_generation="generation",
        cursor={"step": 1},
        publish_activation=False,
    )
    root = cfg.shared_root / "operations" / "maintenance-v1"
    [descriptor_path] = list((root / "active").glob("*.json"))
    descriptor_path.unlink()
    (root / "queue-index.sqlite3").unlink()

    selected = select_due_work(cfg)
    persisted = read_work(cfg, kind="ready", target_id="reconstruct-progress", work_generation="generation")

    assert active["identity"] == progressed["identity"]
    assert selected["descriptor"] is None
    assert persisted is not None
    assert persisted["state"] == "intervention"
    assert persisted["retirement_proof"]["code"] == "maintenance_progress_lost"


def test_stale_activation_cannot_reopen_retired_generation(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    active = activate_work(
        cfg,
        prepare_work(cfg, kind="ready", target_id="retired", work_generation="generation"),
    )
    retired = retire_work(
        cfg,
        kind="ready",
        target_id="retired",
        work_generation="generation",
        proof={"test": True},
    )

    replayed = activate_work(cfg, active)
    prepared_again = prepare_work(cfg, kind="ready", target_id="retired", work_generation="generation")

    assert retired["state"] == "completed"
    assert replayed["state"] == "completed"
    assert prepared_again["state"] == "completed"
    assert replayed["progress_revision"] == retired["progress_revision"]
    assert select_due_work(cfg)["descriptor"] is None


def test_retirement_record_survives_crash_before_identity_index_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    activate_work(cfg, prepare_work(cfg, kind="ready", target_id="retiring", work_generation="generation"))
    from qqtools.plugins.qexp.runtime import maintenance_outbox

    replace = maintenance_outbox.atomic_replace

    def crash_before_index_flip(path: Path, value: dict, *args, **kwargs):
        if path.parent.name == "index" and value.get("location") == "retired":
            raise OSError("crash before retirement index flip")
        return replace(path, value, *args, **kwargs)

    with monkeypatch.context() as crash:
        crash.setattr(maintenance_outbox, "atomic_replace", crash_before_index_flip)
        with pytest.raises(OSError, match="crash before retirement index flip"):
            retire_work(
                cfg,
                kind="ready",
                target_id="retiring",
                work_generation="generation",
                proof={"test": True},
            )

    assert select_due_work(cfg)["descriptor"] is None
    persisted = read_work(cfg, kind="ready", target_id="retiring", work_generation="generation")
    assert persisted is not None
    assert persisted["state"] == "completed"


def test_long_descriptor_identity_publishes_bounded_exact_activation_key(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    record = prepare_work(
        cfg,
        kind="ready",
        target_id="t" * 96,
        work_generation="g" * 96,
    )

    activated = activate_work(cfg, record)
    checkpoint = read_project_activation(cfg.shared_root)["project_activation"]

    assert checkpoint["reason"].startswith("mw:")
    assert checkpoint["reason"].endswith(f":{activated['progress_revision']}")
    assert len(checkpoint["reason"].encode("utf-8")) <= 128


def test_ready_repair_recovers_reset_before_descriptor_activation(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    path = ready_state.ready_state_path(cfg.shared_root)
    value, record = ready_state.read_state_record(cfg)
    ready_state.degrade_state_record(
        record,
        build_diagnostic("build_invalid", stage="build_state", object_name="build"),
        cfg=cfg,
    )
    ready_state.commit_state_under_lock(path, value, record)

    def interrupt_reset(_cfg, _build_id):
        raise OSError("before ready projection reset")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(ready_rebuild, "_reset_ready_projection_for_repair", interrupt_reset)
        with pytest.raises(OSError, match="before ready projection reset"):
            ready_rebuild.repair_ready_index(cfg, max_tasks=1, bounded_initialization=True)

    interrupted_status = ready_state.read_ready_index_status(cfg)
    build_id = interrupted_status["build"]["build_id"]
    assert interrupted_status["state"] == "building"
    assert interrupted_status["build"]["phase"] == "reset-projection"
    selected = select_due_work(cfg)["descriptor"]
    assert selected is not None
    assert selected["state"] == "prepared"
    assert selected["cursor"]["build_id"] == build_id

    result = advance_maintenance_work(cfg, reservation_runtime_root=cfg.runtime_root)

    recovered = ready_state.read_ready_index_status(cfg)
    assert result["maintenance_state"] != "intervention"
    assert recovered["state"] == "building"
    assert recovered["build"]["build_id"] == build_id
    assert recovered["build"]["phase"] != "reset-projection"
    assert (_ready_build_archive(cfg, build_id)).is_dir()


def _ready_build_archive(cfg, build_id: str) -> Path:
    return cfg.shared_root / "indexes" / "ready" / "builds" / build_id / "replaced-projection"
