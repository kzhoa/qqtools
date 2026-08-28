from __future__ import annotations

import threading
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity
from qqtools.plugins.qexp.layout import project_id
from qqtools.plugins.qexp.machine_state import publish_machine_snapshots
from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths
from qqtools.plugins.qexp.runtime.locks import exclusive, schema_lock
from qqtools.plugins.qexp.runtime.ready import (
    READY_WRITER_CAPABILITY,
    advance_ready_index_build,
    assert_ready_writer_compatible,
    classify_ready_marker,
    next_ready_marker,
    read_ready_index_state,
)
from qqtools.plugins.qexp.runtime.records import TaskRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task


def _ready_reference(cfg, task: TaskRecord):
    from qqtools.plugins.qexp.runtime.ready import ReadyMarkerRef

    path = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    )
    record = read_json(path)["ready_reservation"]
    return ReadyMarkerRef(
        task.task_id,
        task.ready_generation,
        record["queue_scope"],
        record["home_machine"],
        record["partition"],
        record["catalog_page"],
        record["marker_name"],
    )


def _make_legacy_task(cfg, task_id: str) -> TaskRecord:
    task = submit(cfg, ["echo", task_id], task_id=task_id)
    reservation = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    )
    record = read_json(reservation)["ready_reservation"]
    marker = (
        shared_paths(cfg.shared_root)[
            "ready_home" if record["queue_scope"] == "home" else "ready_shared"
        ]
        / (record["home_machine"] if record["queue_scope"] == "home" else "")
        / record["partition"]
        / record["marker_name"]
    )
    marker.unlink()
    reservation.unlink()
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    value = read_json(task_path)
    value["task"]["ready_generation"] = 0
    atomic_replace(task_path, value)
    return TaskRecord.from_dict(value)


def _finish_build(cfg, *, max_tasks: int = 64) -> dict:
    record = advance_ready_index_build(cfg, max_tasks=max_tasks)
    while record["state"] == "building":
        record = advance_ready_index_build(cfg, max_tasks=max_tasks)
    return record


def test_build_backfills_legacy_task_and_activates_atomically(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    legacy = _make_legacy_task(cfg, "legacy-task")

    record = _finish_build(cfg, max_tasks=1)

    rebuilt = load_task(cfg, legacy.task_id)
    schema = read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")["schema"]
    assert record["state"] == "active"
    assert record["build"]["watermark"]["task_count"] == 1
    assert record["build"]["processed"] == 1
    assert schema["writer_capabilities"] == [READY_WRITER_CAPABILITY]
    assert rebuilt.ready_generation == 1
    assert classify_ready_marker(cfg, _ready_reference(cfg, rebuilt)).classification == "claimable"


def test_build_gate_waits_for_inflight_schema_writer(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    started = threading.Event()
    finished = threading.Event()

    def start_build() -> None:
        started.set()
        advance_ready_index_build(cfg)
        finished.set()

    with schema_lock(cfg.shared_root):
        thread = threading.Thread(target=start_build)
        thread.start()
        assert started.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        assert read_ready_index_state(cfg) == "absent"
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert read_ready_index_state(cfg) == "building"


def test_build_cursor_is_persistent_and_batch_bounded(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    for index in range(3):
        _make_legacy_task(cfg, f"legacy-{index}")

    first = advance_ready_index_build(cfg, max_tasks=2)
    persisted = read_json(ready_state_path(cfg.shared_root))["ready_index"]

    assert first["state"] == "building"
    assert first["build"]["processed"] == 2
    assert persisted["build"]["cursor"] == first["build"]["cursor"]
    assert _finish_build(cfg, max_tasks=2)["state"] == "active"
    assert all(load_task(cfg, f"legacy-{index}").ready_generation == 1 for index in range(3))


def test_task_created_during_build_is_covered_by_writer_protocol(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    _make_legacy_task(cfg, "legacy-task")
    first = advance_ready_index_build(cfg, max_tasks=1)
    assert first["state"] == "building"

    concurrent = submit(cfg, ["echo", "concurrent"], task_id="concurrent-task")
    record = _finish_build(cfg, max_tasks=1)

    assert record["state"] == "active"
    assert record["build"]["watermark"]["task_count"] == 1
    assert classify_ready_marker(
        cfg, _ready_reference(cfg, load_task(cfg, concurrent.task_id))
    ).classification == "claimable"


def test_mixed_writer_is_rejected_before_task_mutation(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "safe"], task_id="safe-task")
    _finish_build(cfg)
    path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="requires writer capability"):
        assert_ready_writer_compatible(cfg, writer_capability=None)

    assert path.read_bytes() == before


def test_reinitialization_preserves_active_writer_gate(tmp_path: Path) -> None:
    root = tmp_path / ".qexp"
    runtime_root = tmp_path / "rt"
    cfg = init_shared_root(root, "gpu-1", runtime_root=runtime_root)
    submit(cfg, ["echo", "first"], task_id="first-task")
    _finish_build(cfg)
    schema_path = shared_paths(root)["schema"] / "version.json"
    original_schema = read_json(schema_path)

    cfg = init_shared_root(root, "gpu-1", runtime_root=runtime_root)
    second = submit(cfg, ["echo", "second"], task_id="second-task")

    assert read_json(schema_path) == original_schema
    assert load_task(cfg, second.task_id).task_id == "second-task"


def test_final_audit_rejects_non_list_catalog_partitions(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "corrupt"], task_id="corrupt-task")
    first = advance_ready_index_build(cfg, max_tasks=1)
    assert first["build"]["phase"] == "audit"
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    catalog_path = (
        shared_paths(cfg.shared_root)["ready_catalogs"]
        / f"home.{cfg.machine_name}"
        / f"{reference.catalog_page:016d}.json"
    )
    catalog = read_json(catalog_path)
    catalog["ready_catalog"]["partitions"] = reference.partition
    atomic_replace(catalog_path, catalog)

    record = advance_ready_index_build(cfg, max_tasks=1)

    assert record["state"] == "degraded"
    assert f"marker_unindexed:{task.task_id}" in record["degraded_reasons"]


def test_final_audit_rechecks_schema_writer_gate(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "gate"], task_id="gate-task")
    first = advance_ready_index_build(cfg, max_tasks=1)
    assert first["build"]["phase"] == "audit"
    schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    schema = read_json(schema_path)
    del schema["schema"]["writer_capabilities"]
    atomic_replace(schema_path, schema)

    record = advance_ready_index_build(cfg, max_tasks=1)

    assert record["state"] == "degraded"
    assert any("capability gate is missing" in reason for reason in record["degraded_reasons"])


def test_missing_referenced_partition_degrades_active_index(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "missing"], task_id="missing-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    partition_path = (
        shared_paths(cfg.shared_root)["ready_home"]
        / cfg.machine_name
        / reference.partition
        / "partition.json"
    )
    partition_path.unlink()

    candidate, _has_wrapped = next_ready_marker(
        cfg, project_id(cfg.shared_root), "home"
    )

    assert candidate is None
    assert read_ready_index_state(cfg) == "degraded"


def test_partition_removal_recheck_avoids_false_degradation(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "removed"], task_id="removed-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    from qqtools.plugins.qexp.scheduler import claim_task

    assert claim_task(cfg, task.task_id, [0]) is not None
    route_key = f"home.{cfg.machine_name}"
    route_lock = shared_paths(cfg.shared_root)["ready_locks"] / f"{route_key}.lock"
    partition_path = (
        shared_paths(cfg.shared_root)["ready_home"]
        / cfg.machine_name
        / reference.partition
        / "partition.json"
    )
    catalog_path = (
        shared_paths(cfg.shared_root)["ready_catalogs"]
        / route_key
        / f"{reference.catalog_page:016d}.json"
    )
    started = threading.Event()
    finished = threading.Event()

    def read_candidate() -> None:
        started.set()
        next_ready_marker(cfg, project_id(cfg.shared_root), "home")
        finished.set()

    with exclusive(route_lock):
        partition_path.unlink()
        thread = threading.Thread(target=read_candidate)
        thread.start()
        assert started.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        catalog = read_json(catalog_path)
        catalog["ready_catalog"]["partitions"].remove(reference.partition)
        atomic_replace(catalog_path, catalog)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert read_ready_index_state(cfg) == "active"


def test_recent_incompatible_agent_blocks_cutover(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "queued"], task_id="queued-task")
    publish_machine_snapshots(
        cfg,
        instance_id="legacy-agent",
        pid=123,
        agent_mode="machine",
        observed_state="idle",
        active_attempt_ids=[],
        visible_gpu_ids=[0],
        reserved_gpu_ids=[],
        heartbeat_interval_seconds=5,
        started_at="2026-08-28T00:00:00Z",
        idle_since_at=None,
    )
    agent_path = cfg.shared_root / "machines" / "gpu-1" / "state" / "agent.json"
    agent = read_json(agent_path)
    del agent["agent"]["writer_capability"]
    atomic_replace(agent_path, agent)

    record = _finish_build(cfg)

    assert record["state"] == "degraded"
    assert any(
        reason == "incompatible_active_writers:gpu-1"
        for reason in record["degraded_reasons"]
    )


def test_doctor_reports_and_repairs_missing_active_marker(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "repair"], task_id="repair-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    marker = (
        shared_paths(cfg.shared_root)["ready_home"]
        / reference.home_machine
        / reference.partition
        / reference.marker_name
    )
    marker.unlink()

    verification = verify_integrity(cfg, reservation_runtime_root=cfg.runtime_root)
    repaired = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root)
    current = load_task(cfg, task.task_id)

    assert any(issue["code"] == "ready_projection_inconsistent" for issue in verification["issues"])
    assert repaired["ready_index"]["state"] == "active"
    assert current.ready_generation > task.ready_generation
    assert classify_ready_marker(cfg, _ready_reference(cfg, current)).classification == "claimable"


def test_doctor_rebuilds_corrupt_catalog_from_task_truth(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "repair"], task_id="catalog-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    catalog = (
        shared_paths(cfg.shared_root)["ready_catalogs"]
        / f"home.{cfg.machine_name}"
        / f"{reference.catalog_page:016d}.json"
    )
    atomic_replace(catalog, {"corrupt": {}})

    repaired = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root)
    current = load_task(cfg, task.task_id)

    assert repaired["ready_index"]["state"] == "active"
    assert current.ready_generation > task.ready_generation
    assert classify_ready_marker(cfg, _ready_reference(cfg, current)).classification == "claimable"


def test_nonqueued_task_needs_no_marker_at_cutover(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "running"], task_id="running-task")
    from qqtools.plugins.qexp.scheduler import claim_task

    claim_task(cfg, task.task_id, [0])

    assert _finish_build(cfg)["state"] == "active"
    assert read_ready_index_state(cfg) == "active"
