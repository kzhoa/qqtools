from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.project_maintenance import offer_due_tasks
from qqtools.plugins.qexp.runtime.active_operations import (
    iter_active_operation_paths,
    write_active_operation,
)
from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths
from qqtools.plugins.qexp.runtime.ready import (
    ReadyMarkerRef,
    classify_ready_marker,
    delete_ready_marker,
    delete_stale_ready_marker,
    reserve_ready_generation,
    write_ready_marker,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _ready_reference(cfg, task_id: str, generation: int) -> ReadyMarkerRef:
    path = shared_paths(cfg.shared_root)["ready_reservations"] / f"{task_id}.{generation}.json"
    record = read_json(path)["ready_reservation"]
    return ReadyMarkerRef(
        task_id,
        generation,
        record["queue_scope"],
        record["home_machine"],
        record["partition"],
        record["catalog_page"],
        record["marker_name"],
    )


def test_submission_commits_one_durable_ready_generation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")

    task = submit(cfg, ["echo", "ok"])

    stored = load_task(cfg, task.task_id)
    reference = _ready_reference(cfg, task.task_id, stored.ready_generation)
    result = classify_ready_marker(cfg, reference)
    assert stored.ready_generation == 1
    assert result.classification == "claimable"
    assert result.task is not None and result.task.task_id == task.task_id
    assert read_json(ready_state_path(cfg.shared_root))["ready_index"]["state"] == "absent"


def test_claim_commits_truth_before_retiring_ready_marker(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    reservation_path = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    )

    attempt = claim_task(cfg, task.task_id, [0])

    assert attempt is not None
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    assert not reservation_path.exists()


def test_marker_removed_by_successful_claim_is_permanently_stale(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    reference = _ready_reference(cfg, task.task_id, task.ready_generation)

    assert claim_task(cfg, task.task_id, [0]) is not None

    result = classify_ready_marker(cfg, reference)

    assert result.classification == "permanently_stale"
    assert result.reason == "task_not_queued"
    assert result.task is not None and result.task.task_id == task.task_id


def test_missing_queued_marker_with_indexed_slot_is_corrupt(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    reference = _ready_reference(cfg, task.task_id, task.ready_generation)
    marker_path = (
        shared_paths(cfg.shared_root)["ready_home"]
        / reference.home_machine
        / reference.partition
        / reference.marker_name
    )
    marker_path.unlink()

    result = classify_ready_marker(cfg, reference)

    assert result.classification == "corrupt"
    assert result.reason == "marker_missing_indexed"


def test_missing_queued_marker_and_slot_is_corrupt(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    reference = _ready_reference(cfg, task.task_id, task.ready_generation)
    marker_path = (
        shared_paths(cfg.shared_root)["ready_home"]
        / reference.home_machine
        / reference.partition
        / reference.marker_name
    )
    partition_path = (
        shared_paths(cfg.shared_root)["ready_home"]
        / reference.home_machine
        / reference.partition
        / "partition.json"
    )
    marker_path.unlink()
    partition = read_json(partition_path)
    partition["ready_partition"]["slots"].remove(reference.marker_name)
    atomic_replace(partition_path, partition)

    result = classify_ready_marker(cfg, reference)

    assert result.classification == "corrupt"
    assert result.reason == "marker_missing_unindexed"


def test_ready_deletion_tolerates_concurrent_reservation_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    reservation_path = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    )
    from qqtools.plugins.qexp.runtime import ready

    original_read_json = ready.read_json

    def remove_reservation_before_read(path):
        if path == reservation_path:
            reservation_path.unlink()
        return original_read_json(path)

    monkeypatch.setattr(ready, "read_json", remove_reservation_before_read)

    assert delete_ready_marker(cfg, task.task_id, task.ready_generation) is False


def test_ready_generation_publication_is_not_cleaned_as_stale(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    stored = load_task(cfg, task.task_id)
    generation = stored.ready_generation + 1
    reference = reserve_ready_generation(
        cfg,
        stored.task_id,
        generation,
        stored.placement_runtime["queue_scope"],
        stored.placement_policy["home_machine"],
    )
    reservation_path = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{stored.task_id}.{generation}.json"
    )

    before_marker = classify_ready_marker(cfg, reference)

    assert before_marker.classification == "temporarily_unavailable"
    assert before_marker.reason == "marker_publication_pending"
    assert delete_stale_ready_marker(cfg, reference) is False
    assert reservation_path.exists()

    write_ready_marker(
        cfg,
        stored,
        generation=generation,
        source_transition="test_publication",
        source_revision=stored.meta["revision"],
        target_revision=stored.meta["revision"] + 1,
        reference=reference,
    )

    after_marker = classify_ready_marker(cfg, reference)

    assert after_marker.classification == "temporarily_unavailable"
    assert after_marker.reason == "marker_publication_pending"
    assert delete_stale_ready_marker(cfg, reference) is False
    assert reservation_path.exists()


def test_ready_delete_failure_does_not_roll_back_authoritative_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.ready.delete_ready_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("shared storage unavailable")),
    )

    attempt = claim_task(cfg, task.task_id, [0])

    assert attempt is not None
    active_claim = load_task(cfg, task.task_id).claim_control["active_claim"]
    assert active_claim["attempt_id"] == attempt.attempt_id


def test_scope_change_writes_new_generation_before_retiring_old(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    original = submit(cfg, ["echo", "ok"], group="exp")
    old_reservation = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{original.task_id}.{original.ready_generation}.json"
    )

    task_commands.share(cfg, original.task_id)

    shared = load_task(cfg, original.task_id)
    reference = _ready_reference(cfg, shared.task_id, shared.ready_generation)
    assert shared.ready_generation == original.ready_generation + 1
    assert reference.queue_scope == "shared"
    assert not old_reservation.exists()
    assert classify_ready_marker(cfg, reference).classification == "claimable"


def test_offer_due_tasks_never_enumerates_task_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "ok"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    seen = []
    from qqtools.plugins.qexp.runtime import availability

    original_scandir = availability.os.scandir

    def record_directory(directory):
        seen.append(directory)
        assert directory != shared_paths(cfg.shared_root)["tasks"]
        return original_scandir(directory)

    monkeypatch.setattr(availability.os, "scandir", record_directory)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.elapsed_offer_is_proven",
        lambda *_args: True,
    )

    offer_due_tasks(cfg)

    assert len(seen) >= 1
    deadline_home = shared_paths(cfg.shared_root)["offer_deadlines_active"] / "g1"
    assert deadline_home in map(Path, seen)
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_deadline_index_is_partitioned_by_home_and_time_bucket(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "ok"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=60,
    )

    stable = shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task.task_id}.json"
    target = stable.resolve(strict=True)

    assert stable.is_symlink()
    assert target.parent.parent.name == "g1"
    assert len(target.parent.name) == 10


def test_active_operation_enumeration_is_hard_bounded(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    for index in range(65):
        write_active_operation(
            cfg,
            "availability",
            f"operation-{index:03d}",
            {"availability_operation": {"state": "prepared"}},
        )

    paths = list(iter_active_operation_paths(cfg, "availability", limit=64))
    next_paths = list(iter_active_operation_paths(cfg, "availability", limit=64))

    assert len(paths) == 64
    active_directory = shared_paths(cfg.shared_root)["availability_active"]
    assert all(path.parent == active_directory for path in paths)
    assert len(next_paths) == 1
    assert not set(paths).intersection(next_paths)
