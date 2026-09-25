from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.cli.errors import CliUsageError
from qqtools.plugins.qexp.cli.project_handlers import _split_machine_list
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.commands.group import change_worker, create_group, group_control
from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity
from qqtools.plugins.qexp.project_maintenance import offer_due_tasks
from qqtools.plugins.qexp.runtime.availability import offer_deadlines
from qqtools.plugins.qexp.runtime.availability import transitions as availability_runtime
from qqtools.plugins.qexp.runtime.availability.offer_deadlines import rebuild_deadline_indexes
from qqtools.plugins.qexp.runtime.maintenance import advance_maintenance_work
from qqtools.plugins.qexp.runtime.maintenance_outbox import read_work
from qqtools.plugins.qexp.runtime.operation_store import active_operation_path, write_active_operation
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt
from tests.helpers.qexp.clock import set_offer_evaluation_time

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


def _existing_group(cfg, name: str = "exp") -> None:
    create_group(cfg, name)


def _finish_repair(cfg, *, limit: int = 128) -> dict:
    """Run bounded full-audit slices until the captured audit completes."""
    for _ in range(limit):
        result = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root)
        if result["complete"] or result["outcome"] == "blocked":
            return result
    pytest.fail(f"full repair did not complete in {limit} bounded slices")


def test_share_now_persists_journal_audit_and_keeps_home_eligible(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="private", fallback_machines="group")

    result = task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    assert result.action == "share_now"
    assert stored.placement_policy["sharing_mode"] == "spillover"
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert stored.meta["revision"] == task.meta["revision"] + 1
    assert (shared_paths(cfg.shared_root)["availability"] / f"{result.operation_id}.json").exists()
    event_files = list(shared_paths(cfg.shared_root)["events"].glob(f"*/{result.operation_id}.json"))
    assert event_files
    assert claim_task(cfg, task.task_id, [0]) is not None


def test_due_deadline_cursor_rotates_past_first_bounded_batch(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    root = shared_paths(cfg.shared_root)["offer_deadlines_active"] / cfg.machine_name / bucket
    root.mkdir(parents=True, exist_ok=True)
    for index in range(65):
        atomic_replace(root / f"task-{index:03d}.json", {"offer_deadline": {"task_id": str(index)}})

    first = list(offer_deadlines.iter_due_deadline_paths(cfg))
    second = list(offer_deadlines.iter_due_deadline_paths(cfg))

    assert len(first) == 64
    assert len(second) == 1
    assert {path.name for path in first + second} == {f"task-{index:03d}.json" for index in range(65)}


def test_share_without_helpers_replaces_stale_private_fallback_with_group(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="private", fallback_machines=["legacy-helper"])

    task_commands.share(cfg, task.task_id)

    assert load_task(cfg, task.task_id).placement_policy["fallback_constraint"] == "group"


def test_repeated_same_share_is_idempotent_without_revision_change(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    change_worker(cfg, "exp", "g2", "add")

    first = task_commands.share(cfg, task.task_id, helper_machines=["g2"])
    revision = load_task(cfg, task.task_id).meta["revision"]
    second = task_commands.share(cfg, task.task_id, helper_machines=["g2"])

    assert first.operation_id != second.operation_id
    assert second.idempotent is True
    assert load_task(cfg, task.task_id).meta["revision"] == revision


def test_concurrent_availability_replay_reuses_completed_operation_without_reserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    entered_reservation = Event()
    release_reservation = Event()
    original_reserve = availability_runtime.reserve_ready_generation
    reserve_calls = 0

    def reserve_while_holding_task_lock(*args, **kwargs):
        nonlocal reserve_calls
        reference = original_reserve(*args, **kwargs)
        reserve_calls += 1
        if reserve_calls == 1:
            entered_reservation.set()
            assert release_reservation.wait(timeout=5)
        return reference

    monkeypatch.setattr(availability_runtime, "reserve_ready_generation", reserve_while_holding_task_lock)
    request = availability_runtime.AvailabilityTransitionRequest(
        action="share_now", task_id=task.task_id, operation_id="concurrent-share"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(availability_runtime.apply_availability_transition, cfg, request)
        assert entered_reservation.wait(timeout=5)
        second = pool.submit(availability_runtime.apply_availability_transition, cfg, request)
        release_reservation.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert reserve_calls == 1
    assert sorted(result.idempotent for result in results) == [False, True]
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_availability_replay_reads_archived_operation_after_active_path_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    request = availability_runtime.AvailabilityTransitionRequest(
        action="share_now", task_id=task.task_id, operation_id="archive-race"
    )
    availability_runtime.apply_availability_transition(cfg, request)
    active_path = active_operation_path(cfg, "availability", request.operation_id)
    archived_path = shared_paths(cfg.shared_root)["availability"] / f"{request.operation_id}.json"
    write_active_operation(cfg, "availability", request.operation_id, read_json(archived_path))
    original_read = availability_runtime.read_json
    has_archived = False

    def archive_before_read(path):
        nonlocal has_archived
        if path == active_path and not has_archived:
            has_archived = True
            availability_runtime.archive_operation(
                cfg, "availability", request.operation_id, original_read(archived_path)
            )
        return original_read(path)

    monkeypatch.setattr(availability_runtime, "read_json", archive_before_read)

    result = availability_runtime.apply_availability_transition(cfg, request)

    assert has_archived is True
    assert result.idempotent is True
    assert result.operation_id == request.operation_id


def test_replay_does_not_recreate_archived_operation_after_later_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    request = availability_runtime.AvailabilityTransitionRequest(
        action="share_now", task_id=task.task_id, operation_id="archived-share"
    )
    availability_runtime.apply_availability_transition(cfg, request)
    task_commands.keep_local(cfg, task.task_id)
    active_path = active_operation_path(cfg, "availability", request.operation_id)
    archived_path = shared_paths(cfg.shared_root)["availability"] / f"{request.operation_id}.json"
    archived_operation = read_json(archived_path)
    write_active_operation(cfg, "availability", request.operation_id, archived_operation)
    original_write = availability_runtime.write_active_operation
    original_exists = Path.exists
    active_exists_calls = 0

    def fail_if_recreated(*args, **kwargs):
        if args[2] == request.operation_id:
            pytest.fail("a completed availability operation must not be recreated")
        return original_write(*args, **kwargs)

    def archive_between_lookup_and_read(path):
        nonlocal active_exists_calls
        if path == active_path:
            active_exists_calls += 1
            if active_exists_calls == 2:
                availability_runtime.archive_operation(cfg, "availability", request.operation_id, archived_operation)
                return False
        return original_exists(path)

    monkeypatch.setattr(availability_runtime, "write_active_operation", fail_if_recreated)
    monkeypatch.setattr(Path, "exists", archive_between_lookup_and_read)

    replay = availability_runtime.apply_availability_transition(cfg, request)

    assert active_exists_calls == 2
    assert replay.idempotent is True
    assert replay.resulting_state == "shared"
    stored = load_task(cfg, task.task_id)
    assert stored.placement_policy["sharing_mode"] == "private"
    assert stored.placement_runtime["queue_scope"] == "home"


def test_reconcile_continues_when_enumerated_operation_is_archived_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    completed_task = submit(cfg, ["echo", "completed"], group="exp")
    completed_request = availability_runtime.AvailabilityTransitionRequest(
        action="share_now", task_id=completed_task.task_id, operation_id="a-archive-race"
    )
    availability_runtime.apply_availability_transition(cfg, completed_request)
    completed_active = active_operation_path(cfg, "availability", completed_request.operation_id)
    completed_archive = shared_paths(cfg.shared_root)["availability"] / f"{completed_request.operation_id}.json"
    write_active_operation(cfg, "availability", completed_request.operation_id, read_json(completed_archive))
    pending_task = submit(cfg, ["echo", "pending"], group="exp")
    pending_request = availability_runtime.AvailabilityTransitionRequest(
        action="share_now", task_id=pending_task.task_id, operation_id="z-pending"
    )
    availability_runtime._create_operation(cfg, pending_request)
    original_read = availability_runtime.read_json
    has_archived = False

    def archive_before_read(path):
        nonlocal has_archived
        if path == completed_active and not has_archived:
            has_archived = True
            availability_runtime.archive_operation(
                cfg, "availability", completed_request.operation_id, original_read(completed_archive)
            )
        return original_read(path)

    monkeypatch.setattr(availability_runtime, "read_json", archive_before_read)

    reconciled = availability_runtime.reconcile_availability_operations(cfg)

    assert has_archived is True
    assert [item["operation_id"] for item in reconciled] == [pending_request.operation_id]
    assert load_task(cfg, pending_task.task_id).placement_runtime["queue_scope"] == "shared"


def test_keep_local_is_idempotent_for_private_standalone_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    revision = load_task(cfg, task.task_id).meta["revision"]

    result = task_commands.keep_local(cfg, task.task_id)

    assert result.idempotent is True
    assert load_task(cfg, task.task_id).meta["revision"] == revision


def test_share_after_writes_and_repairs_deadline_index(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")

    result = task_commands.share(cfg, task.task_id, after_seconds=0)
    index_path = shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task.task_id}.json"
    assert index_path.exists()
    assert read_json(index_path)["offer_deadline"]["operation_id"] == result.operation_id

    index_path.unlink()
    issues = {issue["code"] for issue in verify_integrity(cfg)["issues"]}
    assert "offer_deadline_index_missing" in issues
    repaired = _finish_repair(cfg)
    assert repaired["complete"] is True
    assert index_path.exists()


def test_doctor_replays_prepared_availability_operation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    operation_id = "repair-share"
    operation_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    atomic_replace(
        operation_path,
        {
            "meta": {
                "schema_version": 6,
                "revision": 1,
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"},
            },
            "availability_operation": {
                "operation_id": operation_id,
                "operation_type": "share_now",
                "task_id": task.task_id,
                "state": "prepared",
                "requested_by": "g1",
                "reason": "manual",
                "helper_machines": None,
                "after_seconds": None,
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "completed_at": None,
                "blocked_reason": None,
                "task_revision_before": None,
                "task_revision_after": None,
                "result": None,
            },
        },
    )

    _finish_repair(cfg)

    stored = load_task(cfg, task.task_id)
    assert stored.placement_policy["sharing_mode"] == "spillover"
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert read_json(operation_path)["availability_operation"]["state"] == "completed"


def test_doctor_archives_blocked_availability_operation_with_reason(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    operation_id = "blocked-share"
    operation_path = active_operation_path(cfg, "availability", operation_id)
    write_active_operation(
        cfg,
        "availability",
        operation_id,
        {
            "meta": {
                "schema_version": 6,
                "revision": 1,
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"},
            },
            "availability_operation": {
                "operation_id": operation_id,
                "operation_type": "share_now",
                "task_id": task.task_id,
                "state": "blocked",
                "requested_by": "g1",
                "reason": "manual",
                "helper_machines": None,
                "after_seconds": None,
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "completed_at": None,
                "blocked_reason": "placement can only change while a Task is queued and unclaimed.",
                "task_revision_before": None,
                "task_revision_after": None,
                "result": None,
            },
        },
    )

    archived_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    assert archived_path.is_symlink()

    repaired = _finish_repair(cfg)

    assert operation_path.exists()
    assert archived_path.is_symlink()
    assert repaired["outcome"] == "blocked"
    assert any(operation_id in item and "blocked" in item for item in repaired["blocked"])


def test_doctor_completes_operation_after_post_task_side_effect_failure(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    original = offer_deadlines.sync_deadline_index
    calls = 0

    def fail_once(cfg, task):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("index unavailable")
        return original(cfg, task)

    monkeypatch.setattr(offer_deadlines, "sync_deadline_index", fail_once)
    with pytest.raises(OSError, match="index unavailable"):
        task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    operation_id = stored.placement_runtime["availability_operation_id"]
    operation_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    assert read_json(operation_path)["availability_operation"]["state"] == "prepared"

    _finish_repair(cfg)

    assert read_json(operation_path)["availability_operation"]["state"] == "completed"


def test_resident_recovers_truth_committed_before_descriptor_activation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")

    def fail_activation(*_args, **_kwargs):
        raise OSError("activation interrupted")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            "qqtools.plugins.qexp.runtime.maintenance_outbox.activate_work",
            fail_activation,
        )
        with pytest.raises(OSError, match="activation interrupted"):
            task_commands.share(cfg, task.task_id)

    [operation_path] = list(shared_paths(cfg.shared_root)["availability_active"].glob("*.json"))
    operation_id = read_json(operation_path)["availability_operation"]["operation_id"]
    descriptor = None
    for _ in range(16):
        result = advance_maintenance_work(cfg, reservation_runtime_root=cfg.runtime_root)
        descriptor = read_work(
            cfg,
            kind="availability",
            target_id=operation_id,
            work_generation=operation_id,
        )
        if descriptor is not None and descriptor["state"] == "completed":
            break

    assert result["maintenance_state"] == "completed"
    assert load_task(cfg, task.task_id).placement_policy["sharing_mode"] == "spillover"
    assert descriptor is not None and descriptor["state"] == "completed"


def test_share_rejects_claimed_task_without_mutation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    assert claim_task(cfg, task.task_id, [0]) is not None
    revision = load_task(cfg, task.task_id).meta["revision"]

    with pytest.raises(ValueError, match="queued and unclaimed"):
        task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    assert stored.meta["revision"] == revision
    assert stored.placement_policy["sharing_mode"] == "private"


def test_failed_availability_operation_does_not_make_doctor_unhealthy(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])

    with pytest.raises(ValueError, match="does not belong to a Group"):
        task_commands.share(cfg, task.task_id)

    verification = verify_integrity(cfg)
    while not verification["complete"]:
        verification = verify_integrity(cfg)
    assert verification["healthy"] is True


def test_cli_availability_json_and_human_outputs(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: True,
    )

    assert main([*_base_args(cfg), "task", "share", task.task_id, "--format=json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "share_now"
    assert payload["task_id"] == task.task_id
    assert payload["resulting_state"] == "shared"
    assert payload["outcome"] == "completed"

    assert main([*_base_args(cfg), "task", "unshare", task.task_id]) == 0
    human = capsys.readouterr().out
    assert "restricted to its home machine" in human
    assert human.count("Queue scope: home") == 1
    assert "Status: home" not in human


def test_cli_share_accepts_comma_separated_helper_machines(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    change_worker(cfg, "exp", "g2", "add")
    change_worker(cfg, "exp", "g3", "add")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: True,
    )

    assert (
        main(
            [
                *_base_args(cfg),
                "task",
                "share",
                task.task_id,
                "--with",
                "g2,g3",
            ]
        )
        == 0
    )

    assert load_task(cfg, task.task_id).placement_policy["fallback_constraint"] == ["g2", "g3"]


def test_cli_share_with_explicit_helper_preserves_home_claim_and_launch(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g3", runtime_root=tmp_path / "home-runtime")
    _existing_group(cfg)
    change_worker(cfg, "exp", "g2", "add")
    change_worker(cfg, "exp", "g4", "add")
    task = submit(cfg, ["echo", "ok"], group="exp")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active",
        lambda cfg, *, reason, **kwargs: True,
    )

    assert main([*_base_args(cfg), "task", "share", task.task_id, "--with", "g2"]) == 0

    stored = load_task(cfg, task.task_id)
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert stored.placement_policy["fallback_constraint"] == ["g2"]
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    expected_attempt_path = attempt_path(cfg.shared_root, task.task_id, 1)
    assert sorted(expected_attempt_path.parent.glob("*.json")) == [expected_attempt_path]
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)


def test_explicit_helper_can_claim_shared_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g3", runtime_root=tmp_path / "home-runtime")
    _existing_group(cfg)
    change_worker(cfg, "exp", "g2", "add")
    task = submit(cfg, ["echo", "ok"], group="exp")
    task_commands.share(cfg, task.task_id, helper_machines=["g2"])
    helper_cfg = replace(cfg, machine_name="g2", runtime_root=tmp_path / "helper-runtime")

    attempt = claim_task(helper_cfg, task.task_id, [0])

    assert attempt is not None
    assert attempt.machine_name == "g2"


def test_unlisted_worker_rejection_does_not_prevent_home_claim(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g3", runtime_root=tmp_path / "home-runtime")
    _existing_group(cfg)
    change_worker(cfg, "exp", "g2", "add")
    change_worker(cfg, "exp", "g4", "add")
    task = submit(cfg, ["echo", "ok"], group="exp")
    task_commands.share(cfg, task.task_id, helper_machines=["g2"])
    unlisted_cfg = replace(cfg, machine_name="g4", runtime_root=tmp_path / "unlisted-runtime")

    assert claim_task(unlisted_cfg, task.task_id, [0]) is None
    stored = load_task(cfg, task.task_id)
    assert stored.claim_control["active_claim"] is None
    assert not attempt_path(cfg.shared_root, task.task_id, 1).exists()

    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert attempt.machine_name == "g3"


@pytest.mark.parametrize("barrier", ["group_pause", "home_drain"])
def test_home_launch_revalidation_rejects_changed_group_authority(tmp_path: Path, barrier: str):
    cfg = init_shared_root(tmp_path / ".qexp", "g3", runtime_root=tmp_path / "home-runtime")
    _existing_group(cfg)
    change_worker(cfg, "exp", "g2", "add")
    task = submit(cfg, ["echo", "ok"], group="exp")
    task_commands.share(cfg, task.task_id, helper_machines=["g2"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    if barrier == "group_pause":
        group_control(cfg, "exp", "pause")
    else:
        change_worker(cfg, "exp", "g3", "drain")

    assert not authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)


def test_cli_share_rejects_empty_comma_separated_helper_machine():
    with pytest.raises(CliUsageError, match="non-empty"):
        _split_machine_list(["g2,,g3"])


def test_rebuild_deadline_indexes_removes_stale_index(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    stale = shared_paths(cfg.shared_root)["offer_deadlines"] / "missing.json"
    atomic_replace(stale, {"offer_deadline": {"task_id": "missing"}})

    assert rebuild_deadline_indexes(cfg) == 1
    assert not stale.exists()


def test_agent_ignores_stale_cold_deadline_index_without_skipping_due_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    stale = shared_paths(cfg.shared_root)["offer_deadlines"] / "missing.json"
    atomic_replace(stale, {"offer_deadline": {"task_id": "missing"}})
    task = submit(cfg, ["echo", "ok"], group="exp")
    task_commands.share(cfg, task.task_id, after_seconds=0)
    set_offer_evaluation_time(monkeypatch, cfg, task.task_id)

    offer_due_tasks(cfg)

    assert stale.exists()
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_offer_due_tasks_skips_stale_deadline_for_claimed_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(
        cfg,
        ["echo", "ok"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.elapsed_offer_is_proven",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.offer",
        lambda *_args, **_kwargs: pytest.fail("claimed Tasks must not be offered"),
    )

    offer_due_tasks(cfg)

    assert load_task(cfg, task.task_id).claim_control["active_claim"]["attempt_id"] == attempt.attempt_id
    assert not list(shared_paths(cfg.shared_root)["availability_active"].glob("*.json"))


def test_offer_due_tasks_skips_stale_deadline_for_terminal_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(
        cfg,
        ["echo", "ok"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.elapsed_offer_is_proven",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.offer",
        lambda *_args, **_kwargs: pytest.fail("terminal Tasks must not be offered"),
    )

    offer_due_tasks(cfg)

    assert load_task(cfg, task.task_id).state["projection"] == "failed"
    assert not list(shared_paths(cfg.shared_root)["availability_active"].glob("*.json"))
