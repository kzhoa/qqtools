"""Worker removal covers live membership and changes without Task history scans."""

import os
from dataclasses import replace

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.commands.group import change_worker, reconcile_group_cancel_operations
from qqtools.plugins.qexp.commands.task import keep_local, retry, share
from qqtools.plugins.qexp.runtime.authority_lock import authority_locks
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from qqtools.plugins.qexp.runtime.operation_store import locate_operation_path
from qqtools.plugins.qexp.runtime.paths import group_path, task_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, cancel_task, claim_task, fail_attempt
from tests.helpers.qexp_discovery import discover_group as discover
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def control(cfg, operation_id):
    return read_json(locate_operation_path(cfg, "group_control", operation_id))["group_control"]


def advance_until(cfg, operation_id, predicate, *, steps=100):
    for _ in range(steps):
        current = control(cfg, operation_id)
        if predicate(current):
            return current
        reconcile_group_cancel_operations(cfg, "experiment", include_legacy=False)
    raise AssertionError(f"worker removal stalled: {current}")


def finish(cfg, operation_id):
    return advance_until(cfg, operation_id, lambda value: value["state"] == "completed")


def remove(cfg, *, terminate_running=False):
    return change_worker(cfg, "experiment", "g1", "remove", terminate_running=terminate_running)["worker_control"][
        "operation_id"
    ]


def guard_task_scans(monkeypatch, cfg):
    original = os.scandir

    def guarded(path):
        if not isinstance(path, int):
            assert os.path.abspath(os.fspath(path)) != str(cfg.shared_root / "tasks"), "removal scanned Task history"
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)


def test_removal_waits_for_coverage_without_history_fallback(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    with monkeypatch.context() as patch:
        guard_task_scans(patch, cfg)
        operation_id = remove(cfg)
        for _ in range(4):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
        assert control(cfg, operation_id)["state"] != "completed"
    assert load_task(cfg, task.task_id).state["projection"] == "queued"
    discover(cfg)
    with monkeypatch.context() as patch:
        guard_task_scans(patch, cfg)
        result = advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    assert result["state"] == "waiting_ack"
    assert read_json(group_path(cfg.shared_root, "experiment"))["group"]["worker_set"]["g1"]["state"] == "draining"


def test_census_reads_one_member_per_step_and_never_unrelated_history(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    tasks = [submit(cfg, ["true"], group="experiment") for _ in range(6)]
    for task in tasks:
        cancel_task(cfg, task.task_id)
    discover(cfg)
    task_path(cfg.shared_root, "unrelated-malformed-history").write_text("not json")
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage

    original = GroupCoverage.read_member
    reads = []

    def observed(self, sequence):
        reads.append(sequence)
        return original(self, sequence)

    with monkeypatch.context() as patch:
        guard_task_scans(patch, cfg)
        patch.setattr(GroupCoverage, "read_member", observed)
        operation_id = remove(cfg)
        assert len(reads) <= 1
        for _ in range(30):
            if control(cfg, operation_id)["state"] == "completed":
                break
            reads.clear()
            reconcile_group_cancel_operations(cfg, include_legacy=False)
            assert len(reads) <= 1
        assert control(cfg, operation_id)["state"] == "completed"


@pytest.mark.parametrize("sharing", ["private", "home", "shared_allowed", "shared_excluded"])
def test_queued_placement_blockers_follow_fallback_contract(tmp_path, sharing):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    change_worker(cfg, "experiment", "g3", "add")
    task = submit(cfg, ["true"], group="experiment")
    if sharing == "home":
        share(cfg, task.task_id, after_seconds=3600, helper_machines=["g2"])
    elif sharing.startswith("shared"):
        share(cfg, task.task_id, helper_machines=["g2" if sharing == "shared_allowed" else "g3"])
    change_worker(cfg, "experiment", "g3", "drain")
    discover(cfg)
    operation_id = remove(cfg)
    if sharing == "shared_allowed":
        assert finish(cfg, operation_id)["blockers"] == []
        assert load_task(cfg, task.task_id).state["projection"] == "queued"
    else:
        current = advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
        assert current["state"] == "waiting_ack"
        assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_later_committed_members_extend_removal_watermark(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    first = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: first.task_id in value["blockers"])
    # Submission correctly rejects a draining home. New members can still join
    # through another active worker and must enter the final membership proof.
    peer = replace(cfg, machine_name="g2", runtime_root=cfg.runtime_root.parent / "peer")
    later = submit(peer, ["true"], group="experiment")
    cancel_task(cfg, first.task_id)
    for _ in range(5):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    discover(cfg)
    completed = finish(cfg, operation_id)
    assert completed["discovery"]["member_cursor"] == 2
    assert load_task(cfg, later.task_id).state["projection"] == "queued"


def test_retry_after_terminal_member_is_visited_blocks_removal(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    first = submit(cfg, ["true"], group="experiment")
    blocker = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, first.task_id, [0])
    assert authorize_launch(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token, "test")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: value["discovery"]["member_cursor"] >= 1)
    retry(cfg, first.task_id)
    cancel_task(cfg, blocker.task_id)
    advance_until(cfg, operation_id, lambda value: first.task_id in value["blockers"])
    assert control(cfg, operation_id)["state"] != "completed"
    cancel_task(cfg, first.task_id)
    finish(cfg, operation_id)


def test_offer_clears_blocker_and_keep_local_reinstates_it(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    first = submit(cfg, ["true"], group="experiment")
    blocker = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: first.task_id in value["blockers"])
    share(cfg, first.task_id, helper_machines=["g2"])
    advance_until(cfg, operation_id, lambda value: first.task_id not in value["blockers"])
    keep_local(cfg, first.task_id)
    cancel_task(cfg, blocker.task_id)
    advance_until(cfg, operation_id, lambda value: first.task_id in value["blockers"])
    share(cfg, first.task_id, helper_machines=["g2"])
    finish(cfg, operation_id)


def test_worker_policy_change_rechecks_only_sensitive_members(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    terminal = submit(cfg, ["true"], group="experiment")
    cancel_task(cfg, terminal.task_id)
    shared = submit(cfg, ["true"], group="experiment")
    blocker = submit(cfg, ["true"], group="experiment")
    share(cfg, shared.task_id, helper_machines=["g2"])
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: value["discovery"]["member_cursor"] == 3)
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage

    original = GroupCoverage.read_member

    def no_terminal_revisit(self, sequence):
        assert sequence != terminal.group_membership_sequence, "policy change restarted historical census"
        return original(self, sequence)

    change_worker(cfg, "experiment", "g2", "drain")
    cancel_task(cfg, blocker.task_id)
    with monkeypatch.context() as patch:
        patch.setattr(GroupCoverage, "read_member", no_terminal_revisit)
        advance_until(cfg, operation_id, lambda value: shared.task_id in value["blockers"])
        assert control(cfg, operation_id)["state"] != "completed"
        change_worker(cfg, "experiment", "g2", "resume")
        finish(cfg, operation_id)


def test_termination_escalation_waits_for_authoritative_acknowledgement(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    assert not load_task(cfg, task.task_id).control.get("terminate_running")
    assert remove(cfg, terminate_running=True) == operation_id
    advance_until(cfg, operation_id, lambda _: load_task(cfg, task.task_id).control.get("terminate_running"))
    for _ in range(5):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "termination_ack")
    finish(cfg, operation_id)


@pytest.mark.parametrize("is_cleaned", [False, True])
def test_missing_member_requires_matching_completed_cleanup(tmp_path, is_cleaned):
    from qqtools.plugins.qexp.commands.cleanup import clean

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    cancel_task(cfg, task.task_id)
    discover(cfg)
    if is_cleaned:
        clean(cfg, task_id=task.task_id)
    else:
        task_path(cfg.shared_root, task.task_id).unlink()
    operation_id = remove(cfg)
    if is_cleaned:
        finish(cfg, operation_id)
    else:
        for _ in range(5):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
        assert control(cfg, operation_id)["state"] == "blocked"
        assert control(cfg, operation_id)["blocked_reason"]


def test_generation_invalidation_rebuilds_census_before_completion(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, load_task(cfg, task.task_id)):
        journal.invalidate()
    cancel_task(cfg, task.task_id)
    completed = finish(cfg, operation_id)
    assert completed["discovery"]["generation"] == journal.snapshot().generation
    assert completed["discovery"]["member_cursor"] == 1


def test_cli_activates_background_only_after_durable_remove(tmp_path, monkeypatch, capsys):
    from qqtools.plugins.qexp import cli
    from qqtools.plugins.qexp.agent.context import MachineRuntime

    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], group="experiment")
    machine_root = tmp_path / "machine-runtime"
    MachineRuntime(machine_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    activated = []

    def activate(config, *, reason, **kwargs):
        snapshot = read_json(group_path(config.shared_root, "experiment"))["worker_control"]
        assert control(config, snapshot["operation_id"])["discovery"]
        activated.append(reason)

    def no_history(*args, **kwargs):
        pytest.fail("removal rendered unused Task history")

    with monkeypatch.context() as patch:
        patch.setattr(cli, "ensure_local_agent_active", activate)
        patch.setattr(cli.observer, "list_tasks", no_history)
        assert (
            cli.main(
                [
                    "--project",
                    str(cfg.shared_root),
                    "--machine",
                    cfg.machine_name,
                    "--runtime-root",
                    str(cfg.runtime_root),
                    "--machine-runtime-root",
                    str(machine_root),
                    "group",
                    "worker",
                    "remove",
                    "experiment",
                    "g1",
                    "--format=human",
                ]
            )
            == 0
        )
    assert activated == ["group-worker-remove"]
    assert "converging" in capsys.readouterr().out


def test_existing_bound_removal_adopts_fresh_census_instead_of_old_empty_blockers(tmp_path):
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    operation_id = remove(cfg)
    path = locate_operation_path(cfg, "group_control", operation_id)
    data = read_json(path)
    data["group_control"].pop("discovery")
    data["group_control"]["blockers"] = []
    atomic_replace(path, data)
    reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    discover(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])


def test_partial_terminal_transition_is_settled_before_removal_completes(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import lifecycle

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])

    def crash(*args, **kwargs):
        raise OSError("terminal Attempt persisted before Task")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "save_task", crash)
        with pytest.raises(OSError, match="before Task"):
            fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    finish(cfg, operation_id)
    assert load_task(cfg, task.task_id).state["projection"] == "failed"


def test_termination_effect_before_operation_progress_is_recovered(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import worker_removal

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    discover(cfg)
    original = worker_removal.write_active_operation

    def uncertain(config, kind, operation_id, value):
        if load_task(config, task.task_id).control.get("terminate_running"):
            raise OSError("effect before removal progress")
        return original(config, kind, operation_id, value)

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "write_active_operation", uncertain)
        with pytest.raises(OSError, match="before removal progress"):
            operation_id = remove(cfg, terminate_running=True)
            for _ in range(20):
                reconcile_group_cancel_operations(cfg, include_legacy=False)
    group = read_json(group_path(cfg.shared_root, "experiment"))
    operation_id = group["group"]["worker_set"]["g1"]["removal_operation_id"]
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    assert control(cfg, operation_id)["state"] != "completed"
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "ack")
    finish(cfg, operation_id)


def test_degraded_coverage_cannot_complete_even_with_sufficient_prefix(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    cancel_task(cfg, task.task_id)
    path = GroupCoverage(cfg.shared_root, "experiment").directory / "state.json"
    data = read_json(path)
    data["blocked_reason"] = "membership_conflict"
    atomic_replace(path, data)
    for _ in range(8):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    assert read_json(group_path(cfg.shared_root, "experiment"))["group"]["worker_set"]["g1"]["state"] == "draining"


def test_group_snapshot_keeps_discovery_receipts_in_operation_only(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    current = advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    assert current["discovery"]["sensitive"]
    group = read_json(group_path(cfg.shared_root, "experiment"))
    assert "sensitive" not in group["worker_control"]["discovery"]
    assert group["worker_control"]["blockers"] == [task.task_id]


def test_cancelled_unmaterialized_claim_does_not_block_removal(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import scheduler

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    original = scheduler.save_task

    def uncertain(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("claim Task saved before Attempt")

    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "save_task", uncertain)
        with pytest.raises(OSError, match="before Attempt"):
            claim_task(cfg, task.task_id, [0])
    cancel_task(cfg, task.task_id)
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
    discover(cfg)
    finish(cfg, remove(cfg))


def test_sensitive_round_robin_survives_sorted_json_persistence(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage

    cfg = isolated_group(tmp_path, tail=0)
    tasks = [submit(cfg, ["true"], group="experiment") for _ in range(3)]
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: value["discovery"]["member_cursor"] == 3)
    reads = []
    original = GroupCoverage.read_member

    def observed(self, sequence):
        reads.append(sequence)
        return original(self, sequence)

    with monkeypatch.context() as patch:
        patch.setattr(GroupCoverage, "read_member", observed)
        for _ in range(12):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert set(reads) == {task.group_membership_sequence for task in tasks}
    assert control(cfg, operation_id)["state"] == "waiting_ack"


def test_archive_retry_does_not_reopen_committed_removal_for_later_task_retry(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import worker_removal

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test")
    discover(cfg)

    def crash(*args, **kwargs):
        raise OSError("removing committed before archive")

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "archive_operation", crash)
        with pytest.raises(OSError, match="before archive"):
            operation_id = remove(cfg)
            finish(cfg, operation_id)
    group = read_json(group_path(cfg.shared_root, "experiment"))
    worker = group["group"]["worker_set"]["g1"]
    assert worker["state"] == "removing"
    operation_id = worker["removal_operation_id"]
    retry(cfg, task.task_id)
    before = load_task(cfg, task.task_id).to_dict()
    assert finish(cfg, operation_id)["state"] == "completed"
    assert load_task(cfg, task.task_id).to_dict() == before
    next_id = remove(cfg)
    assert next_id != operation_id
    advance_until(cfg, next_id, lambda value: task.task_id in value["blockers"])


def test_shared_fallback_static_quota_must_fit_task(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add", gpu_limit_gpus=1, has_gpu_limit=True)
    task = submit(cfg, ["true"], requested_gpus=2, group="experiment")
    share(cfg, task.task_id, helper_machines=["g2"])
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
    change_worker(cfg, "experiment", "g2", "set", gpu_limit_gpus=2, has_gpu_limit=True)
    finish(cfg, operation_id)
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_repeat_remove_returns_addressed_operation_without_replacing_latest_snapshot(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    first = remove(cfg)
    advance_until(cfg, first, lambda value: task.task_id in value["blockers"])
    second = change_worker(cfg, "experiment", "g2", "remove")["worker_control"]["operation_id"]
    finish(cfg, second)
    repeated = change_worker(cfg, "experiment", "g1", "remove")["worker_control"]
    assert repeated["operation_id"] == first
    assert repeated["state"] == "waiting_ack"
    assert read_json(group_path(cfg.shared_root, "experiment"))["worker_control"]["operation_id"] == second


def test_progress_only_removal_steps_do_not_refresh_group_ready_projection(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import worker_removal

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = remove(cfg)
    advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])

    def forbidden(*args, **kwargs):
        raise AssertionError("status update refreshed the Group ready projection")

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "primary_projection_routes_for_group", forbidden)
        patch.setattr(worker_removal, "sync_primary_ready_group", forbidden)
        for _ in range(5):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] == "waiting_ack"


@pytest.mark.parametrize("cut", ["draining", "removing"])
def test_interrupted_projection_is_repaired_before_operation_retires(tmp_path, monkeypatch, cut):
    from qqtools.plugins.qexp.commands import worker_removal

    cfg = isolated_group(tmp_path, tail=0)
    discover(cfg)
    original = worker_removal.sync_primary_ready_group
    repaired = []

    def fail_projection(cfg, group, *, previous_workers):
        state = read_json(group_path(cfg.shared_root, group))["group"]["worker_set"]["g1"]["state"]
        if state == cut:
            raise OSError("projection interrupted")
        return original(cfg, group, previous_workers=previous_workers)

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "sync_primary_ready_group", fail_projection)
        with pytest.raises(OSError, match="projection interrupted"):
            finish(cfg, remove(cfg))
    group = read_json(group_path(cfg.shared_root, "experiment"))
    operation_id = group["group"]["worker_set"]["g1"]["removal_operation_id"]
    assert control(cfg, operation_id)["state"] != "completed"
    # A second failure must still leave recovery discoverable, even when the
    # authoritative Worker already says removing.
    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "sync_primary_ready_group", fail_projection)
        with pytest.raises(OSError, match="projection interrupted"):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"

    def repair(cfg, group, *, previous_workers):
        repaired.append(previous_workers["g1"]["state"])
        return original(cfg, group, previous_workers=previous_workers)

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "sync_primary_ready_group", repair)
        finish(cfg, operation_id)
    assert repaired[0] == "active"


@pytest.mark.parametrize("cancel_before_replay", [False, True])
def test_real_ready_route_recovers_after_interrupted_drain(tmp_path, monkeypatch, cancel_before_replay):
    from qqtools.plugins.qexp.commands import worker_removal
    from qqtools.plugins.qexp.runtime.ready.routes import allocator_path

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    allocator = allocator_path(cfg.shared_root, "home.g1")
    assert read_json(allocator)["ready_allocator"]["primary_state"] == "active"

    def interrupt(*args, **kwargs):
        raise OSError("real route projection interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(worker_removal, "sync_primary_ready_group", interrupt)
        with pytest.raises(OSError, match="real route"):
            remove(cfg)
    group = read_json(group_path(cfg.shared_root, "experiment"))
    operation_id = group["group"]["worker_set"]["g1"]["removal_operation_id"]
    assert control(cfg, operation_id)["state"] == "preparing"
    assert read_json(allocator)["ready_allocator"]["primary_state"] == "degraded"
    if cancel_before_replay:
        cancel_task(cfg, task.task_id)
        # Removing the last ready member repairs its original route even though
        # subsequent Group projection discovery no longer includes that route.
        assert read_json(allocator)["ready_allocator"]["primary_state"] == "active"
        finish(cfg, operation_id)
    else:
        advance_until(cfg, operation_id, lambda value: task.task_id in value["blockers"])
        assert read_json(allocator)["ready_allocator"]["primary_state"] == "active"
        assert control(cfg, operation_id)["state"] == "waiting_ack"
        cancel_task(cfg, task.task_id)
        finish(cfg, operation_id)
    assert read_json(allocator)["ready_allocator"]["primary_state"] == "active"
