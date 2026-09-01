from __future__ import annotations

import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import group as group_commands
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import share
from qqtools.plugins.qexp.machine_agent import dispatch_machine_cycle_locked
from qqtools.plugins.qexp import machine_agent
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths
from qqtools.plugins.qexp.runtime.ready import (
    ReadyCursor,
    ReadyPeek,
    delete_stale_ready_marker,
    load_ready_cursor,
    next_ready_marker,
    peek_primary_ready_marker,
    rebuild_primary_ready_index,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.ready import bump_primary_ready_revision, ready_index_route_revision
from qqtools.plugins.qexp.runtime.reservations import attach, reserve
from qqtools.plugins.qexp.runtime.work_budget import (
    AdaptiveBatchSizer,
    SliceBudget,
    WorkBudgetPolicy,
)
from qqtools.plugins.qexp.scheduler import (
    BorrowAdmissionRequired,
    _BorrowAdmissionGrant,
    _BorrowAdmissionRevision,
    claim_task,
    run_dispatch_cycle,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

class _RecordingExecutor:
    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch_attempt(self, _cfg, task_id, _attempt) -> None:
        self.launched.append(task_id)


def _borrow_project(tmp_path: Path, work: Path):
    cfg = init_shared_root(
        tmp_path / "borrow" / ".qexp", "gpu-1", runtime_root=tmp_path / "borrow-rt"
    )
    create_group(cfg, "borrow-group")
    change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
    task = submit(
        cfg,
        ["echo", "borrow"],
        group="borrow-group",
        sharing_mode="spillover",
        working_dir=work,
    )
    share(cfg, task.task_id)
    return cfg, task


def test_machine_agent_admits_borrow_only_after_no_primary_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, task = _borrow_project(tmp_path, work)
    _activate_ready(cfg)

    def make_budget(policy: WorkBudgetPolicy | None = None) -> SliceBudget:
        # Keep the production limits while making this admission test independent of I/O latency.
        selected_policy = policy if policy is not None else WorkBudgetPolicy()
        return SliceBudget(selected_policy, clock_ns=lambda: 0)

    monkeypatch.setattr(machine_agent, "SliceBudget", make_budget)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [{
        "project_id": binding.project_id,
        "launched": [task.task_id],
        "status": "dispatched",
    }]
    assert executor.launched == [task.task_id]


def test_direct_borrow_claim_requires_machine_agent_admission(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, task = _borrow_project(tmp_path, work)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)

    with pytest.raises(BorrowAdmissionRequired, match="machine-agent admission grant"):
        claim_task(
            cfg,
            task.task_id,
            [0],
            reservation_runtime_root=runtime.root,
            project_id=binding.project_id,
        )


def test_regular_dispatch_reports_borrow_admission_requirement(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, _task = _borrow_project(tmp_path, work)

    with pytest.raises(BorrowAdmissionRequired, match="machine-agent admission grant"):
        run_dispatch_cycle(
            cfg, available_gpus=[0], should_recover_starting=False
        )


def test_primary_projection_cursor_uses_lexicographic_catalog_order(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    # Insert in the reverse lexical order to ensure directory enumeration
    # cannot define cursor progression.
    submit(cfg, ["echo", "z"], task_id="z-task")
    submit(cfg, ["echo", "a"], task_id="a-task")
    _activate_ready(cfg)
    budget = SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000))

    first = peek_primary_ready_marker(cfg, "probe", "home", None, budget)
    second = peek_primary_ready_marker(cfg, "probe", "home", first.cursor, budget)

    assert first.reference is not None and first.reference.task_id == "a-task"
    assert second.reference is not None and second.reference.task_id == "z-task"


def test_group_role_sync_closes_every_affected_primary_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-a", runtime_root=tmp_path / "runtime"
    )
    create_group(cfg, "workers", workers=["gpu-a", "gpu-b"])
    task = submit(cfg, ["echo", "primary"], group="workers", task_id="primary-task")
    _activate_ready(cfg)

    original_sync = group_commands.sync_primary_ready_group

    def assert_routes_are_closed(*args, **kwargs):
        with pytest.raises(ValueError, match="ready allocator is unreadable"):
            ready_index_route_revision(cfg, "home", primary_only=True)
        with pytest.raises(ValueError, match="ready allocator is unreadable"):
            ready_index_route_revision(cfg, "shared", primary_only=True)
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(group_commands, "sync_primary_ready_group", assert_routes_are_closed)
    change_worker(cfg, "workers", "gpu-b", "set", role="borrow")


def test_borrow_admission_grant_is_invalidated_by_primary_revision(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, task = _borrow_project(tmp_path, work)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revisions = tuple(
        _BorrowAdmissionRevision(
            binding.project_id,
            cfg,
            scope,
            ready_index_route_revision(cfg, scope),
        )
        for scope in ("shared", "home")
    )
    grant = _BorrowAdmissionGrant(runtime.root, revisions)
    bump_primary_ready_revision(cfg, "shared", cfg.machine_name)

    with pytest.raises(BorrowAdmissionRequired, match="current machine-agent admission grant"):
        claim_task(
            cfg,
            task.task_id,
            [0],
            reservation_runtime_root=runtime.root,
            project_id=binding.project_id,
            borrow_admission_grant=grant,
        )


def test_borrow_marker_changes_do_not_advance_primary_revision(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, _task = _borrow_project(tmp_path, work)
    before = ready_index_route_revision(cfg, "shared", primary_only=True)
    extra = submit(cfg, ["echo", "borrow-2"], group="borrow-group", sharing_mode="spillover", working_dir=work)
    share(cfg, extra.task_id)

    assert ready_index_route_revision(cfg, "shared", primary_only=True) == before


def test_primary_probe_uses_primary_projection_without_scanning_borrow_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, _task = _borrow_project(tmp_path, work)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)

    def fail_if_marker_is_classified(*_args, **_kwargs):
        raise AssertionError("borrow markers must not be classified by primary probe")

    monkeypatch.setattr(machine_agent, "classify_ready_marker", fail_if_marker_is_classified)
    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )

    assert probe.state == "no_primary_demand"


def test_primary_probe_accepts_an_unestablished_empty_shared_route(
    tmp_path: Path,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)

    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )

    assert probe.state == "no_primary_demand"


def test_missing_primary_projection_fails_closed(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, _task = _borrow_project(tmp_path, work)
    state = read_json(ready_state_path(cfg.shared_root))
    state["ready_index"]["state"] = "active"
    state["ready_index"]["writer_capability"] = "ready-v1"
    atomic_replace(ready_state_path(cfg.shared_root), state)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)

    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )

    assert probe.state == "unresolved"


def test_primary_probe_skips_demand_that_exceeds_remaining_group_gpu_limit(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "workers")
    change_worker(cfg, "workers", "gpu-1", "set", gpu_limit_gpus=1, has_gpu_limit=True)
    task = submit(
        cfg,
        ["echo", "primary"],
        group="workers",
        task_id="primary-task",
        sharing_mode="spillover",
        working_dir=work,
    )
    share(cfg, task.task_id)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    occupied = reserve(
        runtime.root,
        "occupied",
        [0],
        project_id=binding.project_id,
        group_name="workers",
        machine_name="gpu-1",
    )

    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0, 1],
        [1],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
        (occupied["reservation"],),
    )

    assert probe.state == "no_primary_demand", probe.diagnostics


@pytest.mark.parametrize("damage", ["catalog", "partition"])
def test_established_primary_route_damage_fails_closed(tmp_path: Path, damage: str) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    submit(cfg, ["echo", "primary"], task_id="primary-task")
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    catalog_path = shared_paths(cfg.shared_root)["ready_catalogs"] / "home.gpu-1" / (
        "0000000000000000.json"
    )
    if damage == "catalog":
        catalog_path.unlink()
    else:
        partition = read_json(catalog_path)["ready_catalog"]["partitions"][0]
        (shared_paths(cfg.shared_root)["ready_home"] / "gpu-1" / partition / "partition.json").unlink()

    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )

    assert probe.state == "unresolved"
    assert machine_agent._build_borrow_admission_grant(
        runtime, {binding.project_id: cfg}, probe, {binding.project_id}
    ) is None


def test_unprobeable_enabled_project_blocks_borrow_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    blocked_cfg, _blocked_task = _borrow_project(tmp_path / "blocked", work)
    allowed_cfg = init_shared_root(
        tmp_path / "allowed" / ".qexp", "gpu-1", runtime_root=tmp_path / "allowed-rt"
    )
    create_group(allowed_cfg, "borrow-group")
    change_worker(allowed_cfg, "borrow-group", "gpu-1", "set", role="borrow")
    allowed_task = submit(
        allowed_cfg, ["echo", "allowed"], group="borrow-group", sharing_mode="spillover",
        working_dir=work,
    )
    share(allowed_cfg, allowed_task.task_id)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(blocked_cfg.shared_root, blocked_cfg.machine_name)
    allowed_binding = runtime.add_binding(allowed_cfg.shared_root, allowed_cfg.machine_name)
    original_maintenance = __import__(
        "qqtools.plugins.qexp.machine_agent", fromlist=["maintain_project"]
    ).maintain_project

    def fail_blocked(cfg, **kwargs):
        if cfg.shared_root == blocked_cfg.shared_root:
            raise RuntimeError("blocked project maintenance")
        return original_maintenance(cfg, **kwargs)

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.maintain_project", fail_blocked)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime, available_gpus=[0], executor=executor, supervise=False, publish_snapshots=False
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert result_by_project[allowed_binding.project_id]["launched"] == []
    assert executor.launched == []


def test_machine_agent_primary_demand_blocks_new_borrow_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    primary_cfg = init_shared_root(
        tmp_path / "primary" / ".qexp", "gpu-1", runtime_root=tmp_path / "primary-rt"
    )
    create_group(primary_cfg, "primary-group")
    primary_task = submit(
        primary_cfg, ["echo", "primary"], group="primary-group", working_dir=work
    )
    borrow_cfg, borrow_task = _borrow_project(tmp_path, work)
    _activate_ready(primary_cfg)
    _activate_ready(borrow_cfg)

    def make_budget(policy: WorkBudgetPolicy | None = None) -> SliceBudget:
        # Keep the production limits while making this admission test independent of I/O latency.
        selected_policy = policy if policy is not None else WorkBudgetPolicy()
        return SliceBudget(selected_policy, clock_ns=lambda: 0)

    monkeypatch.setattr(machine_agent, "SliceBudget", make_budget)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    primary_binding = runtime.add_binding(primary_cfg.shared_root, primary_cfg.machine_name)
    borrow_binding = runtime.add_binding(borrow_cfg.shared_root, borrow_cfg.machine_name)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime, available_gpus=[0], executor=executor, supervise=False, publish_snapshots=False
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert executor.launched == [primary_task.task_id]
    assert result_by_project[primary_binding.project_id]["launched"] == [primary_task.task_id]
    assert result_by_project[borrow_binding.project_id]["launched"] == []
    assert borrow_task.task_id not in executor.launched


def test_primary_probe_rechecks_completed_shared_scope_before_borrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, borrow_task = _borrow_project(tmp_path, work)
    create_group(cfg, "primary-group")
    change_worker(cfg, "primary-group", "gpu-1", "set", role="primary")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    occupied = reserve(runtime.root, "occupied", [0], project_id="external")
    attach(runtime.root, occupied["reservation"]["reservation_id"], "occupied-attempt", 1)
    executor = _RecordingExecutor()

    original_peek = machine_agent.peek_primary_ready_marker
    should_force_shared_complete = True
    should_force_home_incomplete = True

    def control_probe_progress(cfg_arg, project_id, scope, cursor, budget):
        nonlocal should_force_home_incomplete, should_force_shared_complete
        if scope == "shared" and should_force_shared_complete:
            should_force_shared_complete = False
            return ReadyPeek(
                None,
                ReadyCursor(project_id, cfg_arg.machine_name, scope, 0, None, None, 0),
                wrapped=True,
            )
        peek = original_peek(cfg_arg, project_id, scope, cursor, budget)
        if scope == "home" and should_force_home_incomplete:
            should_force_home_incomplete = False
            return ReadyPeek(None, peek.cursor, exhausted=True)
        return peek

    monkeypatch.setattr(machine_agent, "peek_primary_ready_marker", control_probe_progress)

    first_results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )
    assert first_results == [{
        "project_id": binding.project_id,
        "launched": [],
        "status": "dispatched",
    }]

    primary_task = submit(
        cfg,
        ["echo", "primary"],
        group="primary-group",
        requested_gpus=2,
        working_dir=work,
    )
    share(cfg, primary_task.task_id)

    second_results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )
    assert second_results == [{
        "project_id": binding.project_id,
        "launched": [],
        "status": "dispatched",
    }]
    assert executor.launched == []
    assert borrow_task.task_id not in executor.launched
    shared_probe_key = (binding.project_id, "shared")
    assert runtime.primary_probe_complete[shared_probe_key] is False
    assert runtime.primary_probe_cursors[shared_probe_key] is None


def test_primary_aggregation_waiting_blocks_borrow_admission(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    primary_cfg = init_shared_root(
        tmp_path / "primary" / ".qexp", "gpu-1", runtime_root=tmp_path / "primary-rt"
    )
    create_group(primary_cfg, "primary-group")
    primary_task = submit(
        primary_cfg,
        ["echo", "primary"],
        group="primary-group",
        requested_gpus=2,
        working_dir=work,
    )
    borrow_cfg, borrow_task = _borrow_project(tmp_path, work)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    primary_binding = runtime.add_binding(primary_cfg.shared_root, primary_cfg.machine_name)
    borrow_binding = runtime.add_binding(borrow_cfg.shared_root, borrow_cfg.machine_name)
    occupied = reserve(runtime.root, "occupied", [0], project_id="external")
    attach(runtime.root, occupied["reservation"]["reservation_id"], "occupied-attempt", 1)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert primary_task.task_id not in executor.launched
    assert borrow_task.task_id not in executor.launched
    assert result_by_project[primary_binding.project_id]["launched"] == []
    assert result_by_project[borrow_binding.project_id]["launched"] == []


def test_ungrouped_primary_demand_blocks_borrow_admission(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    primary_cfg = init_shared_root(
        tmp_path / "primary" / ".qexp", "gpu-1", runtime_root=tmp_path / "primary-rt"
    )
    create_group(primary_cfg, "borrow-route")
    change_worker(primary_cfg, "borrow-route", "gpu-1", "set", role="borrow")
    shared_task = submit(
        primary_cfg,
        ["echo", "shared-route"],
        group="borrow-route",
        sharing_mode="spillover",
        working_dir=work,
    )
    share(primary_cfg, shared_task.task_id)
    primary_task = submit(
        primary_cfg,
        ["echo", "primary"],
        requested_gpus=2,
        working_dir=work,
    )
    borrow_cfg, borrow_task = _borrow_project(tmp_path, work)
    _activate_ready(primary_cfg)
    _activate_ready(borrow_cfg)

    runtime = MachineRuntime(tmp_path / "machine-runtime")
    primary_binding = runtime.add_binding(primary_cfg.shared_root, primary_cfg.machine_name)
    borrow_binding = runtime.add_binding(borrow_cfg.shared_root, borrow_cfg.machine_name)
    occupied = reserve(runtime.root, "occupied", [0], project_id="external")
    attach(runtime.root, occupied["reservation"]["reservation_id"], "occupied-attempt", 1)

    probe = machine_agent._probe_primary_demand(
        runtime,
        {primary_binding.project_id: primary_cfg, borrow_binding.project_id: borrow_cfg},
        {primary_binding.project_id: primary_cfg, borrow_binding.project_id: borrow_cfg},
        [0, 1],
        [1],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )
    assert probe.state == "waiting_for_aggregation"

    executor = _RecordingExecutor()
    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert executor.launched == []
    assert result_by_project[primary_binding.project_id]["launched"] == []
    assert result_by_project[borrow_binding.project_id]["launched"] == []
    assert primary_task.task_id not in executor.launched
    assert borrow_task.task_id not in executor.launched


def _activate_ready(cfg) -> None:
    schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["writer_capabilities"] = ["ready-v1"]
    atomic_replace(schema_path, schema)
    value = read_json(ready_state_path(cfg.shared_root))
    value["ready_index"]["state"] = "active"
    value["ready_index"]["writer_capability"] = "ready-v1"
    atomic_replace(ready_state_path(cfg.shared_root), value)
    rebuild_primary_ready_index(cfg)


def test_active_ready_dispatch_does_not_enumerate_task_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready", working_dir=work)
    _activate_ready(cfg)
    executor = _RecordingExecutor()
    original_iter_json = __import__(
        "qqtools.plugins.qexp.scheduler", fromlist=["iter_json"]
    ).iter_json

    def reject_task_enumeration(directory):
        assert Path(directory) != shared_paths(cfg.shared_root)["tasks"]
        return original_iter_json(directory)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.iter_json", reject_task_enumeration
    )
    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        executor=executor,
        should_recover_starting=False,
        work_budget=SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000)),
    )
    assert launched == [task.task_id]
    assert executor.launched == [task.task_id]


def test_ready_cursor_advances_past_temporarily_unavailable_candidate(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    create_group(cfg, "paused")
    paused = submit(
        cfg, ["echo", "paused"], task_id="task-a", group="paused", working_dir=work
    )
    ready = submit(cfg, ["echo", "ready"], task_id="task-b", working_dir=work)
    group_path = shared_paths(cfg.shared_root)["groups"] / "paused.json"
    group = read_json(group_path)
    group["group"]["dispatch_state"] = "paused"
    atomic_replace(group_path, group)
    _activate_ready(cfg)

    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        executor=_RecordingExecutor(),
        should_recover_starting=False,
        max_new_claims=1,
        work_budget=SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000)),
    )

    assert launched == [ready.task_id]
    cursor = load_ready_cursor(cfg, "standalone", "home")
    assert cursor.after_name is None
    assert cursor.revision >= 2
    assert paused.task_id not in launched


def test_candidate_cursor_wraps_without_repeating_a_marker_in_one_slice(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready")
    _activate_ready(cfg)

    first, first_wrapped = next_ready_marker(cfg, "project-one", "home")
    end_of_partition, second_wrapped = next_ready_marker(cfg, "project-one", "home")
    repeated, third_wrapped = next_ready_marker(cfg, "project-one", "home")

    assert first is not None and first.identity == f"{task.task_id}.{task.ready_generation}"
    assert first_wrapped is False
    assert end_of_partition is None
    assert second_wrapped is True
    assert repeated is not None and repeated.identity == first.identity
    assert third_wrapped is False


def test_machine_cycle_fills_multiple_gpus_across_fair_rounds(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    tasks = [
        submit(cfg, ["echo", str(index)], task_id=f"task-{index}", working_dir=work)
        for index in range(3)
    ]
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1, 2],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [{
        "project_id": binding.project_id,
        "launched": [task.task_id for task in tasks],
        "status": "dispatched",
    }]
    assert executor.launched == [task.task_id for task in tasks]


def test_degraded_ready_index_fails_closed_for_new_claims(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    submit(cfg, ["echo", "ready"], task_id="task-ready")
    state = read_json(ready_state_path(cfg.shared_root))
    state["ready_index"]["state"] = "degraded"
    atomic_replace(ready_state_path(cfg.shared_root), state)

    assert run_dispatch_cycle(
        cfg, available_gpus=[0], should_recover_starting=False
    ) == []


def test_claim_race_does_not_degrade_ready_index_or_block_other_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    raced = submit(cfg, ["echo", "raced"], task_id="task-a-raced")
    unrelated = submit(cfg, ["echo", "unrelated"], task_id="task-b-unrelated")
    _activate_ready(cfg)
    from qqtools.plugins.qexp import scheduler

    original_classify = scheduler.classify_ready_marker
    did_claim = False

    def claim_before_classify(config, reference):
        nonlocal did_claim
        if reference.task_id == raced.task_id and not did_claim:
            did_claim = True
            assert claim_task(config, raced.task_id, [1]) is not None
        return original_classify(config, reference)

    monkeypatch.setattr(scheduler, "classify_ready_marker", claim_before_classify)

    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        should_recover_starting=False,
        max_new_claims=1,
        work_budget=SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000)),
    )

    assert did_claim
    assert launched == [unrelated.task_id]
    assert read_json(ready_state_path(cfg.shared_root))["ready_index"]["state"] == "active"


def test_slow_ready_reads_shrink_the_process_local_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    submit(cfg, ["echo", "ready"], task_id="task-ready")
    _activate_ready(cfg)
    policy = WorkBudgetPolicy(soft_deadline_ms=50, initial_batch_size=4)
    budget = SliceBudget(policy)
    sizer = AdaptiveBatchSizer(policy)
    from qqtools.plugins.qexp import scheduler

    original_classify = scheduler.classify_ready_marker

    def slow_classify(*args, **kwargs):
        time.sleep(0.1)
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(scheduler, "classify_ready_marker", slow_classify)

    run_dispatch_cycle(
        cfg,
        available_gpus=[],
        should_recover_starting=False,
        work_budget=budget,
        batch_sizer=sizer,
    )

    assert sizer.batch_size < policy.initial_batch_size


def test_ready_dispatch_stops_when_record_operation_cost_exceeds_budget(
    tmp_path: Path,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready")
    _activate_ready(cfg)
    budget = SliceBudget(
        WorkBudgetPolicy(
            operation_hard_limit=2,
            soft_deadline_ms=60_000,
        )
    )

    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        should_recover_starting=False,
        work_budget=budget,
    )

    assert launched == []
    assert budget.records_used == 0
    assert budget.operations_used == 0
    assert read_json(shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json")[
        "task"
    ]["state"]["projection"] == "queued"


def test_full_capacity_machine_cycle_reads_no_ready_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    from qqtools.plugins.qexp.runtime.reservations import attach, reserve

    reservation = reserve(
        runtime.root,
        "occupied",
        [0],
        attempt_id="occupied-attempt",
        fencing_token=1,
        project_id="external-project",
    )
    attach(runtime.root, reservation["reservation"]["reservation_id"], "occupied-attempt", 1)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.next_ready_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ready candidates must not be read at full capacity")
        ),
    )

    assert dispatch_machine_cycle_locked(
        runtime, available_gpus=[0], supervise=False, publish_snapshots=False
    ) == [{"project_id": binding.project_id, "launched": [], "status": "dispatched"}]


def test_stale_generation_cleanup_cannot_delete_current_marker(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready")
    stale_reference, _ = next_ready_marker(cfg, "project-one", "home")
    assert stale_reference is not None
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    value = read_json(task_path)
    value["task"]["ready_generation"] = task.ready_generation + 1
    atomic_replace(task_path, value)

    assert delete_stale_ready_marker(cfg, stale_reference) is True
    assert not (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    ).exists()
    assert read_json(task_path)["task"]["ready_generation"] == task.ready_generation + 1


def test_project_round_robin_fairness_survives_dynamic_binding(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    projects = []
    for name in ("first", "second"):
        cfg = init_shared_root(
            tmp_path / name / ".qexp",
            "gpu-1",
            runtime_root=tmp_path / f"{name}-runtime",
        )
        task = submit(
            cfg, ["echo", name], task_id=f"task-{name}", working_dir=work
        )
        _activate_ready(cfg)
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        projects.append((cfg, task, binding))
    runtime.save_cursor(projects[1][2].project_id)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert executor.launched == [projects[1][1].task_id, projects[0][1].task_id]
    assert all(
        result_by_project[binding.project_id]["launched"] == [task.task_id]
        for _cfg, task, binding in projects
    )
    assert runtime.load_cursor() == projects[1][2].project_id
