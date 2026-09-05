from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import group as group_commands
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import cancel, edit_dependencies, share
from qqtools.plugins.qexp.machine_agent import dispatch_machine_cycle_locked
from qqtools.plugins.qexp import machine_agent
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths
from qqtools.plugins.qexp.runtime.ready import (
    ReadyClassificationResult,
    ReadyCursor,
    ReadyPeek,
    delete_stale_ready_marker,
    load_ready_cursor,
    next_ready_marker,
    peek_primary_ready_marker,
    rebuild_primary_ready_index,
    write_ready_marker,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.ready import bump_primary_ready_revision, ready_index_route_revision
from qqtools.plugins.qexp.runtime.reservations import attach, reserve
from qqtools.plugins.qexp.runtime.cpu_lane import reserve_cpu, set_cpu_lane_capacity
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


def test_temporary_primary_dependency_gate_is_rechecked_before_borrowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "primary"], task_id="primary", working_dir=work)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    temporary = ReadyClassificationResult("temporarily_unavailable", "dependency_waiting")

    monkeypatch.setattr(machine_agent, "classify_ready_marker", lambda *_args: temporary)
    first = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )
    # Dependency waiting needs revisiting, but is not primary resource demand
    # and therefore must not block a borrow admission.
    assert first.state == "no_primary_demand"
    assert machine_agent._build_borrow_admission_grant(
        runtime, {binding.project_id: cfg}, first, {binding.project_id}
    ) is not None

    claimable = ReadyClassificationResult("claimable", "eligible_truth", task)
    monkeypatch.setattr(machine_agent, "classify_ready_marker", lambda *_args: claimable)
    second = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )
    assert second.state == "runnable_now"


def test_primary_probe_retains_dependency_recheck_across_budgeted_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    first_task = submit(cfg, ["echo", "first"], task_id="first", working_dir=work)
    submit(cfg, ["echo", "second"], task_id="second", working_dir=work)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    temporary = ReadyClassificationResult("temporarily_unavailable", "dependency_waiting")
    route_key = (binding.project_id, "home", "gpu")

    monkeypatch.setattr(machine_agent, "classify_ready_marker", lambda *_args: temporary)
    limited = SliceBudget(
        WorkBudgetPolicy(
            record_hard_limit=1, operation_hard_limit=32, initial_batch_size=1
        ),
        clock_ns=lambda: 0,
    )
    incomplete = machine_agent._probe_primary_demand(
        runtime, {binding.project_id: cfg}, {binding.project_id: cfg}, [0], [0], limited
    )
    assert incomplete.state == "unresolved"
    assert route_key in runtime.primary_probe_recheck_cursors

    completed = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )
    assert completed.state == "no_primary_demand"
    assert runtime.primary_probe_complete[route_key]

    claimable = ReadyClassificationResult("claimable", "eligible_truth", first_task)
    monkeypatch.setattr(machine_agent, "classify_ready_marker", lambda *_args: claimable)
    rechecked = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        [0],
        [0],
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
    )
    assert rechecked.state == "runnable_now"


@pytest.mark.parametrize("is_dependency_first", [True, False])
def test_dependency_rechecks_do_not_starve_later_project_baseline_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_dependency_first: bool,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    dependency_cfg = init_shared_root(
        tmp_path / "dependencies" / ".qexp", "gpu-1", runtime_root=tmp_path / "dependencies-rt"
    )
    create_group(dependency_cfg, "dependencies")
    for index in range(3):
        parent = submit(
            dependency_cfg,
            ["echo", "parent"],
            task_id=f"parent-{index}",
            group="dependencies",
            working_dir=work,
        )
        submit(
            dependency_cfg,
            ["echo", "child"],
            task_id=f"child-{index}",
            group="dependencies",
            working_dir=work,
            depends_on_task_ids=[parent.task_id],
        )
        cancel(dependency_cfg, parent.task_id)
    borrow_cfg, _ = _borrow_project(tmp_path, work)
    borrow_parent = submit(
        borrow_cfg,
        ["echo", "borrow-parent"],
        task_id="borrow-parent",
        group="borrow-group",
        working_dir=work,
    )
    submit(
        borrow_cfg,
        ["echo", "borrow-child"],
        task_id="borrow-child",
        group="borrow-group",
        working_dir=work,
        depends_on_task_ids=[borrow_parent.task_id],
    )
    cancel(borrow_cfg, borrow_parent.task_id)
    _activate_ready(dependency_cfg)
    _activate_ready(borrow_cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    dependency_binding = runtime.add_binding(
        dependency_cfg.shared_root, dependency_cfg.machine_name
    )
    borrow_binding = runtime.add_binding(borrow_cfg.shared_root, borrow_cfg.machine_name)
    # Fix the registry ordering; path-derived project IDs otherwise randomize
    # which project's completed routes consume the next slice's budget first.
    generation, _ = runtime.load_registry()
    dependency_binding = replace(
        dependency_binding, project_id="a" if is_dependency_first else "b"
    )
    borrow_binding = replace(
        borrow_binding, project_id="b" if is_dependency_first else "a"
    )
    monkeypatch.setattr(
        runtime, "load_registry", lambda: (generation, [dependency_binding, borrow_binding])
    )
    readable = {
        dependency_binding.project_id: dependency_cfg,
        borrow_binding.project_id: borrow_cfg,
    }
    budget_policy = WorkBudgetPolicy(
        record_hard_limit=3, operation_hard_limit=12, initial_batch_size=1
    )

    for _round in range(3):
        probe = machine_agent.PrimaryDemandProbe("unresolved")
        for _slice in range(18):
            probe = machine_agent._probe_primary_demand(
                runtime,
                readable,
                readable,
                [0],
                [0],
                SliceBudget(budget_policy, clock_ns=lambda: 0),
            )
            if probe.state == "no_primary_demand":
                break

        assert probe.state == "no_primary_demand"
        grant = machine_agent._build_borrow_admission_grant(
            runtime, readable, probe, set(readable)
        )
        assert grant is not None and grant.is_valid(runtime.root)


def test_dependency_recheck_advances_past_a_still_blocked_first_candidate(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    create_group(cfg, "dependencies")
    children = []
    for index in range(3):
        parent = submit(
            cfg,
            ["echo", "parent"],
            task_id=f"parent-{index}",
            group="dependencies",
            working_dir=work,
        )
        children.append(
            submit(
                cfg,
                ["echo", "child"],
                task_id=f"child-{index}",
                group="dependencies",
                working_dir=work,
                depends_on_task_ids=[parent.task_id],
            )
        )
        cancel(cfg, parent.task_id)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    readable = {binding.project_id: cfg}
    budget_policy = WorkBudgetPolicy(
        record_hard_limit=16, operation_hard_limit=64, initial_batch_size=1
    )

    completed = machine_agent._probe_primary_demand(
        runtime,
        readable,
        readable,
        [0],
        [0],
        SliceBudget(budget_policy, clock_ns=lambda: 0),
    )
    assert completed.state == "no_primary_demand"

    edit_dependencies(cfg, children[-1].task_id, [])
    probe = machine_agent.PrimaryDemandProbe("no_primary_demand")
    for _ in range(8):
        probe = machine_agent._probe_primary_demand(
            runtime,
            readable,
            readable,
            [0],
            [0],
            SliceBudget(budget_policy, clock_ns=lambda: 0),
        )
        if probe.state == "runnable_now":
            break

    assert probe.state == "runnable_now"


@pytest.mark.parametrize("ready_index", [0, 2])
@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_dependency_recheck_retains_aggregation_demand(
    tmp_path: Path, ready_index: int, lane: str,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    create_group(cfg, "dependencies")
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="dependencies", working_dir=tmp_path
    )
    for index in range(3):
        submit(
            cfg, ["echo", "child"], task_id=f"child-{index}", group="dependencies",
            working_dir=tmp_path, depends_on_task_ids=[parent.task_id],
            requested_gpus=2 if lane == "gpu" else 0,
            requested_cpus=2 if lane == "cpu" else None,
        )
    cancel(cfg, parent.task_id)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    readable = {binding.project_id: cfg}

    def probe(free: list[int]) -> machine_agent.PrimaryDemandProbe:
        return machine_agent._probe_primary_demand(
            runtime, readable, readable, [0, 1], free,
            SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0), lane=lane,
        )

    assert probe([0]).state == "no_primary_demand"
    edit_dependencies(cfg, f"child-{ready_index}", [])
    for _ in range(8):
        result = probe([0])
        if result.state == "waiting_for_aggregation":
            break
    assert result.state == "waiting_for_aggregation"
    for _ in range(8):
        result = probe([0])
        assert result.state == "waiting_for_aggregation"
        assert machine_agent._build_borrow_admission_grant(
            runtime, readable, result, set(readable), lane=lane,
        ) is None
    assert probe([0, 1]).state == "runnable_now"
    cancel(cfg, f"child-{ready_index}")
    for _ in range(8):
        result = probe([0])
        if result.state == "no_primary_demand":
            break
    assert result.state == "no_primary_demand"
    grant = machine_agent._build_borrow_admission_grant(
        runtime, readable, result, set(readable), lane=lane,
    )
    assert grant is not None and grant.is_valid(runtime.root)


def test_dependency_recheck_rounds_are_independent_between_lanes(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    readable = {}
    for index in range(2):
        cfg = init_shared_root(
            tmp_path / str(index) / ".qexp", "gpu-1", runtime_root=tmp_path / f"rt-{index}"
        )
        create_group(cfg, "dependencies")
        parent = submit(
            cfg, ["echo", "parent"], task_id="parent", group="dependencies",
            working_dir=tmp_path,
        )
        for lane in ("gpu", "cpu"):
            submit(
                cfg, ["echo", lane], task_id=lane, group="dependencies",
                working_dir=tmp_path, depends_on_task_ids=[parent.task_id],
                requested_gpus=1 if lane == "gpu" else 0,
                requested_cpus=1 if lane == "cpu" else None,
            )
        cancel(cfg, parent.task_id)
        _activate_ready(cfg)
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        readable[binding.project_id] = cfg

    def probe(lane: str) -> machine_agent.PrimaryDemandProbe:
        return machine_agent._probe_primary_demand(
            runtime, readable, readable, [0], [0],
            SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0), lane=lane,
        )

    for lane in ("gpu", "cpu"):
        assert probe(lane).state == "no_primary_demand"
    last_cfg = readable[sorted(readable)[-1]]
    for lane in ("gpu", "cpu"):
        edit_dependencies(last_cfg, lane, [])
    observed = set()
    for _ in range(8):
        for lane in ("gpu", "cpu"):
            if probe(lane).state == "runnable_now":
                observed.add(lane)
    assert observed == {"gpu", "cpu"}


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


def test_cpu_primary_revision_does_not_invalidate_a_gpu_borrow_grant(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg, _task = _borrow_project(tmp_path, work)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revisions = tuple(
        _BorrowAdmissionRevision(binding.project_id, cfg, scope, ready_index_route_revision(
            cfg, scope, primary_only=True, lane="gpu"
        ))
        for scope in ("shared", "home")
    )
    grant = _BorrowAdmissionGrant(runtime.root, revisions, lane="gpu")

    bump_primary_ready_revision(cfg, "shared", cfg.machine_name, lane="cpu")

    assert grant.is_valid(runtime.root)


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


def test_cpu_primary_probe_waits_for_aggregation_with_partial_free_capacity(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "cpu-primary" / ".qexp", "cpu-1", runtime_root=tmp_path / "cpu-project-runtime"
    )
    submit(cfg, ["echo", "primary"], requested_gpus=0, requested_cpus=4, working_dir=work)
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    set_cpu_lane_capacity(runtime.root, capacity=4)
    reserve_cpu(runtime.root, "occupied", 2, project_id="external")

    probe = machine_agent._probe_primary_demand(
        runtime,
        {binding.project_id: cfg},
        {binding.project_id: cfg},
        list(range(4)),
        list(range(2)),
        SliceBudget(WorkBudgetPolicy(), clock_ns=lambda: 0),
        lane="cpu",
    )

    assert probe.state == "waiting_for_aggregation"


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


def test_primary_rebuild_keeps_concurrently_published_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready", working_dir=work)
    _activate_ready(cfg)

    scan_finished = threading.Event()
    allow_rebuild = threading.Event()
    writer_reached_sync = threading.Event()
    writer_finished = threading.Event()
    original_iter_json = __import__(
        "qqtools.plugins.qexp.runtime.ready", fromlist=["iter_json"]
    ).iter_json
    original_sync = __import__(
        "qqtools.plugins.qexp.runtime.ready", fromlist=["_sync_primary_candidate"]
    )._sync_primary_candidate

    def pause_after_task_scan(directory):
        yield from original_iter_json(directory)
        if Path(directory) == shared_paths(cfg.shared_root)["tasks"]:
            scan_finished.set()
            assert allow_rebuild.wait(timeout=5)

    def record_writer_sync(*args, **kwargs):
        writer_reached_sync.set()
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.ready.iter_json", pause_after_task_scan
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.ready._sync_primary_candidate", record_writer_sync
    )
    rebuild_thread = threading.Thread(target=rebuild_primary_ready_index, args=(cfg,))
    rebuild_thread.start()
    assert scan_finished.wait(timeout=5)

    generation = task.ready_generation + 1

    def publish_marker() -> None:
        write_ready_marker(
            cfg,
            task,
            generation=generation,
            source_transition="test",
            source_revision=task.meta["revision"],
            target_revision=task.meta["revision"] + 1,
        )
        writer_finished.set()

    writer_thread = threading.Thread(target=publish_marker)
    writer_thread.start()
    assert writer_reached_sync.wait(timeout=5)
    assert not writer_finished.is_set()

    allow_rebuild.set()
    rebuild_thread.join(timeout=5)
    writer_thread.join(timeout=5)
    assert not rebuild_thread.is_alive()
    assert not writer_thread.is_alive()
    candidate = (
        shared_paths(cfg.shared_root)["ready_primary"]
        / "routes"
        / "home.gpu-1"
        / f"{task.task_id}.{generation}.json"
    )
    assert candidate.exists()


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
