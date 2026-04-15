from __future__ import annotations

import pytest

from qqtools.plugins.qexp.v2.models import (
    ALL_PHASES,
    AGENT_MODE_ON_DEMAND,
    AGENT_MODE_PERSISTENT,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_DRAINING,
    AGENT_STATE_STOPPED,
    AgentSnapshot,
    Batch,
    BatchPolicy,
    BatchSummary,
    GpuInventory,
    LEGAL_TRANSITIONS,
    Machine,
    MachineSummary,
    MachineWorkset,
    Meta,
    PHASE_CANCELLED,
    PHASE_DISPATCHING,
    PHASE_FAILED,
    PHASE_ORPHANED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    PHASE_SUCCEEDED,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    generate_id,
    utc_now_iso,
    validate_machine_name,
    validate_phase,
    validate_phase_transition,
    validate_task_id,
)


# ---------------------------------------------------------------------------
# utc_now_iso
# ---------------------------------------------------------------------------


class TestUtcNowIso:
    def test_ends_with_z(self):
        ts = utc_now_iso()
        assert ts.endswith("Z")

    def test_no_microseconds(self):
        ts = utc_now_iso()
        assert "." not in ts


# ---------------------------------------------------------------------------
# generate_id
# ---------------------------------------------------------------------------


class TestGenerateId:
    def test_length(self):
        assert len(generate_id()) == 12

    def test_unique(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# validate_task_id
# ---------------------------------------------------------------------------


class TestValidateTaskId:
    @pytest.mark.parametrize("tid", ["abc", "a-b_c.1", "TASK_001", "x" * 100])
    def test_valid(self, tid):
        assert validate_task_id(tid) == tid

    @pytest.mark.parametrize(
        "tid",
        ["", "a/b", "a\\b", "a..b", "../etc", "a b", "a@b"],
    )
    def test_invalid(self, tid):
        with pytest.raises(ValueError):
            validate_task_id(tid)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_task_id(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_machine_name
# ---------------------------------------------------------------------------


class TestValidateMachineName:
    @pytest.mark.parametrize("name", ["gpu2a", "machine-01", "m.1"])
    def test_valid(self, name):
        assert validate_machine_name(name) == name

    @pytest.mark.parametrize("name", ["", "a/b", "a\\b", "..x", "a b"])
    def test_invalid(self, name):
        with pytest.raises(ValueError):
            validate_machine_name(name)


# ---------------------------------------------------------------------------
# validate_phase / validate_phase_transition
# ---------------------------------------------------------------------------


class TestPhaseValidation:
    def test_all_phases_accepted(self):
        for phase in ALL_PHASES:
            assert validate_phase(phase) == phase

    def test_invalid_phase(self):
        with pytest.raises(ValueError):
            validate_phase("unknown")

    def test_legal_transitions(self):
        for from_p, to_p in LEGAL_TRANSITIONS:
            validate_phase_transition(from_p, to_p)

    def test_illegal_transition(self):
        with pytest.raises(ValueError):
            validate_phase_transition(PHASE_QUEUED, PHASE_RUNNING)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


class TestMeta:
    def test_new(self):
        m = Meta.new("gpu1")
        assert m.revision == 1
        assert m.updated_by_machine == "gpu1"
        assert m.created_at.endswith("Z")

    def test_round_trip(self):
        m = Meta.new("gpu1")
        d = m.to_dict()
        m2 = Meta.from_dict(d)
        assert m2.revision == m.revision
        assert m2.updated_by_machine == m.updated_by_machine


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_valid(self):
        s = TaskSpec(command=["python", "train.py"], requested_gpus=2)
        assert s.requested_gpus == 2

    def test_empty_command(self):
        with pytest.raises(ValueError):
            TaskSpec(command=[], requested_gpus=1)

    def test_zero_gpus(self):
        with pytest.raises(ValueError):
            TaskSpec(command=["echo"], requested_gpus=0)

    def test_round_trip(self):
        s = TaskSpec(command=["a", "b"], requested_gpus=1)
        assert TaskSpec.from_dict(s.to_dict()).command == ["a", "b"]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> Task:
    now = utc_now_iso()
    defaults = dict(
        meta=Meta.new("dev1"),
        task_id="task-001",
        name=None,
        batch_id=None,
        machine_name="dev1",
        attempt=1,
        spec=TaskSpec(command=["python", "train.py"], requested_gpus=1),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestTask:
    def test_create_valid(self):
        t = _make_task()
        assert t.task_id == "task-001"
        assert t.status.phase == PHASE_QUEUED

    def test_invalid_task_id(self):
        with pytest.raises(ValueError):
            _make_task(task_id="../evil")

    def test_invalid_machine_name(self):
        with pytest.raises(ValueError):
            _make_task(machine_name="a/b")

    def test_round_trip(self):
        t = _make_task(name="my task", batch_id="batch-1")
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.task_id == t.task_id
        assert t2.name == "my task"
        assert t2.batch_id == "batch-1"
        assert t2.meta.revision == t.meta.revision
        assert t2.spec.command == t.spec.command
        assert t2.status.phase == t.status.phase

    def test_dict_structure(self):
        t = _make_task()
        d = t.to_dict()
        assert "meta" in d
        assert "task" in d
        assert d["task"]["task_id"] == "task-001"
        assert d["task"]["spec"]["requested_gpus"] == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def _make_batch(**overrides) -> Batch:
    defaults = dict(
        meta=Meta.new("dev1"),
        batch_id="batch-001",
        name=None,
        source_manifest=None,
        machine_name="dev1",
        task_ids=[],
        summary=BatchSummary(),
        policy=BatchPolicy(),
    )
    defaults.update(overrides)
    return Batch(**defaults)


class TestBatch:
    def test_create_valid(self):
        b = _make_batch()
        assert b.batch_id == "batch-001"

    def test_invalid_batch_id(self):
        with pytest.raises(ValueError):
            _make_batch(batch_id="../x")

    def test_round_trip(self):
        b = _make_batch(
            name="sweep",
            task_ids=["t1", "t2"],
            summary=BatchSummary(total=2, queued=2),
        )
        d = b.to_dict()
        b2 = Batch.from_dict(d)
        assert b2.batch_id == "batch-001"
        assert b2.name == "sweep"
        assert b2.task_ids == ["t1", "t2"]
        assert b2.summary.total == 2

    def test_dict_structure(self):
        b = _make_batch()
        d = b.to_dict()
        assert "meta" in d
        assert "batch" in d
        assert d["batch"]["batch_id"] == "batch-001"


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


class TestMachine:
    def test_create_valid(self):
        m = Machine(
            machine_name="gpu1",
            hostname="host1",
            shared_root="/mnt/share",
            runtime_root="/home/user/.qqtools/runtime",
            agent_mode=AGENT_MODE_ON_DEMAND,
            gpu_inventory=GpuInventory(count=4, visible_gpu_ids=[0, 1, 2, 3]),
        )
        assert m.machine_name == "gpu1"

    def test_invalid_agent_mode(self):
        with pytest.raises(ValueError):
            Machine(
                machine_name="gpu1",
                hostname=None,
                shared_root="/x",
                runtime_root="/y",
                agent_mode="invalid",
                gpu_inventory=GpuInventory(),
            )

    def test_round_trip(self):
        m = Machine(
            machine_name="gpu1",
            hostname="host1",
            shared_root="/mnt/share",
            runtime_root="/home/user/.rt",
            agent_mode=AGENT_MODE_PERSISTENT,
            gpu_inventory=GpuInventory(count=2, visible_gpu_ids=[0, 1]),
        )
        d = m.to_dict()
        m2 = Machine.from_dict(d)
        assert m2.machine_name == "gpu1"
        assert m2.agent_mode == AGENT_MODE_PERSISTENT
        assert m2.gpu_inventory.count == 2


class TestMachineWorkset:
    def test_create_valid(self):
        workset = MachineWorkset(
            machine_name="gpu1",
            queued_count=1,
            dispatching_count=0,
            starting_count=0,
            running_count=2,
            terminal_count=3,
            has_launch_backlog=True,
            has_active_responsibility=True,
            updated_at="2026-04-14T00:00:00Z",
        )
        assert workset.machine_name == "gpu1"

    def test_flags_must_match_counts(self):
        with pytest.raises(ValueError):
            MachineWorkset(
                machine_name="gpu1",
                queued_count=0,
                dispatching_count=0,
                starting_count=0,
                running_count=1,
                terminal_count=0,
                has_launch_backlog=True,
                has_active_responsibility=True,
                updated_at="2026-04-14T00:00:00Z",
            )


class TestAgentSnapshot:
    def test_round_trip(self):
        snapshot = AgentSnapshot(
            schema_version="3.0",
            machine_name="gpu1",
            agent_mode=AGENT_MODE_ON_DEMAND,
            agent_state=AGENT_STATE_DRAINING,
            pid=123,
            started_at="2026-04-14T00:00:00Z",
            last_heartbeat="2026-04-14T00:00:01Z",
            last_transition_at="2026-04-14T00:00:01Z",
            idle_timeout_seconds=600,
            idle_started_at=None,
            idle_deadline_at=None,
            drain_started_at="2026-04-14T00:00:01Z",
            last_exit_reason=None,
            workset=MachineWorkset(
                machine_name="gpu1",
                queued_count=0,
                dispatching_count=0,
                starting_count=0,
                running_count=1,
                terminal_count=0,
                has_launch_backlog=False,
                has_active_responsibility=True,
                updated_at="2026-04-14T00:00:01Z",
            ),
        )
        loaded = AgentSnapshot.from_dict(snapshot.to_dict())
        assert loaded.agent_state == AGENT_STATE_DRAINING
        assert loaded.workset.running_count == 1

    def test_machine_name_must_match_workset(self):
        with pytest.raises(ValueError):
            AgentSnapshot(
                schema_version="3.0",
                machine_name="gpu1",
                agent_mode=AGENT_MODE_ON_DEMAND,
                agent_state=AGENT_STATE_ACTIVE,
                pid=123,
                started_at=None,
                last_heartbeat=None,
                last_transition_at=None,
                idle_timeout_seconds=600,
                idle_started_at=None,
                idle_deadline_at=None,
                drain_started_at=None,
                last_exit_reason=None,
                workset=MachineWorkset(
                    machine_name="gpu2",
                    has_launch_backlog=False,
                    has_active_responsibility=False,
                    updated_at="2026-04-14T00:00:00Z",
                ),
            )


class TestMachineSummary:
    def test_round_trip(self):
        summary = MachineSummary(
            machine_name="gpu1",
            counts_by_phase={"queued": 1, "running": 2},
            updated_at="2026-04-14T00:00:00Z",
        )
        loaded = MachineSummary.from_dict(summary.to_dict())
        assert loaded.machine_name == "gpu1"
        assert loaded.counts_by_phase["running"] == 2
