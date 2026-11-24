"""Unknown demand attribution, bounded cached evidence, and decision equivalence."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import idle_job_integration as integration
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime


class Supervisor:
    def __init__(self):
        self.observations = []
        self.ticks = []

    def set_demand_observation(self, demand, detail, *, agent_instance_id):
        self.observations.append((demand, detail, agent_instance_id))

    def tick(self, demand="idle", gpu_policy=None, *, launch_allowed=True):
        self.ticks.append((demand, launch_allowed))
        return {"state": "waiting", "reason": "unknown_demand" if demand == "unknown" else demand}


@pytest.fixture
def demand_context(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding = ProjectBinding("project-a", tmp_path / "project", "gpu-1", registration_generation="generation-a")
    runtime.authority_ready_generations = {binding.project_id: binding.registration_generation}
    monkeypatch.setattr(integration, "load_idle_job_config", lambda runtime: SimpleNamespace(enabled=True))
    monkeypatch.setattr(integration, "_gpu_policy_observation", lambda *args, **kw: (None, {"revision": 1}, (0,)))
    monkeypatch.setattr(integration, "reservation_snapshot", lambda root: SimpleNamespace(active=(), provisional=()))
    monkeypatch.setattr(
        integration, "cpu_reservation_snapshot", lambda root: (SimpleNamespace(revision=1, capacity=1), ())
    )

    class Controller:
        def __init__(self):
            self.executor = SimpleNamespace(status_view=lambda: {"envelope": "healthy"})
            self.calls = 0
            self.demand = "idle"

        def advance_idle_job_demand(self, *args, **kwargs):
            self.calls += 1
            return self.demand

    monkeypatch.setattr(integration, "ProjectIOController", Controller)
    controller = Controller()
    runtime.project_io_controller = controller
    runtime.project_io_executor = controller.executor

    def advance(supervisor=None, **changes):
        supervisor = supervisor or Supervisor()
        arguments = dict(
            available_gpus=[0],
            instance_id="agent-a",
            cycle_completed=True,
            bindings=(binding,),
            registry_revision=7,
            enablement_converged=True,
            partition_observation="clear",
        )
        arguments.update(changes)
        integration.advance_machine_idle_job(runtime, supervisor, **arguments)
        return supervisor

    return runtime, binding, controller, advance


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("upgrade_discovery_unknown", "upgrade_evidence_unknown"),
        ("upgrade_idle_blocked_projects", "upgrade_blocks_idle"),
        ("recovery_enrollment_pending_projects", "recovery_enrollment_pending"),
        ("maintenance_retry_idle_blocked_projects", "maintenance_retry_pending"),
        ("pending_launch_handoffs", "launch_handoff_pending"),
    ],
)
def test_first_returning_check_retains_original_decision(demand_context, field, code):
    runtime, binding, controller, advance = demand_context
    setattr(runtime, field, True if field == "upgrade_discovery_unknown" else {binding.project_id})
    runtime.authority_ready_generations = {}
    supervisor = advance()
    assert supervisor.ticks == [("unknown", True)]
    assert supervisor.observations[-1][1]["code"] == code
    assert controller.calls == 0


def test_upgrade_discovery_precedes_upgrade_block_and_authority(demand_context):
    runtime, binding, _, advance = demand_context
    runtime.upgrade_discovery_unknown = True
    runtime.upgrade_idle_blocked_projects = {binding.project_id}
    runtime.authority_ready_generations = {}
    assert advance().observations[-1][1]["code"] == "upgrade_evidence_unknown"
    runtime.upgrade_discovery_unknown = False
    assert advance().observations[-1][1]["code"] == "upgrade_blocks_idle"
    runtime.upgrade_idle_blocked_projects.clear()
    detail = advance().observations[-1][1]
    assert detail["code"] == "authority_unready"
    assert detail["projects"][0]["project_id"] == binding.project_id


def test_upgrade_sampling_has_total_and_never_probes_projects(demand_context, monkeypatch):
    runtime, binding, _, advance = demand_context
    bindings = tuple(replace(binding, project_id=f"project-{i}") for i in range(7))
    runtime.upgrade_idle_blocked_projects = {b.project_id for b in bindings}

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics performed a Project-root probe")

    monkeypatch.setattr(ProjectBinding, "root_config", forbidden)
    first = advance(bindings=bindings)
    detail = first.observations[-1][1]
    assert detail["project_count"] == 7
    assert len(detail["projects"]) == 3
    assert {p["project_id"] for p in detail["projects"]} <= runtime.upgrade_idle_blocked_projects
    assert all(p["migration"] is None for p in detail["projects"])
    for _ in range(4):
        advance(first, bindings=bindings)
    assert first.ticks == [("unknown", True)] * 5


def test_census_unknown_and_idle_have_identical_controller_call_counts(demand_context):
    _, _, controller, advance = demand_context
    controller.demand = "unknown"
    supervisor = advance()
    assert supervisor.observations[-1][1]["code"] == "demand_census_unknown"
    controller.demand = "idle"
    advance(supervisor)
    assert supervisor.ticks == [("unknown", True), ("idle", True)]
    assert controller.calls == 2
    assert supervisor.observations[-1][0] == "idle"


def test_unexpected_observation_failure_is_generic_and_non_attributed(demand_context, monkeypatch):
    _, _, _, advance = demand_context

    def fail(root):
        raise OSError("unavailable")

    monkeypatch.setattr(integration, "reservation_snapshot", fail)
    detail = advance().observations[-1][1]
    assert detail["code"] == "observation_failed"
    assert detail["project_count"] is None
    assert detail["projects"] == []


@pytest.mark.parametrize("stale", [False, True])
def test_cached_repair_requires_current_matching_binding(demand_context, stale):
    runtime, binding, _, advance = demand_context
    # The synchronous discovery cache has full migration evidence; the isolated
    # controller's old startup cache is deliberately never treated as current.
    runtime.project_io_controller = None
    runtime.upgrade_idle_blocked_projects = {binding.project_id}
    runtime.upgrade_registry_revision = 7
    runtime.upgrade_binding_by_id = {
        binding.project_id: replace(binding, registration_generation="old") if stale else binding
    }
    runtime.upgrade_discovery_projects = {
        binding.project_id: {
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
            "state": "repair_required",
            "last_progress_at": "2026-11-24T09:00:00Z",
            "migrations": [{"name": "task-dependencies-v2", "state": "repair_required", "phase": "backfill"}],
        }
    }
    detail = advance().observations[-1][1]
    project = detail["projects"][0]
    assert project["state"] == (None if stale else "repair_required")
    assert project["migration"] == (None if stale else "task-dependencies-v2")
    assert project["last_progress_at"] is None  # journal/probe time is not progress


def test_controller_does_not_reuse_stale_startup_migration(demand_context):
    runtime, binding, _, advance = demand_context
    runtime.upgrade_idle_blocked_projects = {binding.project_id}
    runtime.upgrade_registry_revision = 7
    runtime.upgrade_binding_by_id = {binding.project_id: binding}
    runtime.upgrade_discovery_projects = {
        binding.project_id: {
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
            "migrations": [{"name": "task-dependencies-v2", "state": "repair_required", "phase": "backfill"}],
        }
    }
    assert advance().observations[-1][1]["projects"][0]["state"] is None


def test_repeated_agent_decisions_add_no_project_io_or_status_publications(demand_context, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job
    from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
    from qqtools.plugins.qexp.agent.setup import initialize_machine

    runtime, binding, controller, advance = demand_context
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    set_idle_job_config(runtime, enabled=True, command="python -c pass")
    supervisor = idle_job.IdleJobSupervisor(runtime)
    writes = []
    original = idle_job.atomic_replace

    def write(path, value, *args, **kwargs):
        if path == runtime.root / "idle-job/status.json":
            writes.append(value)
        return original(path, value, *args, **kwargs)

    monkeypatch.setattr(idle_job, "atomic_replace", write)
    baseline = [supervisor.tick(demand="unknown")["reason"] for _ in range(4)]
    baseline_count = len(writes)
    writes.clear()
    runtime.upgrade_idle_blocked_projects = {binding.project_id}

    def forbidden(*args, **kwargs):
        pytest.fail("a diagnostic added Project-root I/O")

    monkeypatch.setattr(ProjectBinding, "root_config", forbidden)
    observed = []
    for _ in range(4):
        advance(supervisor)
        observed.append(writes[-1]["idle_job_status"]["reason"])
    assert observed == baseline == ["unknown_demand"] * 4
    assert len(writes) == baseline_count == 4
    assert controller.calls == 0
    assert all(w["idle_job_demand_detail"]["code"] == "upgrade_blocks_idle" for w in writes)
