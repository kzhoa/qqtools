"""Optional Idle Job diagnostics never replace base status or process identity."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from qqtools.plugins.qexp.agent import idle_job as module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.store import atomic_replace

STAMP = "2026-11-24T10:00:05Z"
NOW = datetime(2026, 11, 24, 10, 0, 5, tzinfo=timezone.utc).timestamp()


def _detail(**changes):
    return {
        "code": "upgrade_blocks_idle",
        "agent_instance_id": "agent-a",
        "observed_at": STAMP,
        "since": "2026-11-24T10:00:00Z",
        "project_count": 1,
        "projects": [
            {
                "project_id": "project-a",
                "migration": "task-dependencies-v2",
                "phase": "backfill",
                "state": "repair_required",
                "last_progress_at": None,
                "detail_command": "qexp --machine-runtime-root /srv/machine admin upgrade status --project project-a",
            }
        ],
        **changes,
    }


def _observation(**changes):
    detail = _detail(**changes)
    return {k: v for k, v in detail.items() if k not in {"agent_instance_id", "observed_at", "since"}}


@pytest.fixture
def supervised(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    set_idle_job_config(runtime, enabled=True, command="python -c pass", idle_delay_seconds=300)
    clock = [NOW]
    supervisor = module.IdleJobSupervisor(runtime, wall_clock=lambda: clock[0])
    return runtime, supervisor, clock


def _observe(supervisor, observation=None, *, instance="agent-a", demand="unknown"):
    supervisor.set_demand_observation(
        demand, observation if observation is not None else _observation(), agent_instance_id=instance
    )


def _envelope(runtime):
    return json.loads((runtime.root / "idle-job/status.json").read_text())


def test_publication_continuity_changed_blocker_known_demand_and_restart(supervised):
    runtime, supervisor, clock = supervised
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    first = _envelope(runtime)["idle_job_demand_detail"]
    assert first["observed_at"] == first["since"] == STAMP
    clock[0] += 5
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    repeated = _envelope(runtime)["idle_job_demand_detail"]
    assert repeated["since"] == first["since"]
    assert repeated["observed_at"] != first["observed_at"]
    _observe(supervisor, _observation(code="authority_unready"))
    supervisor._publish(state="waiting", reason="unknown_demand")
    changed = _envelope(runtime)["idle_job_demand_detail"]
    assert changed["since"] == changed["observed_at"]
    _observe(supervisor, demand="idle")
    supervisor._publish(state="waiting", reason="idle_delay")
    assert _envelope(runtime).get("idle_job_demand_detail") is None
    assert module.idle_job_status(runtime)["demand_detail"] is None
    clock[0] += 5
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    restarted = module.IdleJobSupervisor(runtime, wall_clock=lambda: clock[0] + 2)
    # The retained old-instance observation is not adopted as a live interval.
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import display_demand_detail

    retained = module.idle_job_status(runtime)["demand_detail"]
    assert display_demand_detail(retained, agent_instance_id="agent-b", now=clock[0])["live"] is False
    _observe(restarted, instance="agent-b")
    restarted._publish(state="waiting", reason="unknown_demand")
    fresh = _envelope(runtime)["idle_job_demand_detail"]
    assert fresh["agent_instance_id"] == "agent-b"
    assert fresh["since"] == fresh["observed_at"]
    assert fresh["since"] != repeated["since"]


@pytest.mark.parametrize(
    ("age", "identity", "live"),
    [
        (0, "agent-a", True),
        (60, "agent-a", True),
        (60.001, "agent-a", False),
        (-0.001, "agent-a", False),
        (10, None, False),
        (10, "agent-b", False),
    ],
)
def test_freshness_uses_observation_not_heartbeat(age, identity, live):
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import display_demand_detail

    result = display_demand_detail(_detail(), agent_instance_id=identity, now=NOW + age)
    assert result["live"] is live
    assert result["elapsed_seconds"] == (5 + age if live else None)
    assert result["observed_at"] == STAMP


def test_since_after_observation_retains_detail_but_no_duration():
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import display_demand_detail

    result = display_demand_detail(_detail(since="2026-11-24T10:00:06Z"), agent_instance_id="agent-a", now=NOW + 10)
    assert result["live"] is True
    assert result["elapsed_seconds"] is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("agent_instance_id", None),
        ("agent_instance_id", ""),
        ("agent_instance_id", []),
        ("observed_at", None),
        ("observed_at", "invalid"),
        ("observed_at", "2026-11-24T10:00:05"),
        ("since", "invalid"),
        ("project_count", -1),
        ("projects", "invalid"),
    ],
)
def test_malformed_optional_detail_is_null_without_losing_base(supervised, key, value):
    runtime, supervisor, _ = supervised
    supervisor._publish(state="waiting", reason="unknown_demand")
    envelope = _envelope(runtime)
    envelope["idle_job_demand_detail"] = _detail(**{key: value})
    atomic_replace(runtime.root / "idle-job/status.json", envelope)
    result = module.idle_job_status(runtime)
    assert result["state"] == "waiting"
    assert result["reason"] == "unknown_demand"
    assert result["demand_detail"] is None


def test_v1_reader_accepts_enriched_file_and_new_reader_accepts_base(supervised):
    runtime, supervisor, _ = supervised
    supervisor._publish(state="waiting", reason="unknown_demand")
    assert module.idle_job_status(runtime)["demand_detail"] is None
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    envelope = _envelope(runtime)
    # This is precisely the released v1 reader contract: select and validate the member.
    assert module._canonical_status(envelope["idle_job_status"])["reason"] == "unknown_demand"
    assert module.idle_job_status(runtime)["demand_detail"]["code"] == "upgrade_blocks_idle"


@pytest.mark.parametrize("reason", ["foreign_ownership", "agent_stop", "ownership_uncertain", "idle_delay"])
def test_supervisor_override_clears_optional_detail(supervised, reason):
    runtime, supervisor, _ = supervised
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    supervisor._publish(state="blocked" if reason != "idle_delay" else "waiting", reason=reason)
    assert _envelope(runtime).get("idle_job_demand_detail") is None
    assert module.idle_job_status(runtime)["demand_detail"] is None


def test_stopping_unknown_demand_retains_detail_and_config_override_clears(supervised):
    runtime, supervisor, _ = supervised
    _observe(supervisor)
    supervisor._publish(state="stopping", reason="unknown_demand")
    assert _envelope(runtime)["idle_job_demand_detail"]["code"] == "upgrade_blocks_idle"
    # Disabled configuration supersedes stale recorded waiting diagnostics.
    supervisor._publish(state="waiting", reason="unknown_demand")
    set_idle_job_config(runtime, enabled=False)
    result = module.idle_job_status(runtime)
    assert result["state"] == "disabled"
    assert result["demand_detail"] is None


def test_optional_failure_preserves_base_and_does_not_add_writes(supervised, monkeypatch):
    runtime, supervisor, _ = supervised
    writes = []
    original = module.atomic_replace

    def write(path, value, *args, **kwargs):
        writes.append((path, deepcopy(value)))
        return original(path, value, *args, **kwargs)

    monkeypatch.setattr(module, "atomic_replace", write)
    invalid = _observation(projects=[{"unencodable": object()}])
    _observe(supervisor, invalid)
    supervisor._publish(state="waiting", reason="unknown_demand")
    assert _envelope(runtime)["idle_job_status"]["reason"] == "unknown_demand"
    assert _envelope(runtime).get("idle_job_demand_detail") is None
    for _ in range(3):
        _observe(supervisor)
        supervisor._publish(state="waiting", reason="unknown_demand")
    assert len(writes) == 4
    before = (runtime.root / "idle-job/status.json").read_bytes()
    for _ in range(3):
        module.idle_job_status(runtime)
    assert len(writes) == 4
    assert (runtime.root / "idle-job/status.json").read_bytes() == before


def test_truncated_scope_has_no_claimed_continuity(supervised):
    runtime, supervisor, clock = supervised
    observation = _observation(
        project_count=7, projects=[{**_detail()["projects"][0], "project_id": f"project-{i}"} for i in range(3)]
    )
    for _ in range(2):
        _observe(supervisor, observation)
        supervisor._publish(state="waiting", reason="unknown_demand")
        assert _envelope(runtime)["idle_job_demand_detail"]["since"] is None
        clock[0] += 2


@pytest.mark.parametrize("identity", [None, (123, "agent-a", 42), (123, "agent-b", 42)])
def test_machine_status_uses_verified_agent_instance(supervised, monkeypatch, identity):
    from qqtools.plugins.qexp.agent import idle_job_diagnostics, lifecycle

    runtime, supervisor, _ = supervised
    _observe(supervisor)
    supervisor._publish(state="waiting", reason="unknown_demand")
    monkeypatch.setattr(lifecycle, "_active_machine_identity", lambda runtime: identity)
    original_display = idle_job_diagnostics.display_demand_detail
    monkeypatch.setattr(
        lifecycle,
        "display_demand_detail",
        lambda detail, **kwargs: original_display(detail, **{**kwargs, "now": NOW + 10}),
    )
    before = (runtime.root / "idle-job/status.json").read_bytes()
    result = lifecycle.get_machine_agent_status(runtime)
    detail = result["idle_job"]["demand_detail"]
    assert detail["live"] is (identity is not None and identity[1] == "agent-a")
    assert detail["elapsed_seconds"] == (10 if detail["live"] else None)
    assert (runtime.root / "idle-job/status.json").read_bytes() == before


@pytest.mark.parametrize("live", [False, True])
def test_human_explains_repair_and_last_observed_without_raw_nested_dump(live):
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import display_demand_detail
    from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render

    detail = display_demand_detail(_detail(), agent_instance_id="agent-a" if live else None, now=NOW + 2)
    payload = {
        "action": "status",
        "agent_state": "active",
        "registry_revision": 0,
        "upgrade": {"projects": []},
        "machine_runtime_root": "/srv/machine",
        "pid": 123,
        "is_running": live,
        "projects": [],
        "idle_job": {"state": "waiting", "reason": "unknown_demand", "demand_detail": detail},
    }
    human = render(CliOutput(OutputKind.AGENT_STATUS, payload), "human")
    assert "needs repair" in human
    assert "project-a" in human
    assert _detail()["projects"][0]["detail_command"] in human
    assert ("last observed" in human.lower()) is (not live)
    assert "agent_instance_id" not in human
    assert "upgrade_blocks_idle" not in human


def test_human_idle_delay_is_distinct_from_unknown_demand():
    from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render

    payload = {
        "action": "status",
        "agent_state": "active",
        "registry_revision": 0,
        "upgrade": {"projects": []},
        "machine_runtime_root": "/srv/machine",
        "pid": 123,
        "is_running": True,
        "projects": [],
        "idle_job": {"state": "waiting", "reason": "idle_delay", "demand_detail": None},
    }
    human = render(CliOutput(OutputKind.AGENT_STATUS, payload), "human")
    assert "Training is idle. Waiting for the configured start delay." in human
    assert "needs repair" not in human


def test_oversized_optional_encoding_preserves_small_base_status(supervised):
    runtime, supervisor, _ = supervised
    projects = [
        {**_detail()["projects"][0], "project_id": f"project-{i}", "detail_command": "字" * 8000} for i in range(3)
    ]
    _observe(supervisor, _observation(project_count=3, projects=projects))
    supervisor._publish(state="waiting", reason="unknown_demand")
    path = runtime.root / "idle-job/status.json"
    assert path.stat().st_size < 65536
    assert _envelope(runtime)["idle_job_status"]["reason"] == "unknown_demand"
    assert _envelope(runtime).get("idle_job_demand_detail") is None


def test_failed_unknown_tick_cannot_leak_pending_observation_into_fallback(supervised, monkeypatch):
    from qqtools.plugins.qexp.agent.idle_job_integration import _safe_tick

    runtime, supervisor, _ = supervised

    def fail(**kwargs):
        raise OSError("tick failed before publication")

    monkeypatch.setattr(supervisor, "tick", fail)
    assert _safe_tick(supervisor, "unknown", demand_detail=_observation(), agent_instance_id="agent-a") == {}
    supervisor._publish(state="waiting", reason="unknown_demand")
    assert _envelope(runtime).get("idle_job_demand_detail") is None


@pytest.mark.parametrize(("field", "value"), [("phase", "not-a-phase"), ("state", "not-a-state")])
def test_invalid_migration_semantics_cannot_produce_repair_warning(supervised, field, value):
    runtime, supervisor, _ = supervised
    supervisor._publish(state="waiting", reason="unknown_demand")
    envelope = _envelope(runtime)
    detail = _detail()
    detail["projects"][0][field] = value
    envelope["idle_job_demand_detail"] = detail
    atomic_replace(runtime.root / "idle-job/status.json", envelope)
    result = module.idle_job_status(runtime)
    assert result["reason"] == "unknown_demand"
    assert result["demand_detail"] is None
