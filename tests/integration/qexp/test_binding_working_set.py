"""Machine-local residency over durable Project activation checkpoints."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import project_io_supervision as project_io_supervision_module
from qqtools.plugins.qexp.agent import registration as registration_module
from qqtools.plugins.qexp.agent import working_set as working_set_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane
from qqtools.plugins.qexp.agent.dispatch_loop import _advance_activation_working_set
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.working_set import SERVICE_LANES, BindingWorkingSet
from qqtools.plugins.qexp.lease import LeasePolicy
from qqtools.plugins.qexp.runtime.project_activation import (
    activation_checkpoint_path,
    activation_event_path,
    activation_snapshot_path,
    compact_project_activation,
    publish_project_activation,
)
from qqtools.plugins.qexp.runtime.project_activation_consumers import ack_consumer, read_consumer_progress
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_missing_evidence_watch_ignores_sibling_and_detects_exact_target(tmp_path: Path) -> None:
    parent = tmp_path / "local"
    parent.mkdir()
    target = parent / "processes"
    watch = project_io_supervision_module._DirectoryMutationWatch()
    try:
        watch.watch(target)
        atomic_replace(parent / "unrelated.json", {"unrelated": True})
        assert watch.unchanged()
        target.mkdir()
        assert not watch.unchanged()
    finally:
        watch.close()


def test_existing_evidence_watch_detects_same_census_directory_mutation(tmp_path: Path) -> None:
    target = tmp_path / "processes"
    target.mkdir()
    watch = project_io_supervision_module._DirectoryMutationWatch()
    try:
        watch.watch(target)
        atomic_replace(target / "attempt.json", {"process": {"version": 1}})
        assert not watch.unchanged()
    finally:
        watch.close()


def test_evidence_watch_fails_closed_when_inotify_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise OSError("inotify unavailable")

    monkeypatch.setattr(project_io_supervision_module.ctypes, "CDLL", unavailable)
    watch = project_io_supervision_module._DirectoryMutationWatch()
    assert not watch.unchanged()
    watch.close()


def _bindings(tmp_path: Path, count: int):
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    bindings = []
    for index in range(count):
        cfg = init_shared_root(
            tmp_path / f"project-{index}" / ".qexp",
            "gpu-1",
            runtime_root=tmp_path / f"project-{index}-runtime",
        )
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        bindings.append((cfg, binding))
    return runtime, bindings


def _retire(working_set: BindingWorkingSet, binding) -> None:
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        assert not working_set.acknowledge(turn, quiescent=True)
    _drive_activation(working_set, lambda: _state(working_set, binding).state == "dormant")


def _state(working_set: BindingWorkingSet, binding):
    return working_set._states[working_set._identity(binding)]


def _make_cold_poll_due(working_set: BindingWorkingSet, binding) -> None:
    _state(working_set, binding).next_cold_reconcile_at = 0.0


def test_lane_quiescence_survives_other_lanes_but_not_a_local_wake(tmp_path):
    runtime, pairs = _bindings(tmp_path, 1)
    binding = pairs[0][1]
    revision, bindings = runtime.load_registry_snapshot()
    working_set = runtime.working_set
    working_set.reconcile(bindings, revision=revision)
    runtime.activation_wake.capture(revision, bindings)
    try:
        _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)
        submission_turn = working_set.begin_turn(binding, "submission")
        assert not working_set.is_lane_quiescent(binding, "submission")
        assert not working_set.acknowledge(submission_turn, quiescent=True)
        assert working_set.is_lane_quiescent(binding, "submission")
        assert working_set.resident_bindings() == [binding]
        for lane in SERVICE_LANES:
            if lane == "submission":
                continue
            turn = working_set.begin_turn(binding, lane)
            assert not working_set.acknowledge(turn, quiescent=True)
            assert working_set.is_lane_quiescent(binding, "submission")
        # Lane proofs remain parked while the final replay observation is due.
        assert _state(working_set, binding).reason == "activation_ack_pending"
        assert working_set.resident_bindings() == [binding]
        with runtime.agent_lifecycle_guard():
            runtime.activation_wake.publish_locked()
        assert not runtime.activation_wake.is_current()
        runtime.activation_wake.capture(revision, bindings)
        assert not working_set.is_lane_quiescent(binding, "submission")
        assert not working_set.acknowledge(submission_turn, quiescent=True)
        assert working_set.resident_bindings() == [binding]
    finally:
        if getattr(runtime, "project_io_executor", None) is not None:
            runtime.project_io_executor.shutdown()


def _drive_activation(working_set: BindingWorkingSet, predicate, *, timeout: float = 15.0) -> None:
    runtime = working_set._runtime
    runtime.working_set = working_set
    executor = getattr(runtime, "project_io_executor", None)
    if executor is None:
        executor = ProjectIOExecutor(runtime)
        executor.begin_epoch()
        runtime.project_io_executor = executor
        runtime.project_io_controller = ProjectIOController(runtime, executor)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        revision, bindings = runtime.load_registry_snapshot()
        _advance_activation_working_set(runtime, runtime.project_io_controller, revision, bindings)
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("activation working-set state did not converge")


def test_binding_retires_only_after_every_lane_acks_one_checkpoint(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    checkpoint = publish_project_activation(cfg, "initial")["project_activation"]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: _state(working_set, binding).consumer_registered)
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)

    scheduler = working_set.begin_turn(binding, "scheduler")
    authority = working_set.begin_turn(binding, "authority")
    remaining = [
        working_set.begin_turn(binding, lane) for lane in SERVICE_LANES if lane not in {"scheduler", "authority"}
    ]
    assert not working_set.acknowledge(scheduler, quiescent=True)
    assert not working_set.acknowledge(authority, quiescent=True)
    assert working_set.resident_bindings([binding]) == [binding]

    for turn in remaining:
        assert not working_set.acknowledge(turn, quiescent=True)
    _drive_activation(working_set, lambda: _state(working_set, binding).state == "dormant")
    assert working_set.resident_bindings([binding]) == []
    assert working_set.snapshot()["dormant_bindings"] == 1
    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )
    assert progress["project_activation_consumer"]["ack"] == {
        "epoch": checkpoint["epoch"],
        "sequence": checkpoint["sequence"],
    }


def test_activation_wiring_retains_rotated_unresolved_observation_and_ack(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 65)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    runtime.working_set = working_set
    working_set.reconcile(bindings)
    checkpoint = (uuid.uuid4().hex, 1)
    for binding in bindings:
        state = _state(working_set, binding)
        state.consumer_registered = True
        state.activation_observed = True
        state.unknown = False
        state.observation_due = True

    first_observations = working_set.select_activation_observations(limit=64)
    retained_observation = first_observations[-1]
    request = SimpleNamespace(
        operation_kind="activation_observe",
        project_id=retained_observation.binding.project_id,
        registration_generation=retained_observation.binding.registration_generation,
        parameters={"replay_epoch": None, "replay_sequence": 0},
    )

    class _Executor:
        def unresolved_requests(self):
            return (request,)

    class _Controller:
        executor = _Executor()
        observed = ()
        acknowledged = ()

        def advance_activation_observations(self, selected, _revision, _cursors):
            self.observed = tuple(selected)
            return {}

        def advance_activation_consumer_acks(self, selected, _revision, _acks):
            self.acknowledged = tuple(selected)
            return {}

    controller = _Controller()
    _advance_activation_working_set(runtime, controller, 65, bindings)
    assert retained_observation.binding in controller.observed
    assert len(controller.observed) <= 64

    for binding in bindings:
        state = _state(working_set, binding)
        state.checkpoint = checkpoint
        state.replay_proposal = (*checkpoint, None, True)
        state.observation_due = False
        state.next_observation_at = float("inf")
        state.acknowledgements = dict.fromkeys(SERVICE_LANES, checkpoint)
        state.acknowledged_lanes = set(SERVICE_LANES)
    first_acks = working_set.select_activation_acknowledgements(limit=64)
    retained_ack = first_acks[-1]
    request.operation_kind = "activation_consumer_ack"
    request.project_id = retained_ack.binding.project_id
    request.registration_generation = retained_ack.binding.registration_generation
    request.parameters = {
        "epoch": retained_ack.epoch,
        "sequence": retained_ack.sequence,
        "reconstructed_floor": retained_ack.reconstructed_floor,
        "require_current": retained_ack.require_current,
    }
    _advance_activation_working_set(runtime, controller, 65, bindings)
    assert retained_ack.binding in controller.acknowledged
    assert len(controller.acknowledged) <= 64


def test_dormant_registration_renewal_is_bounded_and_fair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, configured = _bindings(tmp_path, 5)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)

    monkeypatch.setattr(
        runtime,
        "binding_write_eligible",
        lambda *_args, **_kwargs: pytest.fail("selection must not touch shared registration"),
    )

    first, first_horizon = working_set.select_dormant_registration_renewals(limit=2)
    second, second_horizon = working_set.select_dormant_registration_renewals(limit=2)

    assert first_horizon == second_horizon == 20.0
    selected = [binding.project_id for binding in (*first, *second)]
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_registration_renewal_covers_resident_and_dormant_bindings_fairly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, configured = _bindings(tmp_path, 5)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    _retire(working_set, bindings[0])
    monkeypatch.setattr(
        runtime,
        "binding_write_eligible",
        lambda *_args, **_kwargs: pytest.fail("selection must remain MachineRuntime-local"),
    )
    selected = []
    for _ in range(3):
        batch, horizon = working_set.select_registration_renewals(limit=2, heartbeat_interval_seconds=0.1)
        assert len(batch) == 2
        assert horizon == pytest.approx(0.4)
        selected.extend(batch)
    assert len(set(selected[:5])) == 5
    assert set(selected) == set(bindings)


@pytest.mark.stress
def test_one_thousand_dormant_registrations_are_selected_with_safe_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime.now(timezone.utc).replace(microsecond=0)]
    policy = LeasePolicy(ttl_seconds=120, renew_interval_seconds=119)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(registration_module, "datetime", FixedDateTime)
    monkeypatch.setattr(registration_module, "load_lease_policy", lambda _cfg: policy)
    monkeypatch.setattr(
        registration_module,
        "lease_expiry",
        lambda _policy: (now[0] + timedelta(seconds=policy.ttl_seconds)).isoformat(),
    )
    runtime, configured = _bindings(tmp_path, 1000)
    all_bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(all_bindings)
    with working_set._lock:
        for identity, state in working_set._states.items():
            state.state = "dormant"
            state.startup_pending = False
            working_set._add_dormant_locked(identity)

    selected = set()
    for _ in range(16):
        selected_bindings, renewal_horizon = working_set.select_dormant_registration_renewals(
            limit=64,
            heartbeat_interval_seconds=5,
        )
        assert len(selected_bindings) == 64
        assert renewal_horizon == 85
        selected.update(binding.project_id for binding in selected_bindings)
        now[0] += timedelta(seconds=5)

    assert selected == {binding.project_id for binding in all_bindings}


def test_change_during_quiescent_handoff_prevents_retirement(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)

    turns = {lane: working_set.begin_turn(binding, lane) for lane in SERVICE_LANES}
    assert not working_set.acknowledge(turns["scheduler"], quiescent=True)
    assert not working_set.acknowledge(turns["authority"], quiescent=True)
    changed = publish_project_activation(cfg, "racing_submission")["project_activation"]
    _state(working_set, binding).observation_due = True
    _drive_activation(
        working_set,
        lambda: _state(working_set, binding).checkpoint == (changed["epoch"], changed["sequence"]),
    )

    assert not working_set.acknowledge(turns["group"], quiescent=True)
    assert working_set.resident_bindings([binding]) == [binding]


def test_publication_before_final_shared_ack_prevents_retirement(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)
    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))
    intent = working_set.select_activation_acknowledgements(limit=1)[0]
    changed = publish_project_activation(cfg, "racing_submission")["project_activation"]
    with pytest.raises(ValueError, match="changed before the consumer could enter dormancy"):
        ack_consumer(
            cfg.shared_root,
            runtime_id=runtime.instance_id,
            project_id=binding.project_id,
            registration_generation=binding.registration_generation,
            process_fence=working_set.process_fence,
            epoch=intent.epoch,
            sequence=intent.sequence,
            reconstructed_floor=intent.reconstructed_floor,
            require_current=intent.require_current,
        )
    working_set.apply_activation_acknowledgements(
        [intent],
        {binding.project_id: {"outcome": "unavailable", "reason": "checkpoint_changed"}},
    )
    _drive_activation(
        working_set,
        lambda: _state(working_set, binding).checkpoint == (changed["epoch"], changed["sequence"]),
    )

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]


def test_bounded_due_cold_poll_is_fair_and_wakes_eligible_bindings(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 5)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)
        _make_cold_poll_due(working_set, binding)
    publish_project_activation(configured[-1][0], "remote_submission")

    assert working_set.poll_dormant(bindings, limit=2) == []
    assert working_set.poll_dormant(bindings, limit=2) == []
    assert working_set.poll_dormant(bindings, limit=2) == []
    _drive_activation(working_set, lambda: set(working_set.resident_bindings(bindings)) == set(bindings))
    assert set(working_set.resident_bindings(bindings)) == set(bindings)
    snapshot = working_set.snapshot()
    assert snapshot["cold_probes"] == 6
    assert snapshot["activations"] == len(bindings)


def test_dormant_binding_wakes_after_activation_from_another_process(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _retire(working_set, binding)
    _make_cold_poll_due(working_set, binding)
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from qqtools.plugins.qexp.runtime.project_activation import publish_project_activation
publish_project_activation(SimpleNamespace(shared_root=Path(sys.argv[1])), "remote_process_submission")
"""

    subprocess.run([sys.executable, "-c", script, str(cfg.shared_root)], check=True)

    assert working_set.poll_dormant(limit=1) == []
    _drive_activation(working_set, lambda: working_set.resident_bindings() == [binding])
    assert working_set.resident_bindings() == [binding]


def test_cold_poll_skips_ineligible_dormant_bindings_without_spending_probe_budget(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 3)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)
        _make_cold_poll_due(working_set, binding)

    publish_project_activation(configured[-1][0], "remote_submission")
    eligible = [bindings[0], bindings[-1]]

    assert working_set.poll_dormant(eligible, limit=1) == []
    assert working_set.poll_dormant(eligible, limit=1) == []
    assert working_set.poll_dormant(eligible, limit=1) == []
    _drive_activation(working_set, lambda: set(working_set.resident_bindings(bindings)) == set(eligible))
    assert working_set.snapshot()["cold_probes"] == 2
    assert set(working_set.resident_bindings(bindings)) == set(eligible)


def test_periodic_cold_reconciliation_recovers_a_change_after_its_early_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    now = [100.0]
    monkeypatch.setattr(working_set_module, "_monotonic", lambda: now[0])
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    publish_project_activation(cfg, "before_truth_commit")
    _retire(working_set, binding)

    assert working_set.poll_dormant([binding], limit=1) == []
    assert working_set.snapshot()["cold_probes"] == 0
    now[0] += 60.0
    assert working_set.poll_dormant([binding], limit=1) == []
    assert working_set.snapshot()["cold_probes"] == 1
    _drive_activation(working_set, lambda: working_set.resident_bindings([binding]) == [binding])
    assert working_set.resident_bindings([binding]) == [binding]


def test_corrupt_wake_checkpoint_activates_unknown_work(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _retire(working_set, binding)
    _make_cold_poll_due(working_set, binding)
    path = cfg.shared_root / "operations/project-activation-v1/checkpoint.json"
    atomic_replace(path, {"project_activation": {"version": 999}})

    assert working_set.poll_dormant([binding], limit=1) == []
    _drive_activation(working_set, lambda: working_set.snapshot()["unknown_bindings"] == 1)
    assert working_set.snapshot()["unknown_bindings"] == 1


def test_missing_activation_event_blocks_shared_ack_and_dormancy(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    checkpoint = publish_project_activation(cfg, "work")["project_activation"]
    activation_event_path(cfg.shared_root, checkpoint["epoch"], checkpoint["sequence"]).unlink()
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: _state(working_set, binding).consumer_registered)
    _drive_activation(
        working_set,
        lambda: _state(working_set, binding).reason == "checkpoint_unreadable",
        timeout=7.0,
    )

    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]
    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )
    assert progress["project_activation_consumer"]["ack"] is None


def test_process_restart_forces_cold_validation_before_dormancy(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, binding = configured[0]
    first = BindingWorkingSet(runtime, process_fence="process-a")
    first.reconcile([binding])
    _retire(first, binding)
    assert first.resident_bindings([binding]) == []

    restarted = BindingWorkingSet(runtime, process_fence="process-b")
    restarted.reconcile([binding])

    assert restarted.resident_bindings([binding]) == [binding]
    assert restarted.snapshot()["startup_pending_bindings"] == 1


def test_registry_revision_change_reactivates_newly_enabled_binding(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    revision, bindings = runtime.load_registry_snapshot()
    working_set.reconcile(bindings, revision=revision)
    _retire(working_set, binding)

    runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry_snapshot()
    working_set.reconcile(bindings, revision=revision)
    assert working_set.resident_bindings() == []

    enabled = runtime.set_enabled(binding.project_id, True)
    revision, bindings = runtime.load_registry_snapshot()
    working_set.reconcile(bindings, revision=revision)

    assert working_set.resident_bindings() == [enabled]

    working_set.reconcile([replace(enabled, enabled=False)], revision=revision - 1)
    assert working_set.resident_bindings() == [enabled]


def test_consumer_registration_retries_after_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, binding = configured[0]
    runtime.working_set = working_set = BindingWorkingSet(runtime, process_fence="process-a")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    original = executor.start
    attempts = [0]

    def fail_once(request_id):
        attempts[0] += 1
        if attempts[0] == 1:
            return None
        return original(request_id)

    monkeypatch.setattr(executor, "start", fail_once)
    working_set.reconcile([binding])
    assert working_set.snapshot()["unknown_bindings"] == 1
    revision, bindings = runtime.load_registry_snapshot()
    _advance_activation_working_set(runtime, controller, revision, bindings)
    assert attempts[0] == 1
    now[0] += 4.99
    _advance_activation_working_set(runtime, controller, revision, bindings)
    assert attempts[0] == 1
    now[0] += 0.02
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)
    assert attempts[0] >= 2
    assert working_set.snapshot()["unknown_bindings"] == 0
    _retire(working_set, binding)
    assert working_set.snapshot()["dormant_bindings"] == 1


def test_epoch_change_bootstraps_existing_consumer_and_retires_again(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    first = publish_project_activation(cfg, "first")["project_activation"]
    publish_project_activation(cfg, "second")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _retire(working_set, binding)

    new_epoch = uuid.uuid4().hex
    replacement = {**first, "epoch": new_epoch, "sequence": 1}
    event_path = activation_event_path(cfg.shared_root, new_epoch, 1)
    event_path.parent.mkdir(parents=True)
    atomic_replace(event_path, {"project_activation_event": replacement})
    atomic_replace(activation_checkpoint_path(cfg.shared_root), {"project_activation": replacement})

    _make_cold_poll_due(working_set, binding)
    assert working_set.poll_dormant([binding], limit=1) == []
    _drive_activation(working_set, lambda: working_set.resident_bindings([binding]) == [binding])
    _retire(working_set, binding)
    assert working_set.snapshot()["dormant_bindings"] == 1
    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )
    assert progress["project_activation_consumer"]["ack"] == {"epoch": new_epoch, "sequence": 1}


def test_offline_consumer_wakes_after_other_consumer_handles_multiple_changes(tmp_path: Path) -> None:
    root = tmp_path / "project" / ".qexp"
    cfg = init_shared_root(root, "gpu-1", runtime_root=tmp_path / "project-runtime-1")
    second_cfg = init_shared_root(root, "gpu-2", runtime_root=tmp_path / "project-runtime-2")
    runtimes = [MachineRuntime(tmp_path / f"machine-{index}") for index in range(2)]
    bindings = [
        runtime.add_binding(machine_cfg.shared_root, machine_cfg.machine_name)
        for runtime, machine_cfg in zip(runtimes, (cfg, second_cfg), strict=True)
    ]
    working_sets = [
        BindingWorkingSet(runtime, process_fence=f"process-{index}") for index, runtime in enumerate(runtimes)
    ]
    for working_set, binding in zip(working_sets, bindings, strict=True):
        working_set.reconcile([binding])
        _retire(working_set, binding)

    publish_project_activation(cfg, "first_change")
    _make_cold_poll_due(working_sets[0], bindings[0])
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == []
    _drive_activation(working_sets[0], lambda: working_sets[0].resident_bindings() == [bindings[0]])
    _retire(working_sets[0], bindings[0])
    publish_project_activation(cfg, "second_change")
    _make_cold_poll_due(working_sets[0], bindings[0])
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == []
    _drive_activation(working_sets[0], lambda: working_sets[0].resident_bindings() == [bindings[0]])

    _make_cold_poll_due(working_sets[1], bindings[1])
    assert working_sets[1].poll_dormant([bindings[1]], limit=1) == []
    _drive_activation(working_sets[1], lambda: working_sets[1].resident_bindings() == [bindings[1]])
    assert working_sets[1].snapshot()["activations"] == 1


def test_offline_consumer_reconstructs_after_repeated_event_compaction(tmp_path: Path) -> None:
    root = tmp_path / "project" / ".qexp"
    cfg = init_shared_root(root, "gpu-1", runtime_root=tmp_path / "project-runtime-1")
    second_cfg = init_shared_root(root, "gpu-2", runtime_root=tmp_path / "project-runtime-2")
    first_activation = publish_project_activation(cfg, "initial")["project_activation"]
    runtimes = [MachineRuntime(tmp_path / f"machine-{index}") for index in range(2)]
    bindings = [
        runtime.add_binding(machine_cfg.shared_root, machine_cfg.machine_name)
        for runtime, machine_cfg in zip(runtimes, (cfg, second_cfg), strict=True)
    ]
    working_sets = [
        BindingWorkingSet(runtime, process_fence=f"process-{index}") for index, runtime in enumerate(runtimes)
    ]
    for working_set, binding in zip(working_sets, bindings, strict=True):
        working_set.reconcile([binding])
        _retire(working_set, binding)

    publish_project_activation(cfg, "first_change")
    last = publish_project_activation(cfg, "second_change")["project_activation"]
    _make_cold_poll_due(working_sets[0], bindings[0])
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == []
    _drive_activation(working_sets[0], lambda: working_sets[0].resident_bindings() == [bindings[0]])
    _retire(working_sets[0], bindings[0])
    compact_project_activation(cfg.shared_root)
    compact_project_activation(cfg.shared_root)

    assert not activation_event_path(cfg.shared_root, first_activation["epoch"], first_activation["sequence"]).exists()
    _make_cold_poll_due(working_sets[1], bindings[1])
    assert working_sets[1].poll_dormant([bindings[1]], limit=1) == []
    _drive_activation(working_sets[1], lambda: working_sets[1].resident_bindings() == [bindings[1]])
    _retire(working_sets[1], bindings[1])
    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtimes[1].instance_id,
        project_id=bindings[1].project_id,
        registration_generation=bindings[1].registration_generation,
    )
    assert progress["project_activation_consumer"]["ack"] == {
        "epoch": last["epoch"],
        "sequence": last["sequence"],
    }


def test_new_consumer_reconstructs_current_work_after_compaction(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "project-runtime")
    last = publish_project_activation(cfg, "existing_work")["project_activation"]
    compact_project_activation(cfg.shared_root)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    working_set = BindingWorkingSet(runtime, process_fence="process-new")

    working_set.reconcile([binding])
    _retire(working_set, binding)

    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )
    assert progress["project_activation_consumer"]["ack"] == {
        "epoch": last["epoch"],
        "sequence": last["sequence"],
    }


def test_corrupt_compaction_snapshot_blocks_reconstruction_and_dormancy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "existing_work")
    compact_project_activation(cfg.shared_root)
    atomic_replace(activation_snapshot_path(cfg.shared_root), {"project_activation_snapshot": {"version": 999}})
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    runtime.working_set = working_set
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    starts = [0]
    original_start = executor.start

    def counted_start(request_id):
        starts[0] += 1
        return original_start(request_id)

    monkeypatch.setattr(executor, "start", counted_start)
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: starts[0] > 0 and bool(controller._service_backoff))

    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1
    assert not _state(working_set, binding).consumer_registered


def test_corrupt_consumer_progress_is_unknown_after_restart(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, binding = configured[0]
    first = BindingWorkingSet(runtime, process_fence="process-a")
    first.reconcile([binding])
    _retire(first, binding)
    progress = runtime.project_paths(binding.project_id)["root"] / "working-set-v1.json"
    atomic_replace(progress, {"binding_working_set": {"version": 999}})

    restarted = BindingWorkingSet(runtime, process_fence="process-b")
    restarted.reconcile([binding])

    assert restarted.resident_bindings([binding]) == [binding]
    assert restarted.snapshot()["unknown_bindings"] == 1


def test_reregistration_does_not_reuse_old_generation_ack(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, old_binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([old_binding])
    _retire(working_set, old_binding)
    new_binding = replace(old_binding, registration_generation=uuid.uuid4().hex)

    working_set.reconcile([new_binding])

    assert new_binding.registration_generation != old_binding.registration_generation
    assert working_set.resident_bindings([new_binding]) == [new_binding]
    assert working_set.snapshot()["startup_pending_bindings"] == 1


def test_unchanged_registry_metadata_reuses_parsed_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configured = _bindings(tmp_path, 3)
    fresh = MachineRuntime(runtime.root)
    registry_path = runtime.paths["registry"]
    original_read = registration_module.read_json
    reads = 0

    def counted_read(path):
        nonlocal reads
        if path == registry_path:
            reads += 1
        return original_read(path)

    monkeypatch.setattr(registration_module, "read_json", counted_read)

    first_revision, first = fresh.load_registry()
    second_revision, second = fresh.load_registry()
    snapshot_revision, first_snapshot = fresh.load_registry_snapshot()
    repeated_revision, second_snapshot = fresh.load_registry_snapshot()

    assert reads == 1
    assert first_revision == second_revision
    assert (
        first
        == second
        == sorted(
            [binding for _cfg, binding in configured],
            key=lambda binding: binding.project_id,
        )
    )
    assert first is not second
    assert snapshot_revision == repeated_revision == first_revision
    assert first_snapshot is second_snapshot


def test_authority_quiescence_never_reuses_eof_after_existing_evidence(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    process_dir = cfg.runtime_root / "processes"
    process_dir.mkdir(parents=True, exist_ok=True)
    atomic_replace(process_dir / "attempt.json", {"process": {"version": 1}})
    control = _MachineControlPlane(
        runtime,
        instance_id="test-agent",
        loop_interval=1.0,
        started_at="2026-09-25T00:00:00Z",
        available_gpus=[],
    )

    assert not control._local_evidence_quiescent(binding, cfg, ("processes",))
    assert not control._local_evidence_quiescent(binding, cfg, ("processes",))


def test_authority_eligibility_scan_treats_incomplete_page_as_evidence(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    process_dir = cfg.runtime_root / "processes"
    process_dir.mkdir(parents=True, exist_ok=True)
    (process_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
    atomic_replace(process_dir / "active.json", {"process": {"version": 1}})
    control = _MachineControlPlane(
        runtime,
        instance_id="test-agent",
        loop_interval=1.0,
        started_at="2026-09-25T00:00:00Z",
        available_gpus=[],
    )

    assert control._has_local_process(binding, cfg)
    assert control._has_local_process(binding, cfg)


def test_progress_write_failure_keeps_binding_resident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    _cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    original_replace = working_set_module.atomic_replace

    def fail_progress(path, value):
        if path.name == "working-set-v1.json":
            raise OSError("local runtime unavailable")
        return original_replace(path, value)

    monkeypatch.setattr(working_set_module, "atomic_replace", fail_progress)
    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert results == [False] * len(SERVICE_LANES)
    assert working_set.resident_bindings([binding]) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1


def test_shared_ack_failure_keeps_binding_resident(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _drive_activation(working_set, lambda: _state(working_set, binding).activation_observed)
    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))
    intent = working_set.select_activation_acknowledgements(limit=1)[0]
    working_set.apply_activation_acknowledgements(
        [intent],
        {binding.project_id: {"outcome": "unavailable", "reason": "shared progress unavailable"}},
    )

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1
