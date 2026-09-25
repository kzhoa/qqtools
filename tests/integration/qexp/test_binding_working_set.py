"""Machine-local residency over durable Project activation checkpoints."""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import registration as registration_module
from qqtools.plugins.qexp.agent import working_set as working_set_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane
from qqtools.plugins.qexp.agent.working_set import SERVICE_LANES, BindingWorkingSet
from qqtools.plugins.qexp.lease import LeasePolicy
from qqtools.plugins.qexp.runtime.project_activation import (
    activation_checkpoint_path,
    activation_event_path,
    activation_snapshot_path,
    compact_project_activation,
    publish_project_activation,
)
from qqtools.plugins.qexp.runtime.project_activation_consumers import read_consumer_progress
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


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
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        assert working_set.acknowledge(turn, quiescent=True)


def test_binding_retires_only_after_every_lane_acks_one_checkpoint(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    checkpoint = publish_project_activation(cfg, "initial")["project_activation"]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])

    scheduler = working_set.begin_turn(binding, "scheduler")
    authority = working_set.begin_turn(binding, "authority")
    remaining = [
        working_set.begin_turn(binding, lane) for lane in SERVICE_LANES if lane not in {"scheduler", "authority"}
    ]
    assert working_set.acknowledge(scheduler, quiescent=True)
    assert working_set.acknowledge(authority, quiescent=True)
    assert working_set.resident_bindings([binding]) == [binding]

    for turn in remaining:
        assert working_set.acknowledge(turn, quiescent=True)
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


def test_dormant_registration_renewal_is_bounded_and_fair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, configured = _bindings(tmp_path, 5)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)

    renewed = []

    def record_renewal(binding, *, renew=False, renewal_horizon_seconds=0.0):
        assert renew is True
        assert renewal_horizon_seconds == 20.0
        renewed.append(binding.project_id)
        return True

    monkeypatch.setattr(runtime, "binding_write_eligible", record_renewal)

    assert working_set.renew_dormant_registrations(limit=2) == (2, 2)
    assert working_set.renew_dormant_registrations(limit=2) == (2, 2)
    assert len(renewed) == 4
    assert len(set(renewed)) == 4


@pytest.mark.stress
def test_one_thousand_dormant_registrations_remain_eligible_with_real_ttl(
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
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    with working_set._lock:
        for identity, state in working_set._states.items():
            state.state = "dormant"
            state.startup_pending = False
            working_set._add_dormant_locked(identity)

    for _ in range(32):
        assert working_set.renew_dormant_registrations(limit=64, heartbeat_interval_seconds=5) == (64, 64)
        now[0] += timedelta(seconds=5)

    assert all(runtime.registration_status(binding)["write_eligible"] for binding in bindings)


def test_change_during_quiescent_handoff_prevents_retirement(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])

    turns = {lane: working_set.begin_turn(binding, lane) for lane in SERVICE_LANES}
    assert working_set.acknowledge(turns["scheduler"], quiescent=True)
    assert working_set.acknowledge(turns["authority"], quiescent=True)
    publish_project_activation(cfg, "racing_submission")

    assert not working_set.acknowledge(turns["group"], quiescent=True)
    assert working_set.resident_bindings([binding]) == [binding]


def test_publication_before_final_shared_ack_prevents_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    original_ack = working_set_module.ack_consumer
    published = [False]

    def race_ack(*args, **kwargs):
        if kwargs.get("require_current") and not published[0]:
            published[0] = True
            publish_project_activation(cfg, "racing_submission")
        return original_ack(*args, **kwargs)

    monkeypatch.setattr(working_set_module, "ack_consumer", race_ack)
    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert published[0]
    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]


def test_bounded_cold_poll_is_fair_and_wakes_changed_binding(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 5)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)
    publish_project_activation(configured[-1][0], "remote_submission")

    assert working_set.poll_dormant(bindings, limit=2) == []
    assert working_set.poll_dormant(bindings, limit=2) == []
    assert working_set.poll_dormant(bindings, limit=2) == [bindings[-1]]
    assert working_set.resident_bindings(bindings) == [bindings[-1]]
    snapshot = working_set.snapshot()
    assert snapshot["cold_probes"] == 6
    assert snapshot["activations"] == 1


def test_dormant_binding_wakes_after_activation_from_another_process(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _retire(working_set, binding)
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from qqtools.plugins.qexp.runtime.project_activation import publish_project_activation
publish_project_activation(SimpleNamespace(shared_root=Path(sys.argv[1])), "remote_process_submission")
"""

    subprocess.run([sys.executable, "-c", script, str(cfg.shared_root)], check=True)

    assert working_set.poll_dormant(limit=1) == [binding]
    assert working_set.resident_bindings() == [binding]


def test_cold_poll_skips_ineligible_dormant_bindings_without_spending_probe_budget(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 3)
    bindings = [binding for _cfg, binding in configured]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile(bindings)
    for binding in bindings:
        _retire(working_set, binding)

    publish_project_activation(configured[-1][0], "remote_submission")
    eligible = [bindings[0], bindings[-1]]

    assert working_set.poll_dormant(eligible, limit=1) == []
    assert working_set.poll_dormant(eligible, limit=1) == []
    assert working_set.poll_dormant(eligible, limit=1) == [bindings[-1]]
    assert working_set.snapshot()["cold_probes"] == 2
    assert working_set.resident_bindings(bindings) == [bindings[-1]]


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
    now[0] += 60.0
    assert working_set.poll_dormant([binding], limit=1) == [binding]
    assert working_set.resident_bindings([binding]) == [binding]


def test_corrupt_wake_checkpoint_activates_unknown_work(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    _retire(working_set, binding)
    path = cfg.shared_root / "operations/project-activation-v1/checkpoint.json"
    atomic_replace(path, {"project_activation": {"version": 999}})

    assert working_set.poll_dormant([binding], limit=1) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1


def test_missing_activation_event_blocks_shared_ack_and_dormancy(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    checkpoint = publish_project_activation(cfg, "work")["project_activation"]
    activation_event_path(cfg.shared_root, checkpoint["epoch"], checkpoint["sequence"]).unlink()
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])

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
    original = working_set_module.register_consumer
    attempts = [0]

    def fail_once(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            raise OSError("temporary shared I/O failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(working_set_module, "register_consumer", fail_once)
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])
    assert working_set.snapshot()["unknown_bindings"] == 1

    working_set.reconcile([binding])
    assert attempts[0] == 2
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

    assert working_set.poll_dormant([binding], limit=1) == [binding]
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
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == [bindings[0]]
    _retire(working_sets[0], bindings[0])
    publish_project_activation(cfg, "second_change")
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == [bindings[0]]

    assert working_sets[1].poll_dormant([bindings[1]], limit=1) == [bindings[1]]
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
    assert working_sets[0].poll_dormant([bindings[0]], limit=1) == [bindings[0]]
    _retire(working_sets[0], bindings[0])
    compact_project_activation(cfg.shared_root)
    compact_project_activation(cfg.shared_root)

    assert not activation_event_path(cfg.shared_root, first_activation["epoch"], first_activation["sequence"]).exists()
    assert working_sets[1].poll_dormant([bindings[1]], limit=1) == [bindings[1]]
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


def test_corrupt_compaction_snapshot_blocks_reconstruction_and_dormancy(tmp_path: Path) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "existing_work")
    compact_project_activation(cfg.shared_root)
    atomic_replace(activation_snapshot_path(cfg.shared_root), {"project_activation_snapshot": {"version": 999}})
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])

    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1


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


def test_shared_ack_failure_keeps_binding_resident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, configured = _bindings(tmp_path, 1)
    cfg, binding = configured[0]
    publish_project_activation(cfg, "initial")
    working_set = BindingWorkingSet(runtime, process_fence="process-a")
    working_set.reconcile([binding])

    def fail_ack(*_args, **_kwargs):
        raise OSError("shared progress unavailable")

    monkeypatch.setattr(working_set_module, "ack_consumer", fail_ack)
    results = []
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        results.append(working_set.acknowledge(turn, quiescent=True))

    assert results[-1] is False
    assert working_set.resident_bindings([binding]) == [binding]
    assert working_set.snapshot()["unknown_bindings"] == 1
