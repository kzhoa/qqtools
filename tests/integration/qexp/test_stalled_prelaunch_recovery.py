"""Exercise the owner and launch boundary through machine cycles and the operator CLI."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle
from qqtools.plugins.qexp.agent.project_admin import register_project
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.lease import ClockObservation
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import (
    ReservationIdentity,
    active_reservations,
    release_if_matches,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task
from tests.helpers.qexp.lifecycle import LifecycleLab, wait_until

pytestmark = pytest.mark.integration


@pytest.fixture
def machine(tmp_path, qexp_resource_scope):
    runtime = MachineRuntime(qexp_resource_scope.runtime_root)
    initialize_machine(runtime, "g3", agent_mode="daemon")
    cfg = init_shared_root(tmp_path / "project one" / ".qexp", "g3")
    binding = register_project(runtime, cfg.shared_root, cfg.machine_name).binding
    cfg = replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])
    stamp = datetime.now(timezone.utc).isoformat()
    observation = ClockObservation(
        "a" * 32,
        "linux_adjtimex",
        stamp,
        time.monotonic(),
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        -0.001,
        0.001,
        0,
        0.001,
    )
    atomic_replace(
        local_paths(cfg.runtime_root)["clock_health"],
        {
            "clock_capability": {
                "status": "healthy",
                "providers": ["linux_adjtimex"],
                "observation": asdict(observation),
            }
        },
    )
    return runtime, cfg, binding


def claim(machine, task, gpu, *, lease_seconds=None):
    runtime, cfg, binding = machine
    attempt = claim_task(
        cfg,
        task.task_id,
        [gpu],
        project_id=binding.project_id,
        reservation_runtime_root=runtime.root,
        lease_seconds=lease_seconds,
    )
    assert attempt is not None and attempt.authority_mode == "bounded_lease"
    return attempt


def set_deadline(cfg, attempt, seconds):
    stamp = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
    task = load_task(cfg, attempt.task_id)
    claimed = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    task.claim_control["active_claim"].update(lease_expires_at=stamp, claimed_at=claimed)
    attempt.lease.update(claimed_at=claimed, renewed_at=claimed)
    save_task(cfg, task)
    attempt.lease["expires_at"] = stamp
    atomic_replace(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number), attempt.to_dict())


def publish_heartbeat(runtime):
    now = datetime.now(timezone.utc).isoformat()
    atomic_replace(
        runtime.paths["agent"] / "status.json",
        {"machine_agent": {"instance_id": "test-agent", "runtime_id": runtime.instance_id, "state": "active"}},
    )
    atomic_replace(
        runtime.root / "control_plane_health.json",
        {
            "schema_version": 1,
            "instance_id": "test-agent",
            "heartbeat": {"state": "healthy", "last_success_at": now, "effective_stale_after_seconds": 15},
        },
    )


def cli(capsys, args):
    code = main([*args, "--format=json"])
    captured = capsys.readouterr()
    assert code == 0, captured
    return json.loads(captured.out)


def test_copyable_status_action_recovers_once_and_cannot_target_successor(machine, capsys):
    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    result = cli(capsys, [*prefix, "status"])
    (command,) = [shlex.split(cmd)[1:] for cmd in result["next_actions"] if "recover-prelaunch" in cmd]
    assert result["primary_action"]["command"] in result["next_actions"]
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["attempt_id"] == first.attempt_id
    recovered = cli(capsys, command)
    assert recovered["project_outcome"] == "requeued"
    assert recovered["machine_outcome"] == "released"
    assert load_task(cfg, task.task_id).state["projection"] == "queued"
    original = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
    assert original["phase"] == "cancelled" and original["result"]["reason"] == "lease_expired_before_launch"
    release = local_paths(runtime.root)["released"] / f"{first.reservation_id}.json"
    release_stat = release.stat()
    cli(capsys, command)
    successor = claim(machine, task, 1)
    set_deadline(cfg, successor, -60)
    before = read_json(attempt_path(cfg.shared_root, task.task_id, 2))
    cli(capsys, command)
    assert release.stat().st_mtime_ns == release_stat.st_mtime_ns
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 2)) == before
    assert [record["reservation_id"] for record in active_reservations(runtime.root)] == [successor.reservation_id]


@pytest.mark.parametrize("entry", ["cycle", "cli"])
def test_split_launch_commit_retains_original_reservation_and_replays(machine, capsys, monkeypatch, entry):
    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    args = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root), "status"]
    status = cli(capsys, args)
    (command,) = [shlex.split(cmd)[1:] for cmd in status["next_actions"] if "recover-prelaunch" in cmd]
    path = attempt_path(cfg.shared_root, task.task_id, 1)
    claimed = path.read_bytes()

    class BeforeDeadline(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) - timedelta(seconds=90)

    with monkeypatch.context() as launch_time:
        launch_time.setattr("qqtools.plugins.qexp.scheduler.datetime", BeforeDeadline)
        assert authorize_launch(
            cfg, task.task_id, first.attempt_id, first.current_fencing_token, reservation_runtime_root=runtime.root
        )
    receipt = load_task(cfg, task.task_id).claim_control["active_claim"]["ownership_transition"]
    path.write_bytes(claimed)
    if entry == "cli":
        cli(capsys, command)
    else:
        dispatch_machine_cycle(runtime, available_gpus=[], executor=SimpleNamespace(launch_attempt=lambda *_: None))
    assert [r["reservation_id"] for r in active_reservations(runtime.root)] == [first.reservation_id]
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["attempt_id"] == first.attempt_id
    assert authorize_launch(
        cfg, task.task_id, first.attempt_id, first.current_fencing_token, reservation_runtime_root=runtime.root
    )
    replay = read_json(path)["attempt"]
    assert replay["phase"] == "starting" and replay["authority_mode"] == "holder_bound"
    assert replay["authorization"]["launch_id"] == receipt["launch_id"]
    assert replay["timestamps"]["launch_authorized_at"] == receipt["authorization_timestamp"]


def test_daemon_recovers_prelaunch_gpu_while_other_execution_continues(machine, tmp_path, monkeypatch):
    runtime, cfg, _ = machine
    gate = tmp_path / "finish"
    incumbent_marker = tmp_path / "incumbent"
    script = (
        "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); "
        "exec('while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.05)')"
    )
    incumbent = submit(cfg, [sys.executable, "-c", script, str(incumbent_marker), str(gate)])
    lab = LifecycleLab(runtime=runtime, available_gpus=[0, 1])
    try:
        lab.start_agent()
        wait_until("incumbent training", incumbent_marker.exists, timeout=30)
        (incumbent_record,) = active_reservations(runtime.root)
        (gpu,) = {0, 1} - set(incumbent_record["gpu_ids"])
        # Start from the committed pre-launch boundary while the ordinary daemon remains live.
        marker = tmp_path / "executions"
        task = submit(cfg, [sys.executable, "-c", "import sys;open(sys.argv[1],'a').write('ran\\n')", str(marker)])

        class ClaimTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) - timedelta(minutes=3)

        with monkeypatch.context() as clock:
            clock.setattr("qqtools.plugins.qexp.runtime.records.datetime", ClaimTime)
            first = claim(machine, task, gpu, lease_seconds=120)
        queued = submit(cfg, ["true"])
        path = attempt_path(cfg.shared_root, task.task_id, 1)
        release_path = local_paths(runtime.root)["released"] / f"{first.reservation_id}.json"
        wait_until("owner committed release", release_path.exists, timeout=20, stage="pre-launch discovery")
        wait_until(
            "original expired and replacement completed",
            lambda: (
                load_task(cfg, task.task_id).state["projection"] == "succeeded"
                and load_task(cfg, queued.task_id).state["projection"] == "succeeded"
            ),
            timeout=60,
            on_timeout=lambda: (
                load_task(cfg, task.task_id).state,
                load_task(cfg, queued.task_id).state,
                active_reservations(runtime.root),
                marker.exists(),
                [read_json(p)["attempt"]["phase"] for p in sorted(path.parent.glob("*.json"))],
            ),
        )
        assert read_json(path)["attempt"]["result"]["reason"] == "lease_expired_before_launch"
        assert marker.read_text().splitlines() == ["ran"]
        released = read_json(local_paths(runtime.root)["released"] / f"{first.reservation_id}.json")
        assert released["reservation"]["release_reason"] == "lease_expired_before_launch"
        assert active_reservations(runtime.root)[0]["attempt_id"] == incumbent_record["attempt_id"]
        assert load_task(cfg, incumbent.task_id).state["projection"] == "running"
    finally:
        gate.touch()
        lab.close()


@pytest.mark.parametrize("successor_gpu", [None, 0, 1])
def test_project_commit_survives_owner_cleanup_interruption(machine, capsys, monkeypatch, successor_gpu):
    from qqtools.plugins.qexp import prelaunch_recovery

    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    args = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root), "status"]
    result = cli(capsys, args)
    (command,) = [shlex.split(cmd)[1:] for cmd in result["next_actions"] if "recover-prelaunch" in cmd]

    class OwnerInterrupted(BaseException):
        pass

    def interrupt_owner(*args):
        assert load_task(cfg, task.task_id).state["projection"] == "queued"
        raise OwnerInterrupted

    with monkeypatch.context() as boundary:
        boundary.setattr(prelaunch_recovery, "release_if_matches", interrupt_owner)
        with pytest.raises(OwnerInterrupted):
            main([*command, "--format=json"])
    atomic_replace(runtime.paths["agent"] / "status.json", {"machine_agent": {"state": "stopped"}})
    offline = cli(capsys, args)
    (pending,) = offline["prelaunch_claims"]
    assert pending["task_id"] == task.task_id
    assert pending["machine_outcome"] != "released"
    assert not any("recover-prelaunch" in cmd for cmd in offline["next_actions"])
    assert [record["gpu_ids"] for record in active_reservations(runtime.root)] == [[1]]
    release = local_paths(runtime.root)["released"] / f"{first.reservation_id}.json"
    assert not release.exists()
    deferred = cli(capsys, command)
    assert deferred["reason"] == "owner_runtime_unavailable"
    assert deferred["machine_outcome"] == "cleanup_pending"
    assert not release.exists()
    if successor_gpu == 1:
        # The owner committed release while the old command was in flight.
        identity = ReservationIdentity.from_record(active_reservations(runtime.root)[0])
        assert release_if_matches(runtime.root, identity, "lease_expired_before_launch")
    successor = claim(machine, task, successor_gpu) if successor_gpu is not None else None
    if successor_gpu == 1:
        publish_heartbeat(runtime)
        cli(capsys, command)
    returning = MachineRuntime(runtime.root)
    release_mtime = None
    for _ in range(2):
        dispatch_machine_cycle(returning, available_gpus=[], executor=SimpleNamespace(launch_attempt=lambda *_: None))
        assert read_json(release)["reservation"]["release_reason"] == "lease_expired_before_launch"
        if release_mtime is not None:
            assert release.stat().st_mtime_ns == release_mtime
        release_mtime = release.stat().st_mtime_ns
        assert [record["reservation_id"] for record in active_reservations(runtime.root)] == (
            [successor.reservation_id] if successor else []
        )
    current = load_task(cfg, task.task_id)
    if successor:
        assert current.claim_control["active_claim"]["attempt_id"] == successor.attempt_id
    else:
        assert current.state["projection"] == "queued"


@pytest.mark.parametrize("failures", [0, 3])
def test_snapshot_warning_explains_age_and_scope_without_recovery_action(machine, capsys, failures, monkeypatch):
    runtime, cfg, binding = machine
    publish_heartbeat(runtime)
    receipt = runtime.project_paths(binding.project_id)["root"] / "snapshot-observation.json"
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=67)).isoformat()
    envelope = {
        "runtime_id": runtime.instance_id,
        "agent_instance_id": "test-agent",
        "registration_generation": binding.registration_generation,
        "snapshot_at": stamp,
        "heartbeat_interval_seconds": 5,
    }
    atomic_replace(receipt, {"snapshot_observation": envelope})
    store = SchedulerDiagnosticStore(runtime)
    identity = {
        "producer": "control_plane_health",
        "reason_code": "project_snapshot_overdue",
        "component": "control_plane",
        "stage": "snapshot_publication",
        "check": "progress",
        "scope_type": "project_binding",
        "runtime_id": runtime.instance_id,
        "target_id": "test-agent",
        "project_id": binding.project_id,
        "registration_generation": binding.registration_generation,
        "operation_kind": "machine_snapshot_publish",
    }
    health_path = runtime.root / "control_plane_health.json"
    health = read_json(health_path)
    publication = {
        **envelope,
        "executor_epoch": "executor",
        "project_id": binding.project_id,
        "registry_revision": 1,
        "operation_kind": "machine_snapshot_publish",
        "overdue": True,
        "stage": "backoff" if failures else "intent",
        "consecutive_failures": failures,
        "reason_code": "publication_failed" if failures else None,
    }
    health["project_snapshot_publications"] = {"items": [publication]}
    atomic_replace(health_path, health)
    finding = store.observe_finding(
        identity=identity,
        severity="fault",
        source_revision={"registry_revision": 1},
        details={"consecutive_failures": failures},
    )
    args = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root), "status"]
    for _ in range(2):
        observed = cli(capsys, args)
        warning = observed["project_snapshot_warning"]
        assert warning["project_id"] == binding.project_id
        assert warning["registration_generation"] == binding.registration_generation
        assert warning["age_seconds"] >= 67 and warning["scheduling_impact"] == "unknown"
        assert warning["consecutive_failures"] == failures
        assert warning["publication_failure_reason"] == publication["reason_code"]
        assert warning["diagnostics_command"] not in observed["next_actions"]
        assert not any("recover-prelaunch" in cmd for cmd in observed["next_actions"])
        assert observed["local_agent"]["heartbeat"]["freshness"] == "fresh"
        assert observed["scheduling_readiness"]["state"] == "unknown"
        assert observed["budget"]["read_attempts"] <= 32
    assert (
        store.active_view(project_id=binding.project_id)["items"][0]["observation_revision"]
        == finding["observation_revision"]
    )
    from qqtools.plugins.qexp import prelaunch_observation

    training = submit(cfg, ["true"])
    incumbent = claim(machine, training, 0)
    assert authorize_launch(
        cfg,
        training.task_id,
        incumbent.attempt_id,
        incumbent.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    scan = prelaunch_observation._project_reservations
    monkeypatch.setattr(
        prelaunch_observation,
        "_project_reservations",
        lambda *args: sorted(scan(*args), key=lambda item: item.reservation_id != incumbent.reservation_id),
    )
    stalled = submit(cfg, ["true"])
    stranded = claim(machine, stalled, 1)
    set_deadline(cfg, stranded, -60)
    combined = cli(capsys, args)
    assert combined["project_snapshot_warning"]["diagnostics_command"] not in combined["next_actions"]
    assert "recover-prelaunch" in combined["primary_action"]["command"]
    assert combined["budget"]["read_attempts"] <= 32
    envelope["snapshot_at"] = datetime.now(timezone.utc).isoformat()
    atomic_replace(receipt, {"snapshot_observation": envelope})
    publication.update(overdue=False, stage="published", reason_code=None, consecutive_failures=0)
    atomic_replace(health_path, health)
    assert store.resolve_finding(
        identity=identity,
        episode=finding["episode"],
        observation_revision=finding["observation_revision"],
        source_revision={"registry_revision": 1},
    )
    assert cli(capsys, args)["project_snapshot_warning"] is None


@pytest.mark.parametrize(
    "state",
    [
        "waiting",
        "safety_window",
        "invalid_clock",
        "reservation_changed",
        "receipt_conflict",
        "registration_expired",
        "future_clock",
    ],
)
def test_copied_command_rechecks_current_authority(machine, capsys, state, monkeypatch):
    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    status = cli(capsys, [*prefix, "status"])
    (command,) = [shlex.split(cmd)[1:] for cmd in status["next_actions"] if "recover-prelaunch" in cmd]
    if state == "waiting":
        set_deadline(cfg, first, 60)
    elif state == "safety_window":
        set_deadline(cfg, first, -0.05)
        current = load_task(cfg, task.task_id)
        current.claim_control["active_claim"]["clock_error_bound_seconds"] = 0.9
        first.lease["clock_evidence"]["clock_error_bound_seconds"] = 0.9
        save_task(cfg, current)
        atomic_replace(attempt_path(cfg.shared_root, task.task_id, 1), first.to_dict())
    elif state == "registration_expired":
        from qqtools.plugins.qexp.runtime.paths import machine_registration_path

        path = machine_registration_path(cfg.shared_root, cfg.machine_name)
        record = read_json(path)
        record["registration"]["eligibility_expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        atomic_replace(path, record)
    elif state == "future_clock":
        path = local_paths(cfg.runtime_root)["clock_health"]
        record = read_json(path)
        record["clock_capability"]["observation"]["monotonic_observed_at"] = time.monotonic() + 1000
        atomic_replace(path, record)
        monkeypatch.setattr("qqtools.plugins.qexp.lease._provider_observations", lambda _: ([], ["unavailable"]))
    elif state == "reservation_changed":
        record_path = local_paths(runtime.root)["active"] / f"{first.reservation_id}.json"
        record = read_json(record_path)
        record["reservation"]["acquisition_id"] = "new-acquisition"
        atomic_replace(record_path, record)
    else:
        current = load_task(cfg, task.task_id)
        if state == "receipt_conflict":
            current.claim_control["active_claim"]["ownership_transition"] = {"operation_id": "incomplete"}
        else:
            current.claim_control["active_claim"]["clock_error_bound_seconds"] = None
        save_task(cfg, current)
    report = cli(capsys, command)
    assert report["reason"]
    assert report["machine_outcome"] != "released"
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["attempt_id"] == first.attempt_id
    assert [r["reservation_id"] for r in active_reservations(runtime.root)] == [first.reservation_id]
    if state != "reservation_changed":
        observed = cli(capsys, [*prefix, "status"])
        assert not any("recover-prelaunch" in cmd for cmd in observed["next_actions"])


def test_daemon_wins_before_copied_command_and_successor_is_also_overdue(machine, capsys):
    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    original = claim(machine, task, 1)
    set_deadline(cfg, original, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    status = cli(capsys, [*prefix, "status"])
    (command,) = [shlex.split(cmd)[1:] for cmd in status["next_actions"] if "recover-prelaunch" in cmd]
    dispatch_machine_cycle(runtime, available_gpus=[], executor=SimpleNamespace(launch_attempt=lambda *_: None))
    release_path = local_paths(runtime.root)["released"] / f"{original.reservation_id}.json"
    assert read_json(release_path)["reservation"]["release_reason"] == "lease_expired_before_launch"
    successor = claim(machine, task, 1)
    set_deadline(cfg, successor, -60)
    output = cfg.project_root / "training.log"
    output.write_text("successor output")
    before = load_task(cfg, task.task_id).to_dict()
    cli(capsys, command)
    assert load_task(cfg, task.task_id).to_dict() == before
    assert [record["reservation_id"] for record in active_reservations(runtime.root)] == [successor.reservation_id]
    assert output.read_text() == "successor output"


def test_placement_identifies_reserved_gpu_before_launch(machine, capsys):
    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    detail = cli(capsys, [*prefix, "task", "show", task.task_id, "--details"])
    data = json.dumps(detail)
    assert first.reservation_id in data
    assert runtime.instance_id in data
    assert "waiting_for_launch" in data
    for view in (["show", task.task_id, "--details"], ["list", "--view", "placement", "--limit", "1"]):
        assert main([*prefix, "task", *view]) == 0
        text = capsys.readouterr().out
        assert "reserved" in text.lower() and "launch" in text.lower()
        assert "1" in text


def test_authorization_wins_between_recovery_observation_and_authority_lock(machine, capsys, monkeypatch):
    from qqtools.plugins.qexp import prelaunch_recovery

    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    result = cli(capsys, [*prefix, "status"])
    (command,) = [shlex.split(cmd)[1:] for cmd in result["next_actions"] if "recover-prelaunch" in cmd]
    original_lock = prelaunch_recovery.authority_locks

    @contextmanager
    def interleaved_lock(*args, **kwargs):
        # The holder renews and authorizes before the reclaimer acquires the Task lock.
        set_deadline(cfg, first, 60)
        assert authorize_launch(
            cfg, task.task_id, first.attempt_id, first.current_fencing_token, reservation_runtime_root=runtime.root
        )
        with original_lock(*args, **kwargs) as held:
            yield held

    monkeypatch.setattr(prelaunch_recovery, "authority_locks", interleaved_lock)
    outcome = cli(capsys, command)
    assert outcome["machine_outcome"] != "released"
    claim_state = load_task(cfg, task.task_id).claim_control["active_claim"]
    assert claim_state["launch_state"] == "starting"
    assert claim_state["ownership_transition"]["launch_id"]
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "starting"
    assert active_reservations(runtime.root)[0]["reservation_id"] == first.reservation_id


def test_recovery_rechecks_registration_before_project_commit(machine, capsys, monkeypatch):
    runtime, cfg, binding = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    command = cli(capsys, [*prefix, "status"])["primary_action"]["command"]
    monkeypatch.setattr(MachineRuntime, "binding_write_eligible", lambda *_args, **_kwargs: False)
    report = cli(capsys, shlex.split(command)[1:])
    assert report["outcome"] == "deferred"
    assert report["reason"] == "runtime_binding_changed"
    assert report["project_outcome"] == "retained"
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    assert active_reservations(runtime.root)[0]["reservation_id"] == first.reservation_id


def test_recovery_holds_registration_fence_through_project_commit(machine, capsys, monkeypatch):
    from qqtools.plugins.qexp import prelaunch_recovery
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.paths import shared_paths

    runtime, cfg, _ = machine
    publish_heartbeat(runtime)
    task = submit(cfg, ["true"])
    first = claim(machine, task, 1)
    set_deadline(cfg, first, -60)
    prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
    command = cli(capsys, [*prefix, "status"])["primary_action"]["command"]
    commit = prelaunch_recovery._commit_expiry

    def check_fence(*args, **kwargs):
        with exclusive(shared_paths(cfg.shared_root)["locks"] / "registrations.lock", blocking=False) as acquired:
            assert not acquired
        return commit(*args, **kwargs)

    monkeypatch.setattr(prelaunch_recovery, "_commit_expiry", check_fence)
    assert cli(capsys, shlex.split(command)[1:])["outcome"] == "completed"


def test_public_recovery_allows_exactly_one_replacement_through_machine_dispatch(machine, tmp_path, capsys):
    runtime, cfg, _ = machine
    output = tmp_path / "launches"
    task = submit(cfg, [sys.executable, "-c", "import sys;open(sys.argv[1], 'a').write('ran\\n')", str(output)])
    original = claim(machine, task, 1)
    set_deadline(cfg, original, -60)
    # Omit only the original reservation from this daemon's automatic discovery.
    # Replacement reservations and all other daemon services use production paths.
    script = dedent("""
        import sys
        from pathlib import Path
        from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
        from qqtools.plugins.qexp.agent.project_io.controller import ProjectIOController
        original = ProjectIOController.advance_scheduler_reservation_reconciliations
        def paused(self, bindings, registry_revision, *, reservations):
            candidates = [r for r in reservations if r.reservation_id != sys.argv[2]]
            if len(candidates) != len(reservations):
                Path(sys.argv[3]).touch()
            return original(self, bindings, registry_revision, reservations=candidates)
        ProjectIOController.advance_scheduler_reservation_reconciliations = paused
        run_machine_agent_loop(sys.argv[1], available_gpus=[1], loop_interval=.1)
    """)
    reached = tmp_path / "discovery-paused"
    lab = LifecycleLab(runtime, [1])
    try:
        daemon = lab.adopt_process(
            subprocess.Popen(
                [sys.executable, "-c", script, str(runtime.root), original.reservation_id, str(reached)],
                start_new_session=True,
            )
        )
        wait_until("automatic discovery paused for original reservation", reached.exists, timeout=30)
        prefix = ["--machine-runtime-root", str(runtime.root), "--project", str(cfg.project_root)]
        details = cli(capsys, [*prefix, "task", "show", task.task_id, "--details"])
        assert details["prelaunch"][0]["attempt_phase"] == "claimed"
        assert details["prelaunch"][0]["gpu_ids"] == [1]
        from qqtools.plugins.qexp.agent.project_io.controller import _PRELAUNCH_RESERVATION_RECHECK_SECONDS

        time.sleep(_PRELAUNCH_RESERVATION_RECHECK_SECONDS)
        command = []

        def read_recovery_command():
            assert main([*prefix, "status"]) == 0
            for line in capsys.readouterr().out.splitlines():
                if line.startswith("Next: qexp ") and " recover-prelaunch " in line:
                    command[:] = shlex.split(line.removeprefix("Next: "))[1:]
                    return True
            return False

        wait_until("owner offers manual recovery", read_recovery_command, timeout=30, poll_interval=0.2)
        result = cli(capsys, command)
        assert (result["project_outcome"], result["machine_outcome"]) == ("requeued", "released"), result
        wait_until(
            "replacement execution", lambda: load_task(cfg, task.task_id).state["projection"] == "succeeded", timeout=30
        )
        assert daemon.poll() is None
        assert output.read_text().splitlines() == ["ran"]
        second = read_json(attempt_path(cfg.shared_root, task.task_id, 2))["attempt"]
        assert second["attempt_id"] != original.attempt_id
    finally:
        lab.close()
