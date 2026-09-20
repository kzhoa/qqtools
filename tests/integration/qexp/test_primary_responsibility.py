"""Qualified active discovery preserves startup acknowledgement and degraded safety."""

import os
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.recovery_capture import RecoveryCapture, recovery_owner
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.runtime import responsibility as membership
from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes
from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.recovery_admission import fence_recovery_admission
from qqtools.plugins.qexp.runtime.responsibility import ResponsibilityReader, responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_completion import COMPLETION_FILE
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger, identity_key
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    root = runtime.project_paths(binding.project_id)["root"]
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").symlink_to("/proc/self/ns/pid")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    with runtime.scheduler_authority() as owns:
        assert owns
        assert runtime.prepare_recovery_registration(binding)
        assert fence_recovery_admission(runtime, binding).is_fenced
        capture = RecoveryCapture(runtime, binding)
        for _ in range(10):
            if capture.advance():
                break
        else:
            pytest.fail("finite capture did not complete")
        yield replace(cfg, runtime_root=root), recovery_owner(runtime, binding), Ledger(responsibility_root(root))


def wait_reader(reader):
    if reader._done is not None:
        assert reader._done.wait(5), "isolated discovery worker did not finish"


def close_reader(reader):
    reader.close()
    wait_reader(reader)


def last_bucket_member(ledger):
    identity = next(f"task-{n}-attempt-1" for n in range(1000) if identity_key(f"task-{n}-attempt-1")[0] == "f")
    ledger.publish(identity, {"task_id": identity.rsplit("-attempt-", 1)[0], "attempt_number": 1})
    return identity


def test_qualified_empty_startup_and_steady_service_never_enumerate_history(qualified, monkeypatch):
    cfg, owner, _ledger = qualified
    original = EvidenceScan.take
    forbidden = set(local_paths(cfg.runtime_root).values())
    original_scandir = os.scandir

    def guarded(scan, *args, **kwargs):
        assert scan.directory not in forbidden, f"primary discovery enumerated {scan.directory}"
        return original(scan, *args, **kwargs)

    def guarded_scandir(path="."):
        if isinstance(path, int):
            raise AssertionError("qualified primary discovery used an integer-fd scandir")
        if Path(path) in forbidden:
            raise AssertionError(f"primary discovery enumerated {path}")
        return original_scandir(path)

    supervisor = None
    reader = None
    # This qualifies supervisor construction, startup recovery, and steady work;
    # machine registration/bootstrap and whole-machine scale remain out of scope.
    with monkeypatch.context() as guarded_scope:
        guarded_scope.setattr(EvidenceScan, "take", guarded)
        guarded_scope.setattr(os, "scandir", guarded_scandir)
        try:
            supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
            supervisor.recover_startup()
            reader = supervisor._work._responsibilities
            for _ in range(80):
                supervisor.tick()
                wait_reader(reader)
                if supervisor.work_snapshot["startup_complete"]:
                    break
            else:
                pytest.fail("empty qualified startup did not complete")
            assert supervisor.work_snapshot["discovery_mode"] == "primary"
            for _ in range(32):
                supervisor.tick()
                wait_reader(reader)
            assert all(lane["entries"] == 0 for lane in supervisor.work_snapshot["lanes"].values())
            assert supervisor.work_snapshot["cleanup_entries"] == 0
        finally:
            if reader is not None:
                try:
                    close_reader(reader)
                finally:
                    if supervisor is not None:
                        supervisor.close()
            elif supervisor is not None:
                supervisor.close()


def test_prefetched_last_candidate_requires_semantic_acknowledgement(qualified):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    held = None
    try:
        for _ in range(80):
            entry = reader.take()
            if entry is not None:
                held = entry
                break
            wait_reader(reader)
        assert held and held["identity"] == identity
        assert reader._result_initial_complete
        assert not reader.is_initial_sweep_complete
        reader.poll()
        wait_reader(reader)
        reader.poll()
        assert not reader.is_initial_sweep_complete
        reader.acknowledge()
        # A fresh batch may have replaced the last-bucket completion flag. Drive
        # it to another EOF without allowing that prefetch to acknowledge held work.
        for _ in range(80):
            if reader.is_initial_sweep_complete:
                break
            if reader.take() is not None:
                reader.acknowledge()
            wait_reader(reader)
        assert reader.is_initial_sweep_complete
    finally:
        close_reader(reader)


def test_failed_candidate_restarts_initial_coverage(qualified):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    rejected = False
    recovered = False
    try:
        for _ in range(120):
            entry = reader.take()
            if entry is not None:
                assert entry["identity"] == identity
                if not rejected:
                    reader.acknowledge(is_success=False)
                    rejected = True
                    assert not reader.is_initial_sweep_complete
                else:
                    reader.acknowledge()
                    recovered = True
            wait_reader(reader)
            if recovered and reader.is_initial_sweep_complete:
                break
        assert rejected and recovered and reader.is_initial_sweep_complete
    finally:
        close_reader(reader)


@pytest.mark.parametrize("damage", ["certificate", "generation", "capability", "ledger"])
def test_unavailable_qualification_never_completes_startup(qualified, damage):
    cfg, owner, _ledger = qualified
    if damage == "certificate":
        (cfg.runtime_root / COMPLETION_FILE).write_text("broken")
    elif damage == "generation":
        owner = {**owner, "registration_generation": "different-generation"}
    elif damage == "ledger":
        (responsibility_root(cfg.runtime_root) / "marker").write_text("broken")
    else:
        path = cfg.shared_root / "schema/version.json"
        schema = read_json(path)
        schema["schema"]["required_capabilities"].remove("local-recovery-v1")
        atomic_replace(path, schema)
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        for _ in range(24):
            supervisor.tick()
            wait_reader(reader)
        assert supervisor.work_snapshot["discovery_mode"] == "unavailable"
        assert not supervisor.work_snapshot["startup_complete"]
        assert supervisor.work_snapshot["discovery_error"]
    finally:
        close_reader(reader)
        supervisor.close()


def test_indexed_termination_only_member_is_serviced(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / identity
    atomic_replace(
        directory / "decision.json",
        {
            "termination_decision": {
                "attempt_id": identity,
                "decision_id": "decision",
                "state": "signal_committed",
            }
        },
    )
    sent = []
    original = EvidenceScan.take
    historical_roots = set(local_paths(cfg.runtime_root).values())
    original_scandir = os.scandir

    def guarded(scan, *args, **kwargs):
        assert scan.directory == directory, f"indexed termination enumerated unrelated evidence: {scan.directory}"
        return original(scan, *args, **kwargs)

    def guarded_scandir(path="."):
        if isinstance(path, int):
            raise AssertionError("indexed termination used an integer-fd scandir")
        scan_path = Path(path)
        if scan_path in historical_roots and scan_path != directory:
            raise AssertionError(f"indexed termination enumerated {scan_path}")
        return original_scandir(path)

    supervisor = None
    reader = None
    # This qualifies the indexed termination path during startup and steady
    # service; machine registration/bootstrap and whole-machine scale remain out of scope.
    with monkeypatch.context() as guarded_scope:
        guarded_scope.setattr(EvidenceScan, "take", guarded)
        guarded_scope.setattr(os, "scandir", guarded_scandir)
        try:
            supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
            monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
            monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
            monkeypatch.setattr(supervisor, "_send_signals", lambda *args: sent.append(args))
            supervisor.recover_startup()
            reader = supervisor._work._responsibilities
            for _ in range(80):
                supervisor.tick()
                wait_reader(reader)
                if sent:
                    break
            assert sent == [(identity, "decision")]
        finally:
            if reader is not None:
                try:
                    close_reader(reader)
                finally:
                    if supervisor is not None:
                        supervisor.close()
            elif supervisor is not None:
                supervisor.close()


def test_failed_maintenance_does_not_reset_active_startup(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    generation = ledger.publish("finished", {"task_id": "finished", "attempt_number": 1})
    ledger.handoff("finished", generation)
    original = Ledger.service_page
    maintenance_calls = []

    def unavailable(self, *args, **kwargs):
        if kwargs.get("stage") == "maintenance":
            maintenance_calls.append(args)
            raise OSError("maintenance storage unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Ledger, "service_page", unavailable)
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    try:
        for _ in range(80):
            if reader.take() is not None:
                reader.acknowledge()
            wait_reader(reader)
            if reader.is_initial_sweep_complete:
                break
        assert reader.is_initial_sweep_complete
        for _ in range(32):
            if reader.take() is not None:
                reader.acknowledge()
            wait_reader(reader)
            assert reader.is_initial_sweep_complete
        assert maintenance_calls and reader.failures
    finally:
        close_reader(reader)


def test_disappearing_completion_revokes_readiness_until_revalidated(qualified):
    cfg, owner, _ledger = qualified
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    path = cfg.runtime_root / COMPLETION_FILE
    proof = read_json(path)
    try:
        for _ in range(80):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert supervisor.work_snapshot["startup_complete"]
        path.unlink()
        for _ in range(24):
            supervisor.tick()
            wait_reader(reader)
        assert supervisor.work_snapshot["discovery_mode"] == "unavailable"
        assert not supervisor.work_snapshot["startup_complete"]
        atomic_replace(path, proof)
        for _ in range(80):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert supervisor.work_snapshot["startup_complete"]
    finally:
        close_reader(reader)
        supervisor.close()


def test_failed_initial_termination_is_retried_before_readiness(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / identity
    atomic_replace(
        directory / "decision.json",
        {
            "termination_decision": {
                "attempt_id": identity,
                "decision_id": "decision",
                "state": "signal_committed",
            }
        },
    )
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
    failures = []

    def unavailable(*args):
        failures.append(args)
        raise OSError("signalling temporarily unavailable")

    monkeypatch.setattr(supervisor, "_send_signals", unavailable)
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        for _ in range(40):
            supervisor.tick()
            wait_reader(reader)
        assert failures
        assert not supervisor.work_snapshot["startup_complete"]
        sent = []
        monkeypatch.setattr(supervisor, "_send_signals", lambda *args: sent.append(args))
        for _ in range(80):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert sent and supervisor.work_snapshot["startup_complete"]
    finally:
        close_reader(reader)
        supervisor.close()


def test_worker_completion_cannot_acknowledge_uncollected_candidates(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    # End the preceding four-bucket batch so the last candidate stays uncollected.
    earlier = next(f"task-{n}-attempt-1" for n in range(1000) if identity_key(f"task-{n}-attempt-1")[0] == "b")
    ledger.publish(earlier, {"task_id": earlier.rsplit("-attempt-", 1)[0], "attempt_number": 1})
    last = last_bucket_member(ledger)
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    published = Event()
    release = Event()
    original = membership._READ_WORKERS.release

    def delay_publication():
        published.set()
        assert release.wait(5), "test did not release completed worker"
        original()

    try:
        for _ in range(80):
            entry = reader.take()
            if entry is not None:
                break
            wait_reader(reader)
        assert entry["identity"] == earlier
        monkeypatch.setattr(membership._READ_WORKERS, "release", delay_publication)
        reader.poll()
        assert published.wait(5)
        assert reader._result_initial_complete
        assert any(item["identity"] == last for item in reader._result)
        reader.acknowledge()
        assert not reader.is_initial_sweep_complete
        release.set()
        wait_reader(reader)
        entry = reader.take()
        assert entry["identity"] == last
        assert not reader.is_initial_sweep_complete
        reader.acknowledge()
        assert reader.is_initial_sweep_complete
    finally:
        release.set()
        close_reader(reader)


def test_qualification_io_does_not_block_foreground_or_close(qualified, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_qualification

    cfg, owner, _ledger = qualified
    entered = Event()
    release = Event()
    original = responsibility_qualification.qualify_discovery

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release qualification"
        return original(*args, **kwargs)

    monkeypatch.setattr(responsibility_qualification, "qualify_discovery", delayed)
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        supervisor.tick()
        assert entered.wait(5)
        supervisor.tick()
        assert supervisor.work_snapshot["discovery_mode"] == "checking"
        assert not supervisor.work_snapshot["startup_complete"]
        supervisor.close()
        assert not release.is_set()
    finally:
        release.set()
        close_reader(reader)
        supervisor.close()


def test_initial_termination_queue_drains_each_indexed_attempt(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    paths = local_paths(cfg.runtime_root)
    identities = {f"task-{number}-attempt-1" for number in range(16)}
    for identity in identities:
        ledger.publish(identity, {"task_id": identity.rsplit("-attempt-", 1)[0], "attempt_number": 1})
        for number in range(10):
            atomic_replace(
                paths["termination_decisions"] / identity / f"decision-{number}.json",
                {
                    "termination_decision": {
                        "attempt_id": identity,
                        "decision_id": f"decision-{number}",
                        "state": "signal_committed",
                    }
                },
            )
    supervisor = AuthoritySupervisor(cfg, work_limit=4, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
    sent = set()
    monkeypatch.setattr(supervisor, "_send_signals", lambda *args: sent.add(args))
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        for _ in range(400):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert supervisor.work_snapshot["startup_complete"]
        assert sent == {(identity, f"decision-{number}") for identity in identities for number in range(10)}
    finally:
        close_reader(reader)
        supervisor.close()


@pytest.mark.parametrize("lane", ["registrations", "launch_intents"])
def test_materialized_manifest_is_supervised_before_candidate_ack(qualified, monkeypatch, lane):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    paths = local_paths(cfg.runtime_root)
    atomic_replace(paths[lane] / f"{identity}.json", {})
    supervisor = AuthoritySupervisor(cfg, work_limit=1, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)

    def materialize(*args, **kwargs):
        atomic_replace(
            paths["processes"] / f"{identity}.json", {"process": {"attempt_id": identity, "protocol_version": 1}}
        )

    monkeypatch.setattr(supervisor, "_materialize_registrations", materialize)
    monkeypatch.setattr(supervisor, "_materialize_unverified_intent", materialize)
    supervised = []
    monkeypatch.setattr(supervisor, "_supervise", lambda value: supervised.append(value["attempt_id"]))
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        for _ in range(160):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert supervisor.work_snapshot["startup_complete"]
        assert identity in supervised
    finally:
        close_reader(reader)
        supervisor.close()


@pytest.mark.parametrize("kind", ["supervision", "recovery"])
@pytest.mark.parametrize("cache_limit", [0, 256])
def test_occupied_control_slot_requires_each_initial_candidate_to_finish(qualified, monkeypatch, kind, cache_limit):
    from qqtools.plugins.qexp.runtime import attempt_recovery

    cfg, owner, ledger = qualified
    identities = [f"task-{n}-attempt-1" for n in range(1000) if identity_key(f"task-{n}-attempt-1")[0] == "f"][:2]
    for identity in identities:
        task_id = identity.rsplit("-attempt-", 1)[0]
        ledger.publish(identity, {"task_id": task_id, "attempt_number": 1})
        atomic_replace(
            local_paths(cfg.runtime_root)["processes"] / f"{identity}.json",
            {"process": {"protocol_version": 1, "task_id": task_id, "attempt_id": identity, "fencing_token": 1}},
        )
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
    supervisor.recover_startup()
    work = supervisor._work
    work._active_limit = cache_limit
    completed = set()

    def steps(identity):
        if kind == "recovery" and identity in completed:
            return
        yield
        yield
        completed.add(identity)

    if kind == "supervision":
        monkeypatch.setattr(supervisor, "supervision_steps", lambda process: steps(process["attempt_id"]))
        monkeypatch.setattr(supervisor, "_supervise", work.supervise)
    else:
        monkeypatch.setattr(
            attempt_recovery, "recovery_steps", lambda cfg, task, identity, token, **kw: steps(identity)
        )
        # Successful recovery changes authority truth; the real supervisor no
        # longer queues recovery for that Attempt on each subsequent discovery.
        monkeypatch.setattr(
            supervisor,
            "_supervise",
            lambda process: None if process["attempt_id"] in completed else work.recover(process),
        )
    reader = work._responsibilities
    try:
        for _ in range(200):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                assert completed == set(identities)
                break
        assert supervisor.work_snapshot["startup_complete"]
    finally:
        close_reader(reader)
        supervisor.close()


@pytest.mark.parametrize("failure", ["later_step", "cancellation"])
def test_unsuccessful_retained_control_requires_replay_before_readiness(qualified, monkeypatch, failure):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    atomic_replace(
        local_paths(cfg.runtime_root)["processes"] / f"{identity}.json",
        {"process": {"protocol_version": 1, "attempt_id": identity}},
    )
    supervisor = AuthoritySupervisor(cfg, work_limit=64, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
    supervisor.recover_startup()
    work = supervisor._work
    can_complete = False
    completed = []

    def steps(process):
        yield
        if not can_complete:
            raise OSError("deferred control unavailable")
        completed.append(process["attempt_id"])

    monkeypatch.setattr(supervisor, "supervision_steps", steps)
    monkeypatch.setattr(supervisor, "_supervise", work.supervise)
    reader = work._responsibilities
    try:
        for _ in range(80):
            supervisor.tick()
            wait_reader(reader)
            if work._pending_control is not None:
                break
        assert work._pending_control is not None
        assert not supervisor.work_snapshot["startup_complete"]
        if failure == "cancellation":
            supervisor.cancel_pending_control()
        else:
            supervisor.tick()
        assert not work.is_startup_complete
        for _ in range(32):
            supervisor.tick()
            wait_reader(reader)
            assert not supervisor.work_snapshot["startup_complete"]
        can_complete = True
        for _ in range(120):
            supervisor.tick()
            wait_reader(reader)
            if supervisor.work_snapshot["startup_complete"]:
                break
        assert completed and supervisor.work_snapshot["startup_complete"]
    finally:
        close_reader(reader)
        supervisor.close()


def test_cached_short_supervision_progresses_while_another_control_is_pending(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    identity = last_bucket_member(ledger)
    process = {"protocol_version": 1, "attempt_id": identity}
    atomic_replace(local_paths(cfg.runtime_root)["processes"] / f"{identity}.json", {"process": process})
    supervisor = AuthoritySupervisor(cfg, work_limit=1, recovery_owner=owner)
    monkeypatch.setattr(supervisor, "_cleanup_request_for_membership", lambda _entry: None)
    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda _identity: False)
    supervisor.recover_startup()
    work = supervisor._work
    reader = work._responsibilities
    serviced = []

    def steps(value):
        if value["attempt_id"] == "long-running":
            yield
            yield
        else:
            serviced.append(value["attempt_id"])

    monkeypatch.setattr(supervisor, "supervision_steps", steps)
    monkeypatch.setattr(supervisor, "_supervise", work.supervise)
    try:
        reader.poll()
        wait_reader(reader)
        reader.poll()
        assert reader.discovery_mode == "primary"
        work._remember(identity)
        work.supervise({"attempt_id": "long-running"})
        work._active_step()
        assert work.is_control_pending("long-running")
        assert serviced == [identity]
    finally:
        close_reader(reader)
        supervisor.close()


def test_empty_initial_coverage_batches_buckets_with_bounded_read_work(qualified, monkeypatch):
    cfg, owner, _ledger = qualified
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    calls = []
    original = Ledger.service_page

    def observed(ledger, bucket, cursor=None, limit=64, *, stage=None):
        calls.append((bucket, limit, stage))
        return original(ledger, bucket, cursor, limit, stage=stage)

    monkeypatch.setattr(Ledger, "service_page", observed)
    try:
        for batch in range(4):
            reader.poll()
            wait_reader(reader)
            assert len(calls) == (batch + 1) * 4
            assert sum(limit for _bucket, limit, _stage in calls[batch * 4 :]) == 64
        assert {bucket for bucket, _limit, _stage in calls} == set(range(16))
        assert all(stage == "active" for _bucket, _limit, stage in calls)
        reader.poll()
        assert reader.is_initial_sweep_complete
        wait_reader(reader)
        steady = calls[16:]
        assert [limit for _bucket, limit, _stage in steady] == [12, 12, 12, 12, 16]
        assert [stage for _bucket, _limit, stage in steady] == ["active"] * 4 + ["maintenance"]
        assert len({bucket for bucket, _limit, stage in steady if stage == "active"}) == 4
    finally:
        close_reader(reader)


def test_late_launch_discovery_does_not_wait_sixteen_machine_cycles(qualified, monkeypatch):
    cfg, owner, ledger = qualified
    reader = ResponsibilityReader(cfg.runtime_root, owner=owner)
    try:
        for _ in range(20):
            reader.poll()
            wait_reader(reader)
            if reader.is_initial_sweep_complete:
                break
        assert reader.is_initial_sweep_complete
        # Collect the last prefetch without starting another one, then publish
        # just behind the cursor: the worst bucket for a new local launch.
        with monkeypatch.context() as guard:
            guard.setattr(reader, "_start", lambda: None)
            assert reader.take() is None
        bucket = (reader._traversal.next_bucket - 1) % 16
        identity = next(f"late-{n}" for n in range(1000) if int(identity_key(f"late-{n}")[0], 16) == bucket)
        ledger.publish(identity, {"task_id": "late", "attempt_number": 1})
        found = None
        for _ in range(4):
            reader.poll()
            wait_reader(reader)
            with monkeypatch.context() as guard:
                guard.setattr(reader, "_start", lambda: None)
                found = reader.take()
            if found is not None:
                reader.acknowledge()
                break
        assert found is not None and found["identity"] == identity
    finally:
        close_reader(reader)
        assert reader._done is None or reader._done.is_set()
