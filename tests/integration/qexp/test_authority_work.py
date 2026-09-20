"""Bounded authority lanes over durable project and machine-local records."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.records import utc_now
from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _registration(cfg):
    task = submit(cfg, ["echo", "bounded"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    claim = load_task(cfg, task.task_id).claim_control["active_claim"]
    value = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "machine_name": cfg.machine_name,
        "authority_mode": attempt.authority_mode,
        "wrapper_pid": 99999991,
        "wrapper_start_time_ticks": 1,
        "process_group_id": 99999992,
        "process_group_start_time_ticks": 2,
        "process_created_at": utc_now(),
        "lease_expires_at": claim.get("lease_expires_at"),
        "clock_error_bound_seconds": claim.get("clock_error_bound_seconds"),
    }
    atomic_replace(
        local_paths(cfg.runtime_root)["registrations"] / f"{attempt.attempt_id}.json", {"process_registration": value}
    )
    return task, attempt


def _finish(cfg, attempt):
    path = local_paths(cfg.runtime_root)["observations"] / f"{attempt.attempt_id}.json"
    value = {
        "exit_observation": {
            "protocol_version": 1,
            "task_id": attempt.task_id,
            "attempt_id": attempt.attempt_id,
            "observed_exit_code": 0,
            "observed_at": utc_now(),
        }
    }
    atomic_replace(path, value)
    return path, value


def test_startup_and_each_slice_do_not_drain_retained_inventory(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    paths = local_paths(cfg.runtime_root)
    paths["registrations"].mkdir(parents=True, exist_ok=True)
    for number in range(1024):
        atomic_replace(
            paths["registrations"] / f"retained-{number}.json", {"process_registration": {"protocol_version": 0}}
        )
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    selected = []
    original = supervisor._materialize_registration

    def materialize(path):
        selected.append(path)
        original(path)

    def full_scan():
        pytest.fail("bounded service invoked a complete inventory scan")

    monkeypatch.setattr(supervisor, "_materialize_registration", materialize)
    monkeypatch.setattr(supervisor, "_remove_terminal_evidence", full_scan)
    supervisor.recover_startup()
    assert selected == []
    supervisor.tick()
    assert 30 <= len(selected) <= 38
    assert supervisor.work_snapshot["idle_steps_reassigned"] == 32
    assert supervisor.work_snapshot["operations"]["counters"]["store.read_json.calls"] >= len(selected)
    assert not supervisor.work_snapshot["startup_complete"]
    # Empty active turns are lent to discovery, keeping the total at 64 steps.
    for _ in range(30):
        before = len(selected)
        supervisor.tick()
        assert len(selected) - before <= 38
        if supervisor.work_snapshot["startup_complete"]:
            break
    assert supervisor.work_snapshot["startup_complete"]
    assert len(set(selected)) == 1024
    supervisor.close()


@pytest.mark.parametrize("authority_mode", ["bounded_lease", "holder_bound"])
def test_bounded_materialization_and_offline_completion_survive_restart(tmp_path, monkeypatch, authority_mode):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    if authority_mode == "holder_bound":
        from qqtools.plugins.qexp.lease import ClockCapability

        monkeypatch.setattr(
            "qqtools.plugins.qexp.scheduler.clock_capability",
            lambda *_args: ClockCapability("unavailable", "no_qualified_provider"),
        )
    task, attempt = _registration(cfg)
    assert attempt.authority_mode == authority_mode
    monkeypatch.setattr(AuthoritySupervisor, "_wrapper_matches", staticmethod(lambda process: True))
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    try:
        supervisor.recover_startup()
        supervisor.tick()
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "running"
        if authority_mode == "holder_bound":
            assert supervisor.metrics["renewal.not_required"] >= 1
            assert "renewal_lateness.maximum_seconds" not in supervisor.metrics
    finally:
        supervisor.close()
    _finish(cfg, attempt)
    restarted = AuthoritySupervisor(cfg, work_limit=1)
    try:
        restarted.recover_startup()
        for _ in range(80):
            restarted.tick()
            if load_task(cfg, task.task_id).state["projection"] == "succeeded":
                break
        current = load_task(cfg, task.task_id)
        assert current.state["projection"] == "succeeded"
        stored = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert stored["attempt_id"] == attempt.attempt_id
        assert stored["phase"] == "succeeded"
        assert not attempt_path(cfg.shared_root, task.task_id, 2).exists()
        assert not active_reservations(cfg.runtime_root)

        def full_inventory(*_args):
            pytest.fail("terminal replay enumerated unrelated reservations")

        monkeypatch.setattr("qqtools.plugins.qexp.authority.reservation_snapshot", full_inventory)
        assert restarted._reconcile_terminal_accounting(task.task_id, attempt.attempt_id)
        assert restarted._has_terminal_attempt(task.task_id, attempt.attempt_id)
    finally:
        restarted.close()


def test_failed_registration_replay_retains_exit_evidence_and_releases_local_capacity(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, attempt = _registration(cfg)
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor._materialize_registrations()
    observation, evidence = _finish(cfg, attempt)

    def unavailable(**_kwargs):
        raise OSError("shared publication unavailable")

    monkeypatch.setattr(supervisor, "_materialize_registrations", unavailable)
    try:
        supervisor.recover_startup()
        supervisor.tick()
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        assert read_json(observation) == evidence
        assert not active_reservations(cfg.runtime_root)
        assert supervisor.work_snapshot["lanes"]["registrations"]["failures"] > 0
    finally:
        supervisor.close()


def test_malformed_registration_does_not_starve_healthy_terminal_work(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    (local_paths(cfg.runtime_root)["registrations"] / "broken.json").write_text("invalid")
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    try:
        supervisor.recover_startup()
        for _ in range(10):
            supervisor.tick()
            if load_task(cfg, task.task_id).state["projection"] == "succeeded":
                break
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert not supervisor.work_snapshot["startup_complete"]
        assert supervisor.work_snapshot["lanes"]["registrations"]["failures"] > 0
    finally:
        supervisor.close()


def test_terminal_cleanup_removes_only_a_bounded_flat_decision_slice(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest, complete_cleanup
    from qqtools.plugins.qexp.runtime.responsibility_store import Ledger

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    paths = local_paths(cfg.runtime_root)
    directory = paths["termination_decisions"] / "completed"
    directory.mkdir(parents=True)
    for number in range(25):
        (directory / f"{number}.json").write_text("{}")
    manifest = paths["processes"] / "completed.json"
    atomic_replace(manifest, {"process": {"attempt_id": "completed"}})
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    request = CleanupRequest(
        "completed",
        {"task_id": "finished-task", "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": "finished-task", "attempt_id": "completed", "basis": "terminal_attempt"},
    )

    def recursive(*_args, **_kwargs):
        pytest.fail("bounded cleanup recursively enumerated decisions")

    with monkeypatch.context() as patch:
        patch.setattr("shutil.rmtree", recursive)
        assert not complete_cleanup(ledger, cfg.runtime_root, request)
        assert len(list(directory.iterdir())) == 17
        assert manifest.exists()
        assert ledger.lookup("completed")["cleanup_receipt"] == request.receipt
        for _ in range(3):
            complete_cleanup(ledger, cfg.runtime_root, request)
        assert not directory.exists()
        assert not manifest.exists()
        assert ledger.find("completed") is None


def test_unchanged_local_policy_and_state_avoid_writes_but_repair_damage(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import authority

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    supervisor = AuthoritySupervisor(cfg)
    writes = []
    original = authority.atomic_replace

    def record(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(authority, "atomic_replace", record)
    supervisor._refresh_policy()
    supervisor._refresh_policy()
    assert writes == []
    policy_path = local_paths(cfg.runtime_root)["lease_policy_cache"]
    policy_path.write_text("damaged")
    supervisor._refresh_policy()
    assert writes == [policy_path]
    assert read_json(policy_path)["lease_policy"]["ttl_seconds"] == 120
    process = {"attempt_id": "a", "fencing_token": 1}
    supervisor._set_authority_state(process, "healthy")
    writes.clear()
    supervisor._set_authority_state(process, "healthy")
    assert writes == []
    process["fencing_token"] = 2
    supervisor._set_authority_state(process, "healthy")
    assert len(writes) == 1
    assert read_json(writes[0])["process"]["fencing_token"] == 2


def test_exit_work_progresses_before_a_large_registration_sweep_finishes(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    registrations = local_paths(cfg.runtime_root)["registrations"]
    registrations.mkdir(parents=True, exist_ok=True)
    for number in range(1024):
        atomic_replace(registrations / f"retained-{number}.json", {"process_registration": {"protocol_version": 0}})
    task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    try:
        supervisor.recover_startup()
        supervisor.tick()
        assert not supervisor.work_snapshot["startup_complete"]
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        snapshot = supervisor.work_snapshot
        assert snapshot["lanes"]["registrations"]["entries"] <= 38
        assert sum(lane["entries"] for lane in snapshot["lanes"].values()) + snapshot["cleanup_entries"] <= 64
        assert supervisor.metrics["exit_to_terminal.count"] == 1
        assert supervisor.metrics["terminal_to_accounting.count"] == 1
    finally:
        supervisor.close()


@pytest.mark.parametrize("active_count", [4, 64, 256])
def test_active_service_remains_fair_during_arrivals_and_failed_cleanup(tmp_path, monkeypatch, active_count):
    from collections import defaultdict
    from types import SimpleNamespace

    from qqtools.plugins.qexp import authority_work

    paths = local_paths(tmp_path / "runtime")
    service_turns = defaultdict(list)
    now = [0.0]
    monkeypatch.setattr(authority_work, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def supervise(process):
        service_turns[process["attempt_id"]].append(int(now[0]))

    def failed_cleanup(_attempt_id):
        raise OSError("injected unavailable cleanup")

    supervisor = SimpleNamespace(
        cfg=SimpleNamespace(runtime_root=tmp_path / "runtime", shared_root=tmp_path / "shared"),
        recovery_owner=None,
        metrics={},
        renewal_interval_seconds=1.0,
        _materialize_unverified_intent=lambda _path: None,
        _materialize_registrations=lambda **_kwargs: None,
        _supervise=supervise,
        _remove_terminal_attempt_evidence=failed_cleanup,
        _cleanup_request_for_membership=lambda _entry: None,
        _record_diagnostic=lambda *_args: None,
    )
    work = authority_work.AuthorityWork(supervisor)
    for number in range(active_count):
        attempt_id = f"active-{number}"
        atomic_replace(
            paths["processes"] / f"{attempt_id}.json", {"process": {"protocol_version": 1, "attempt_id": attempt_id}}
        )
        work._remember(attempt_id)
    try:
        for turn in range(24):
            now[0] = float(turn)
            # Arrival rate exceeds registration discovery throughput on every turn.
            for arrival in range(8):
                atomic_replace(paths["registrations"] / f"new-{turn}-{arrival}.json", {})
            work.tick(64)
        assert len(service_turns) == active_count
        for turns in service_turns.values():
            assert turns[0] < 8
            assert 23 - turns[-1] < 8
            assert all(later - earlier <= 8 for earlier, later in zip(turns, turns[1:]))
        snapshot = work.snapshot()
        assert snapshot["cleanup_failures"] > 0
        assert snapshot["lanes"]["registrations"]["processed"] > 0
        assert snapshot["active_cache_size"] <= 256
    finally:
        work.close()


def test_outage_capacity_discovery_is_bounded_and_retains_evidence(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import authority
    from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _task, attempt = _registration(cfg)
    observation, value = _finish(cfg, attempt)
    paths = local_paths(cfg.runtime_root)
    registration = paths["registrations"] / f"{attempt.attempt_id}.json"
    process = read_json(registration)["process_registration"]
    for number in range(64):
        atomic_replace(paths["active"] / f"unrelated-{number}.json", {"reservation": {}})
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    original_read = authority.read_json
    reservation_reads = []

    def measured_read(path):
        if path.parent in {paths["active"], paths["cpu_active"]}:
            reservation_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(authority, "read_json", measured_read)
    try:
        for _ in range(40):
            before = len(reservation_reads)
            diagnostics = RuntimeDiagnostics()
            with activate_diagnostics(diagnostics):
                supervisor._release_finished_local_capacity(process)
            assert len(reservation_reads) - before <= 8
            assert diagnostics.counters["store.iter_json.calls"] == 0
        assert not (paths["active"] / f"{attempt.reservation_id}.json").exists()
        assert len(set(reservation_reads)) == 65
        assert read_json(observation) == value
        assert registration.exists()
        assert all((paths["active"] / f"unrelated-{number}.json").exists() for number in range(64))
    finally:
        supervisor.close()


def test_valid_arrivals_do_not_evict_due_cached_attempts(tmp_path, monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace

    from qqtools.plugins.qexp import authority_work

    paths = local_paths(tmp_path / "runtime")
    service_turns = defaultdict(list)
    now = [0.0]
    monkeypatch.setattr(authority_work, "time", SimpleNamespace(monotonic=lambda: now[0]))
    supervisor = SimpleNamespace(
        cfg=SimpleNamespace(runtime_root=tmp_path / "runtime", shared_root=tmp_path / "shared"),
        recovery_owner=None,
        metrics={},
        renewal_interval_seconds=1.0,
        _materialize_unverified_intent=lambda _path: None,
        _materialize_registrations=lambda **_kwargs: None,
        _supervise=lambda process: service_turns[process["attempt_id"]].append(int(now[0])),
        _remove_terminal_attempt_evidence=lambda _attempt_id: None,
        _cleanup_request_for_membership=lambda _entry: None,
        _record_diagnostic=lambda *_args: None,
    )
    work = authority_work.AuthorityWork(supervisor)

    def publish(attempt_id):
        path = paths["processes"] / f"{attempt_id}.json"
        atomic_replace(path, {"process": {"protocol_version": 1, "attempt_id": attempt_id}})
        return path

    cached = {f"active-{number}" for number in range(256)}
    for attempt_id in sorted(cached):
        publish(attempt_id)
        work._remember(attempt_id)
    try:
        for turn in range(24):
            now[0] = float(turn)
            # Real process records arrive before each slice. Direct discovery also
            # exercises admission independently of filesystem enumeration order.
            for number in range(16):
                work._process(publish(f"arrival-{turn}-{number}"))
            work.tick(64)
            assert set(work._active) == cached
            assert len(work._last_served) <= 256
        for attempt_id in cached:
            turns = service_turns[attempt_id]
            assert turns[0] < 8
            assert 23 - turns[-1] < 8
            assert all(later - earlier <= 8 for earlier, later in zip(turns, turns[1:]))
        assert all(service_turns[f"arrival-{turn}-{number}"] for turn in range(24) for number in range(16))
        assert work.snapshot()["active_admission_deferred"] >= 24 * 16

        # Removal frees a slot; a later discovery may join the rotation.
        removed = next(iter(work._active))
        (paths["processes"] / f"{removed}.json").unlink()
        work._active_step()
        work._process(publish("replacement"))
        assert "replacement" in work._active
        assert removed not in work._active
        assert removed not in work._last_served
        assert len(work._active) == 256
    finally:
        work.close()


def test_capacity_discovery_progresses_across_more_than_256_attempts(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    from qqtools.plugins.qexp import authority

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, initial = _registration(cfg)
    _finish(cfg, initial)
    original_task = load_task(cfg, task.task_id).to_dict()
    paths = local_paths(cfg.runtime_root)
    template = read_json(paths["registrations"] / f"{initial.attempt_id}.json")["process_registration"]
    reservation = read_json(paths["active"] / f"{initial.reservation_id}.json")["reservation"]
    processes = [template]
    retained = set()
    for index in range(260):
        attempt_id = f"attempt-{index}"
        process = dict(template, attempt_id=attempt_id, task_id=f"task-{index}")
        atomic_replace(paths["registrations"] / f"{attempt_id}.json", {"process_registration": process})
        if index != 257:  # No exit evidence must retain capacity.
            _finish(cfg, SimpleNamespace(attempt_id=attempt_id, task_id=process["task_id"]))
        record = deepcopy(reservation)
        record.update(
            reservation_id=f"reservation-{index}",
            acquisition_id=f"acquisition-{index}",
            attempt_id=attempt_id,
            task_id=process["task_id"],
            gpu_ids=[index + 1],
        )
        if index == 258:
            record["project_id"] = "other-project"
        if index == 259:
            record["fencing_token"] = process["fencing_token"] + 1
        atomic_replace(paths["active"] / f"{record['reservation_id']}.json", {"reservation": record})
        if index >= 257:
            retained.add(record["reservation_id"])
        else:
            processes.append(process)

    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    original_read = authority.read_json
    reservation_reads = []

    def measured_read(path):
        if path.parent in {paths["active"], paths["cpu_active"]}:
            reservation_reads.append(path)
        return original_read(path)

    monkeypatch.setattr(authority, "read_json", measured_read)
    monkeypatch.setattr(authority, "load_task", lambda *_args: pytest.fail("local release read shared Task truth"))
    try:
        for process in processes:
            before = len(reservation_reads)
            supervisor._release_finished_local_capacity(process)
            assert len(reservation_reads) - before <= 8
        assert {path.stem for path in paths["active"].glob("*.json")} == retained
        assert len(supervisor._capacity_scans) == 2
        assert load_task(cfg, task.task_id).to_dict() == original_task
        assert all((paths["registrations"] / f"{process['attempt_id']}.json").exists() for process in processes)
    finally:
        supervisor.close()


@pytest.mark.parametrize("is_outage", [False, True])
def test_capacity_discovery_keeps_cpu_progress_when_gpu_scan_fails(tmp_path, monkeypatch, is_outage):
    from qqtools.plugins.qexp import authority

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    paths = local_paths(cfg.runtime_root)
    process = read_json(paths["registrations"] / f"{attempt.attempt_id}.json")["process_registration"]
    gpu_path = paths["active"] / f"{attempt.reservation_id}.json"
    cpu_record = read_json(gpu_path)["reservation"]
    cpu_record.pop("gpu_ids")
    cpu_record.update(reservation_id="cpu-finished", acquisition_id="cpu-acquisition", cpu_slots=1)
    cpu_path = paths["cpu_active"] / "cpu-finished.json"
    atomic_replace(cpu_path, {"reservation": cpu_record})
    for number in range(17):
        atomic_replace(paths["cpu_active"] / f"malformed-{number}.json", {"reservation": {}})
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    original_take = authority.EvidenceScan.take

    def failed_gpu(scan, limit):
        if scan.directory == paths["active"]:
            raise OSError("GPU inventory unavailable")
        return original_take(scan, limit)

    monkeypatch.setattr(authority.EvidenceScan, "take", failed_gpu)
    try:
        for _ in range(8):
            if is_outage:
                supervisor.reconcile_local_exit_evidence(limit=8)
            else:
                supervisor._release_finished_local_capacity(process)
        assert not cpu_path.exists()
        assert gpu_path.exists()
        assert len(list(paths["cpu_active"].glob("*.json"))) == 17
        assert (paths["observations"] / f"{attempt.attempt_id}.json").exists()
    finally:
        supervisor.close()


@pytest.mark.parametrize("invalid", [None, [], "invalid"])
@pytest.mark.parametrize("record_kind", ["reservation", "process", "registration", "registration_with_manifest"])
def test_malformed_capacity_candidate_does_not_stop_healthy_release(tmp_path, invalid, record_kind):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    paths = local_paths(cfg.runtime_root)
    process = read_json(paths["registrations"] / f"{attempt.attempt_id}.json")["process_registration"]
    good_reservation = paths["active"] / f"{attempt.reservation_id}.json"
    bad_process = dict(process, attempt_id="bad-attempt", task_id="bad-task")
    bad_reservation = read_json(good_reservation)["reservation"]
    bad_reservation.update(
        reservation_id="bad-reservation", acquisition_id="bad-acquisition", attempt_id="bad-attempt", task_id="bad-task"
    )
    bad_path = paths["active"] / "bad-reservation.json"
    bad_value = {"reservation": invalid if record_kind == "reservation" else bad_reservation}
    atomic_replace(bad_path, bad_value)
    atomic_replace(
        paths["registrations"] / "bad-attempt.json",
        {"process_registration": invalid if record_kind.startswith("registration") else bad_process},
    )
    atomic_replace(
        paths["observations"] / "bad-attempt.json",
        {"exit_observation": {"attempt_id": "bad-attempt", "task_id": "bad-task", "observed_exit_code": 0}},
    )
    if record_kind in {"process", "registration_with_manifest"}:
        atomic_replace(
            paths["processes"] / "bad-attempt.json", {"process": invalid if record_kind == "process" else bad_process}
        )
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    try:
        for _ in range(4):
            supervisor._release_finished_local_capacity(process)
        assert not good_reservation.exists()
        assert read_json(bad_path) == bad_value
        assert list(paths["authority_diagnostics"].glob("*.json"))
        assert (paths["observations"] / "bad-attempt.json").exists()
    finally:
        supervisor.close()


@pytest.mark.parametrize("invalid", [None, [], "invalid"])
def test_malformed_nested_lane_records_retain_evidence_without_stopping_service(tmp_path, invalid):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    paths = local_paths(cfg.runtime_root)
    records = {
        paths["registrations"] / "broken-registration.json": {"process_registration": invalid},
        paths["processes"] / "broken-process.json": {"process": invalid},
        paths["launch_intents"] / "broken-intent.json": {"launch_intent": invalid},
        paths["termination_decisions"] / "broken-attempt" / "decision.json": {"termination_decision": invalid},
    }
    for path, value in records.items():
        atomic_replace(path, value)
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    try:
        for _ in range(5):
            supervisor.tick()
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert not supervisor.work_snapshot["startup_complete"]
        for lane in ("registrations", "supervision", "intents", "termination"):
            assert supervisor.work_snapshot["lanes"][lane]["failures"] > 0
        for path, value in records.items():
            assert read_json(path) == value
    finally:
        supervisor.close()


def test_launch_intent_materializes_registration_without_waiting_for_inventory(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, attempt = _registration(cfg)
    paths = local_paths(cfg.runtime_root)
    registration = read_json(paths["registrations"] / f"{attempt.attempt_id}.json")["process_registration"]
    intent = paths["launch_intents"] / f"{attempt.attempt_id}.json"
    atomic_replace(intent, {"launch_intent": registration})
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    try:
        assert not (paths["processes"] / intent.name).exists()
        supervisor._materialize_unverified_intent(intent)
        assert (paths["processes"] / intent.name).exists()
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "running"
        assert not supervisor._work.is_startup_complete
    finally:
        supervisor.close()


@pytest.mark.parametrize("limit", [1, 8])
@pytest.mark.parametrize("history_count", [0, 1024])
def test_outage_capacity_discovery_ignores_history_and_serves_both_lanes(tmp_path, monkeypatch, limit, history_count):
    import os

    from qqtools.plugins.qexp import authority
    from qqtools.plugins.qexp.runtime.resources.cpu_lane import attach_cpu, reserve_cpu, set_cpu_lane_capacity

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task, attempt = _registration(cfg)
    _finish(cfg, attempt)
    before = load_task(cfg, task.task_id).to_dict()
    paths = local_paths(cfg.runtime_root)
    set_cpu_lane_capacity(cfg.runtime_root, capacity=1)
    cpu = reserve_cpu(
        cfg.runtime_root,
        task.task_id,
        1,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.current_fencing_token,
    )["reservation"]
    attach_cpu(cfg.runtime_root, cpu["reservation_id"], attempt.attempt_id, attempt.current_fencing_token)
    for number in range(history_count):
        atomic_replace(paths["observations"] / f"settled-{number}.json", {"exit_observation": {}})
    supervisor = AuthoritySupervisor(cfg)
    original_take = authority.EvidenceScan.take
    original_scandir = os.scandir
    visited = []

    def active_only(directory):
        assert Path(directory) in {paths["active"], paths["cpu_active"]}
        return original_scandir(directory)

    def counted_take(scan, budget):
        page = original_take(scan, budget)
        visited.append(page.entries_visited)
        return page

    def unexpected_shared_access(*_args, **_kwargs):
        pytest.fail("outage capacity release accessed shared Task truth")

    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(os, "scandir", active_only)
            guarded.setattr(authority.EvidenceScan, "take", counted_take)
            guarded.setattr(authority, "load_task", unexpected_shared_access)
            for _ in range(2):
                visited.clear()
                supervisor.reconcile_local_exit_evidence(limit=limit)
                assert sum(visited) <= limit
        assert not (paths["active"] / f"{attempt.reservation_id}.json").exists()
        assert not (paths["cpu_active"] / f"{cpu['reservation_id']}.json").exists()
        assert load_task(cfg, task.task_id).to_dict() == before
        assert (paths["registrations"] / f"{attempt.attempt_id}.json").exists()
        assert (paths["observations"] / f"{attempt.attempt_id}.json").exists()
        assert len(list(paths["observations"].glob("*.json"))) == history_count + 1
    finally:
        supervisor.close()


@pytest.mark.parametrize("blocker", ["foreign_project", "fencing", "observation", "live_process", "missing_exit"])
def test_outage_capacity_discovery_preserves_unverified_occupancy(tmp_path, monkeypatch, blocker):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _task, attempt = _registration(cfg)
    observation, value = _finish(cfg, attempt)
    paths = local_paths(cfg.runtime_root)
    reservation_path = paths["active"] / f"{attempt.reservation_id}.json"
    reservation = read_json(reservation_path)
    if blocker == "foreign_project":
        reservation["reservation"]["project_id"] = "other-project"
    elif blocker == "fencing":
        reservation["reservation"]["fencing_token"] += 1
    elif blocker == "observation":
        value["exit_observation"]["attempt_id"] = "other-attempt"
        atomic_replace(observation, value)
    elif blocker == "missing_exit":
        observation.unlink()
    else:
        monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_group_alive", lambda _pid: True)
    atomic_replace(reservation_path, reservation)
    supervisor = AuthoritySupervisor(cfg)
    try:
        supervisor.reconcile_local_exit_evidence(limit=8)
        assert read_json(reservation_path) == reservation
        assert (paths["registrations"] / f"{attempt.attempt_id}.json").exists()
    finally:
        supervisor.close()


@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_outage_capacity_discovery_rejects_projectless_cross_project_collision(tmp_path, lane):
    from dataclasses import replace

    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.runtime.resources.cpu_lane import attach_cpu, reserve_cpu, set_cpu_lane_capacity
    from qqtools.plugins.qexp.runtime.resources.reservations import attach, reserve

    runtime = MachineRuntime(tmp_path / "machine")
    projects = []
    for name in ("owner", "other"):
        cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1", runtime_root=tmp_path / f"{name}-legacy")
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        cfg = replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])
        paths = local_paths(cfg.runtime_root)
        atomic_replace(
            paths["registrations"] / "same-attempt.json",
            {
                "process_registration": {
                    "task_id": "same-task",
                    "attempt_id": "same-attempt",
                    "fencing_token": 1,
                    "process_group_id": 99999992,
                    "process_group_start_time_ticks": 2,
                }
            },
        )
        projects.append(cfg)
    other_paths = local_paths(projects[1].runtime_root)
    atomic_replace(
        other_paths["observations"] / "same-attempt.json",
        {"exit_observation": {"task_id": "same-task", "attempt_id": "same-attempt", "observed_exit_code": 0}},
    )
    if lane == "gpu":
        record = reserve(runtime.root, "same-task", [0])["reservation"]
        attach(runtime.root, record["reservation_id"], "same-attempt", 1)
    else:
        set_cpu_lane_capacity(runtime.root, capacity=1)
        record = reserve_cpu(runtime.root, "same-task", 1, attempt_id="same-attempt", fencing_token=1)["reservation"]
        attach_cpu(runtime.root, record["reservation_id"], "same-attempt", 1)
    active = local_paths(runtime.root)["active" if lane == "gpu" else "cpu_active"] / f"{record['reservation_id']}.json"
    before = read_json(active)
    supervisor = AuthoritySupervisor(projects[1], reservation_runtime_root=runtime.root)
    try:
        supervisor.reconcile_local_exit_evidence(limit=8)
        assert read_json(active) == before
        assert (other_paths["observations"] / "same-attempt.json").exists()
        # Explicit identity still allows the owning partition's verified release.
        before["reservation"]["project_id"] = projects[1].runtime_root.name
        atomic_replace(active, before)
        for _ in range(3):
            supervisor.reconcile_local_exit_evidence(limit=8)
        assert not active.exists()
    finally:
        supervisor.close()
