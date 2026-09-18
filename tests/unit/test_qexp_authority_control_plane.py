"""Deterministic control-plane scheduling tests; no daemon or training process."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import control_plane
from qqtools.plugins.qexp.agent.deadlines import advance_deadline


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.stopped = False

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        return self.stopped

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True


@pytest.fixture
def plane_factory(monkeypatch, tmp_path):
    def factory(project_ids=("a", "b", "c"), *, interval=0.25):
        calls = []
        created = []
        clock = FakeClock()
        bindings = [
            SimpleNamespace(
                project_id=name,
                enabled=True,
                registration_generation="generation-1",
                shared_root=tmp_path / name / ".qexp",
                machine_name="machine-1",
            )
            for name in project_ids
        ]
        runtime = SimpleNamespace(
            root=tmp_path / "machine",
            registration_status=lambda binding: {"state": "eligible"},
            binding_write_eligible=lambda binding, **kwargs: True,
            reactivate_binding=lambda binding: True,
            project_paths=lambda project_id: {"root": tmp_path / project_id},
        )

        class Supervisor:
            renewal_interval_seconds = 0.25
            work_snapshot = {"startup_complete": True}

            def close(self):
                pass

            def cancel_pending_control(self):
                self.recovery_cancelled = True

            def __init__(self, cfg, **kwargs):
                self.cfg = cfg
                self.work_limit = kwargs.get("work_limit")
                created.append(self)

            def recover_startup(self):
                calls.append((self.cfg.runtime_root.name, "startup"))

            def tick(self):
                calls.append((self.cfg.runtime_root.name, "tick"))

            def reconcile_local_exit_evidence(self, **kwargs):
                calls.append((self.cfg.runtime_root.name, "local_exit"))

        monkeypatch.setattr(control_plane, "time", SimpleNamespace(monotonic=clock.monotonic))
        monkeypatch.setattr(control_plane, "ProgressObservationLoop", lambda runtime: None)
        monkeypatch.setattr(
            control_plane,
            "load_lease_policy",
            lambda cfg: SimpleNamespace(ttl_seconds=120, renew_interval_seconds=10),
        )
        monkeypatch.setattr(control_plane, "_AuthoritySupervisor", Supervisor)
        monkeypatch.setattr(
            control_plane._helpers,
            "_binding_config",
            lambda runtime, binding: SimpleNamespace(runtime_root=tmp_path / binding.project_id),
        )
        plane = control_plane._MachineControlPlane(
            runtime,
            instance_id="test-instance",
            loop_interval=interval,
            started_at="2026-09-17T00:00:00+00:00",
            available_gpus=[],
        )
        monkeypatch.setattr(plane, "_supervised_bindings", lambda: bindings)
        monkeypatch.setattr(plane, "_reserved_gpu_ids", lambda: set())
        return SimpleNamespace(
            plane=plane, bindings=bindings, runtime=runtime, calls=calls, created=created, clock=clock
        )

    return factory


@pytest.mark.parametrize(
    ("scheduled", "period", "finished", "expected", "skipped"),
    [
        (0.0, 1.0, 0.2, 1.0, 0),
        (0.0, 1.0, 1.0, 2.0, 1),
        (0.0, 0.25, 0.6, 0.75, 2),
        (1.0, 0.1, 1.05, 1.1, 0),
        (0.0, 1.0, 1_000_000.1, 1_000_001.0, 1_000_000),
    ],
)
def test_deadline_skips_missed_slots(scheduled, period, finished, expected, skipped):
    actual, count = advance_deadline(scheduled, period, finished)
    assert actual == pytest.approx(expected)
    assert count == skipped
    assert actual > finished


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_period_is_rejected(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        advance_deadline(0.0, interval, 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        control_plane._MachineControlPlane(
            SimpleNamespace(), instance_id="test", loop_interval=interval, started_at="unused", available_gpus=[]
        )


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_invalid_monotonic_timestamp_is_rejected(timestamp):
    with pytest.raises(ValueError, match="timestamps"):
        advance_deadline(timestamp, 1.0, 0.0)
    with pytest.raises(ValueError, match="timestamps"):
        advance_deadline(0.0, 1.0, timestamp)


@pytest.mark.parametrize(
    ("work", "starts", "waits"),
    [(0.1, [0, 0.25, 0.5], [0.15, 0.15]), (0.6, [0, 0.75, 1.5], [0.15, 0.15])],
)
def test_authority_deadlines_are_anchored_and_overruns_are_paced(plane_factory, monkeypatch, work, starts, waits):
    case = plane_factory(project_ids=())
    observed_starts = []
    case.plane._stop_event = case.clock

    def cycle():
        observed_starts.append(case.clock.now)
        case.clock.now += work
        if len(observed_starts) == 3:
            case.clock.set()
        return 0.25

    monkeypatch.setattr(case.plane, "_run_authority_cycle", cycle)
    case.plane._run_authority_loop()
    assert observed_starts == pytest.approx(starts)
    assert case.clock.waits == pytest.approx(waits)
    timing = case.plane._authority_snapshot["schedule"]
    assert timing["cycle_seconds"] == pytest.approx(work)
    assert timing["observed_wait_seconds"] == pytest.approx(waits[-1])
    assert timing["skipped_intervals_total"] == (6 if work == 0.6 else 0)


def test_interval_change_takes_effect_on_next_deadline(plane_factory, monkeypatch):
    case = plane_factory(project_ids=())
    case.plane._stop_event = case.clock
    periods = iter([1.0, 0.1, 0.1])
    starts = []

    def cycle():
        starts.append(case.clock.now)
        case.clock.now += 0.01
        if len(starts) == 3:
            case.clock.set()
        return next(periods)

    monkeypatch.setattr(case.plane, "_run_authority_cycle", cycle)
    case.plane._run_authority_loop()
    assert starts == pytest.approx([0.0, 1.0, 1.1])


def test_stop_during_wait_does_not_start_a_cycle(plane_factory, monkeypatch):
    case = plane_factory(project_ids=())
    case.plane._stop_event = case.clock
    calls = []
    monkeypatch.setattr(case.clock, "wait", lambda seconds: True)
    case.plane._run_deadline_loop(lambda: calls.append("ran"), initial_delay=1.0)
    assert calls == []


def test_heartbeat_uses_same_overrun_arithmetic(plane_factory):
    case = plane_factory(project_ids=())
    case.plane._stop_event = case.clock
    starts = []

    def operation():
        starts.append(case.clock.now)
        case.clock.now += 0.6
        if len(starts) == 3:
            case.clock.set()

    case.plane._run_deadline_loop(operation)
    assert starts == pytest.approx([0.0, 0.75, 1.5])
    assert case.clock.waits == pytest.approx([0.15, 0.15])


def test_each_project_leads_once_and_inventory_check_is_eliminated(plane_factory, monkeypatch):
    case = plane_factory()

    def forbidden_inventory_scan(*args):
        raise AssertionError("eligible binding must not perform the discarded inventory query")

    monkeypatch.setattr(case.plane, "_has_local_process", forbidden_inventory_scan)
    orders = []
    for _ in range(3):
        case.plane._run_authority_cycle()
        orders.append([item["project_id"] for item in case.plane._authority_snapshot["projects"]])
    assert orders == [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
    assert len(case.created) == 3
    assert sum(action == "tick" for _, action in case.calls) == 9
    assert all(item["eligibility_inventory_checks"] == 0 for item in case.plane._authority_snapshot["projects"])


def test_rotation_survives_removal_empty_registry_and_new_binding(plane_factory):
    case = plane_factory()
    case.plane._run_authority_cycle()
    case.bindings[:] = [case.bindings[0], case.bindings[2]]
    case.plane._run_authority_cycle()
    assert set(case.plane._supervisors) == {"a", "c"}
    assert set(case.plane._supervisor_generations) == {"a", "c"}
    added = copy.copy(case.bindings[0])
    added.project_id = "new"
    case.bindings.clear()
    case.plane._run_authority_cycle()
    assert case.plane._next_authority_project is None
    assert case.plane._supervisors == {}
    case.bindings.append(added)
    case.plane._run_authority_cycle()
    assert [item["project_id"] for item in case.plane._authority_snapshot["projects"]] == ["new"]


def test_generation_change_reconstructs_supervisor(plane_factory):
    case = plane_factory(project_ids=("a",))
    case.plane._run_authority_cycle()
    first = case.plane._supervisors["a"]
    case.bindings[0].registration_generation = "generation-2"
    case.plane._run_authority_cycle()
    assert case.plane._supervisors["a"] is not first
    assert case.plane._supervisor_generations["a"] == "generation-2"
    assert case.calls == [("a", "startup"), ("a", "tick"), ("a", "startup"), ("a", "tick")]


@pytest.mark.parametrize(("has_process", "reactivated"), [(False, False), (True, False), (True, True)])
def test_ineligible_binding_retains_local_process_and_reactivation_checks(
    plane_factory, monkeypatch, has_process, reactivated
):
    case = plane_factory(project_ids=("a",))
    case.runtime.binding_write_eligible = lambda binding, **kwargs: False
    case.runtime.reactivate_binding = lambda binding: reactivated
    scans = []

    def inventory(path):
        scans.append(path)
        return [Path("process.json")] if has_process else []

    monkeypatch.setattr(case.plane, "_has_local_process", lambda binding, cfg: bool(inventory(cfg.runtime_root)))
    case.plane._run_authority_cycle()
    assert len(scans) == 1
    assert (("a", "tick") in case.calls) is (has_process and reactivated)
    assert (("a", "startup") in case.calls) is has_process
    assert (("a", "local_exit") in case.calls) is (not has_process or not reactivated)
    assert case.plane._authority_snapshot["projects"][0]["eligibility_inventory_checks"] == 1


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_one_project_failure_does_not_skip_remaining_projects(plane_factory, monkeypatch, error_type):
    case = plane_factory()
    original = control_plane._helpers._binding_config

    def config(runtime, binding):
        if binding.project_id == "a":
            case.clock.now += 0.5
            raise error_type("injected")
        return original(runtime, binding)

    monkeypatch.setattr(control_plane._helpers, "_binding_config", config)
    case.plane._run_authority_cycle()
    assert ("b", "tick") in case.calls and ("c", "tick") in case.calls
    sample = case.plane._authority_snapshot["projects"][0]
    assert sample["error_type"] == error_type.__name__
    assert sample["phase_seconds"]["configuration"] == pytest.approx(0.5)
    if error_type is OSError:
        assert ("a", "local_exit") in case.calls
        assert sample["local_reconciliation"] == "returned"


def test_failed_local_reconciliation_keeps_supervising_other_projects(plane_factory, monkeypatch):
    case = plane_factory()
    original = control_plane._helpers._binding_config

    def config(runtime, binding):
        if binding.project_id == "a":
            raise OSError("shared store unavailable")
        return original(runtime, binding)

    def unavailable(self, **kwargs):
        raise OSError("local evidence unavailable")

    monkeypatch.setattr(control_plane._helpers, "_binding_config", config)
    monkeypatch.setattr(control_plane._AuthoritySupervisor, "reconcile_local_exit_evidence", unavailable)
    case.plane._run_authority_cycle()
    assert case.plane._authority_snapshot["projects"][0]["local_reconciliation"] == "unavailable"
    assert ("b", "tick") in case.calls and ("c", "tick") in case.calls


def test_unavailable_registry_is_unknown_not_an_empty_success(plane_factory, monkeypatch):
    case = plane_factory()
    case.plane._run_authority_cycle()
    existing = dict(case.plane._supervisors)

    def unavailable():
        raise OSError("registry unavailable")

    monkeypatch.setattr(case.plane, "_supervised_bindings", unavailable)
    case.plane._run_authority_cycle()
    sample = case.plane._authority_snapshot
    assert sample["observation_status"] == "registry_unavailable"
    assert sample["project_count"] is None
    assert case.plane._supervisors == existing


def test_reservation_release_still_wakes_dispatch(plane_factory, monkeypatch):
    case = plane_factory(project_ids=())
    values = iter([{1, 2}, {1}])
    wakeups = []
    monkeypatch.setattr(case.plane, "_reserved_gpu_ids", lambda: next(values))
    case.plane._scheduler_wakeup = SimpleNamespace(set=lambda: wakeups.append(True))
    case.plane._run_authority_cycle()
    assert wakeups == [True]


def test_completed_snapshots_are_not_mutated_by_later_cycles(plane_factory):
    case = plane_factory()
    case.plane._run_authority_cycle()
    previous = case.plane._authority_snapshot
    saved = copy.deepcopy(previous)
    case.plane._run_authority_cycle()
    assert previous == saved
    assert case.plane._authority_snapshot is not previous
    assert previous["diagnostic_only"] is True
    assert previous["instance_id"] == "test-instance"


def test_diagnostic_writes_are_local_throttled_and_best_effort(plane_factory, monkeypatch):
    case = plane_factory(project_ids=())
    writes = []

    def write(path, value):
        writes.append((path, copy.deepcopy(value)))
        if len(writes) > 1:
            raise OSError("injected local storage error")

    monkeypatch.setattr(control_plane, "atomic_replace", write)
    case.plane._publish_authority_diagnostics()
    assert writes == []
    case.plane._run_authority_cycle()
    case.plane._publish_authority_diagnostics()
    case.clock.now = 0.9
    case.plane._publish_authority_diagnostics()
    assert len(writes) == 1
    case.clock.now = 1.0
    case.plane._publish_authority_diagnostics()
    case.plane._publish_authority_diagnostics()
    assert len(writes) == 2
    assert writes[0][0] == case.runtime.root / "authority_control_plane.json"
    assert writes[0][1]["authority_control_plane"]["diagnostic_only"] is True


def test_diagnostics_publish_even_when_shared_heartbeat_registry_is_unreadable(plane_factory, monkeypatch):
    case = plane_factory(project_ids=())
    writes = []
    case.plane._run_authority_cycle()
    monkeypatch.setattr(control_plane, "atomic_replace", lambda path, value: writes.append(path))

    def unavailable():
        raise OSError("unreadable registry")

    monkeypatch.setattr(case.plane, "_supervised_bindings", unavailable)
    case.plane._publish_heartbeat()
    assert writes == [case.runtime.root / "authority_control_plane.json"]


@pytest.mark.parametrize("project_count", [1, 4, 16])
def test_registry_size_does_not_drop_a_project(plane_factory, project_count):
    names = tuple(f"project-{index}" for index in range(project_count))
    case = plane_factory(project_ids=names)
    for _ in range(2):
        case.plane._run_authority_cycle()
        observed = case.plane._authority_snapshot["projects"]
        assert len(observed) == project_count
        assert {item["project_id"] for item in observed} == set(names)
    assert sum(action == "tick" for _, action in case.calls) == 2 * project_count


def test_eligibility_is_rechecked_for_a_cached_supervisor(plane_factory, monkeypatch):
    case = plane_factory(project_ids=("a",))
    eligibility = iter([True, False])
    checks = []

    def eligible(binding, *, renew):
        checks.append((binding.project_id, renew))
        return next(eligibility)

    case.runtime.binding_write_eligible = eligible
    monkeypatch.setattr(case.plane, "_has_local_process", lambda *args: False)
    case.plane._run_authority_cycle()
    case.plane._run_authority_cycle()
    assert checks == [("a", True), ("a", True)]
    assert case.calls == [("a", "startup"), ("a", "tick"), ("a", "local_exit")]
    assert case.plane._authority_snapshot["projects"][0]["observation_status"] == "ineligible_no_local_process"


@pytest.mark.parametrize("heartbeat_fails", [False, True])
def test_heartbeat_publication_precedes_optional_diagnostics(plane_factory, monkeypatch, heartbeat_fails):
    case = plane_factory(project_ids=())
    order = []

    def publish():
        order.append("heartbeat")
        if heartbeat_fails:
            raise OSError("injected heartbeat failure")

    monkeypatch.setattr(case.plane, "_publish_project_heartbeats", publish)
    monkeypatch.setattr(case.plane, "_publish_authority_diagnostics", lambda: order.append("diagnostics"))
    if heartbeat_fails:
        with pytest.raises(OSError, match="injected heartbeat failure"):
            case.plane._publish_heartbeat()
    else:
        case.plane._publish_heartbeat()
    assert order == ["heartbeat", "diagnostics"]


def test_large_registry_limits_shared_work_and_reaches_every_project(plane_factory):
    case = plane_factory(project_ids=tuple(f"p-{index}" for index in range(35)))
    statuses = []
    case.runtime.registration_status = lambda binding: statuses.append(binding.project_id) or {"state": "eligible"}
    visited = set()
    for _ in range(3):
        before = len(statuses)
        case.plane._run_authority_cycle()
        assert len(statuses) - before <= 16
        projects = case.plane._authority_snapshot["projects"]
        assert len(projects) <= 16
        visited.update(item["project_id"] for item in projects)
    assert visited == {binding.project_id for binding in case.bindings}


def test_registry_discovery_never_reads_shared_status(plane_factory):
    case = plane_factory()
    case.runtime.load_registry = lambda: (1, case.bindings)

    def shared_read(_binding):
        raise AssertionError("registry discovery performed project I/O")

    case.runtime.registration_status = shared_read
    assert control_plane._MachineControlPlane._supervised_bindings(case.plane) == case.bindings


def test_startup_admission_tracks_successful_recovery_and_generation(plane_factory, monkeypatch):
    case = plane_factory(project_ids=("a",))
    monkeypatch.setattr(control_plane._AuthoritySupervisor, "work_snapshot", {"startup_complete": False})
    case.plane._run_authority_cycle()
    assert case.runtime.authority_ready_generations == {}
    assert case.plane._supervisors["a"].work_limit == 256
    case.plane._supervisors["a"].work_snapshot = {"startup_complete": True}
    case.plane._run_authority_cycle()
    assert case.runtime.authority_ready_generations == {"a": "generation-1"}
    assert case.plane._supervisors["a"].work_limit == 64
    case.plane._run_authority_cycle()
    assert case.plane._authority_snapshot["projects"][0]["step_limit"] == 64
    case.bindings[0].registration_generation = "generation-2"
    case.plane._run_authority_cycle()
    assert case.runtime.authority_ready_generations == {}
    assert case.plane._supervisors["a"].work_limit == 256


def test_project_service_gaps_include_failed_storage_and_reset_on_generation(plane_factory):
    case = plane_factory(("a", "b"))
    case.plane._run_authority_cycle()
    assert all(project["service_gap_seconds"] is None for project in case.plane._authority_snapshot["projects"])
    case.clock.now = 2.0

    def unavailable(binding):
        case.clock.now += 3.0
        raise OSError("returning storage failure")

    case.runtime.registration_status = unavailable
    case.plane._run_authority_cycle()
    projects = {project["project_id"]: project for project in case.plane._authority_snapshot["projects"]}
    assert projects["b"]["service_gap_seconds"] == 2.0
    assert projects["a"]["service_gap_seconds"] == 5.0
    assert projects["a"]["maximum_service_gap_seconds"] == 5.0
    assert projects["a"]["observation_status"] == "storage_error"
    case.runtime.registration_status = lambda binding: {"state": "eligible"}
    case.bindings[0].registration_generation = "generation-2"
    case.plane._run_authority_cycle()
    projects = {project["project_id"]: project for project in case.plane._authority_snapshot["projects"]}
    assert projects["a"]["registration_generation"] == "generation-2"
    assert projects["a"]["service_gap_seconds"] is None
    assert projects["a"]["maximum_service_gap_seconds"] == 0.0
    assert projects["b"]["service_gap_seconds"] == 6.0
    assert projects["b"]["maximum_service_gap_seconds"] == 6.0
    case.bindings.clear()
    case.plane._run_authority_cycle()
    assert case.plane._project_service_times == {}


def test_eligibility_diagnostics_are_exposed_per_project(plane_factory):
    from qqtools.plugins.qexp.runtime.work_budget import diagnostic_increment, diagnostic_observe_ns

    case = plane_factory(("a", "b"))

    def renew(binding, **_kwargs):
        diagnostic_increment("registration.renewal")
        diagnostic_observe_ns("registration.renewal_lateness", 10 if binding.project_id == "a" else 20)
        return True

    case.runtime.binding_write_eligible = renew
    case.plane._run_authority_cycle()
    projects = {project["project_id"]: project for project in case.plane._authority_snapshot["projects"]}
    for project_id, expected in (("a", 10), ("b", 20)):
        diagnostics = projects[project_id]["eligibility_operations"]
        assert diagnostics["counters"]["registration.renewal"] == 1
        assert diagnostics["timings"]["registration.renewal_lateness"]["maximum_ns"] == expected


@pytest.mark.parametrize("publication_fails", [False, True])
def test_heartbeat_collects_both_renewal_boundaries_and_replaces_failed_samples(
    plane_factory, monkeypatch, publication_fails
):
    from contextlib import contextmanager

    from qqtools.plugins.qexp.runtime.work_budget import diagnostic_increment, diagnostic_observe_ns

    case = plane_factory(("a", "b"))
    case.plane._run_authority_cycle()
    authority_sample = copy.deepcopy(case.plane._authority_snapshot)
    writes = []
    monkeypatch.setattr(control_plane, "atomic_replace", lambda path, value: writes.append(copy.deepcopy(value)))
    monkeypatch.setattr(control_plane, "_read_pid", lambda _runtime: 123)
    monkeypatch.setattr(control_plane, "reservation_snapshot", lambda _root: SimpleNamespace(reservations=[]))

    def renew(binding, **_kwargs):
        diagnostic_increment("registration.renewal")
        diagnostic_observe_ns("registration.renewal_lateness", 100)
        return True

    @contextmanager
    def guard(binding):
        diagnostic_increment("registration.renewal")
        diagnostic_observe_ns("registration.renewal_lateness", 200)
        yield True

    def publish(configs, *, write_guard, **_kwargs):
        for project_id in configs:
            with write_guard(project_id) as is_eligible:
                assert is_eligible
        if publication_fails:
            raise OSError("heartbeat publication failed")

    case.runtime.binding_write_eligible = renew
    case.runtime.binding_write_guard = guard
    monkeypatch.setattr(control_plane, "_publish_project_snapshots", publish)
    case.plane._publish_heartbeat()
    heartbeat = writes[0]["authority_control_plane"]["heartbeat"]
    assert heartbeat["sequence"] == 1
    assert heartbeat["observation_status"] == ("publication_unavailable" if publication_fails else "returned")
    operations = heartbeat["operations"]
    assert operations["counters"]["registration.renewal"] == 4
    assert operations["counters"]["registration.renewal_lateness.observations"] == 4
    assert operations["timings"]["registration.renewal_lateness"] == {"total_ns": 600, "maximum_ns": 200}
    assert case.plane._authority_snapshot == authority_sample

    def unavailable():
        raise OSError("registry unavailable")

    monkeypatch.setattr(case.plane, "_supervised_bindings", unavailable)
    case.clock.now = 1.0
    case.plane._publish_heartbeat()
    failed = writes[1]["authority_control_plane"]["heartbeat"]
    assert failed["sequence"] == 2
    assert failed["observation_status"] == "registry_unavailable"
    assert failed["operations"]["counters"] == {}
    assert heartbeat["operations"]["counters"]["registration.renewal"] == 4


@pytest.mark.parametrize("failure", ["configuration", "eligibility", "registry"])
def test_unserviced_binding_releases_pending_recovery(plane_factory, monkeypatch, failure):
    case = plane_factory(project_ids=("a",))
    case.plane._run_authority_cycle()
    supervisor = case.plane._supervisors["a"]

    def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    if failure == "configuration":
        monkeypatch.setattr(control_plane._helpers, "_binding_config", unavailable)
    elif failure == "eligibility":
        case.runtime.binding_write_eligible = lambda *_args, **_kwargs: False
        case.runtime.reactivate_binding = lambda *_args: False
        monkeypatch.setattr(case.plane, "_has_local_process", lambda *_args: True)
    else:
        monkeypatch.setattr(case.plane, "_supervised_bindings", unavailable)
    case.plane._run_authority_cycle()
    assert supervisor.recovery_cancelled


@pytest.mark.parametrize(
    "error",
    [
        OSError("offline"),
        ValueError("invalid policy"),
        TypeError("invalid fields"),
        KeyError("missing field"),
        AttributeError("invalid shape"),
    ],
)
def test_cadence_policy_read_failure_preserves_existing_supervision(plane_factory, monkeypatch, error):
    fixture = plane_factory(project_ids=("a",))

    def fail_policy(_cfg):
        raise error

    monkeypatch.setattr(control_plane, "load_lease_policy", fail_policy)
    fixture.plane._run_authority_cycle()
    assert ("a", "tick") in fixture.calls
