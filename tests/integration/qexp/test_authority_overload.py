"""Exercise real lease transactions while cooperative discovery is overloaded."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import authority, authority_work, init_shared_root, lease, scheduler, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.lease import ClockCapability
from qqtools.plugins.qexp.runtime import records
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize(
    "active_count,work_limit,arrivals_per_turn",
    [
        pytest.param(16, 12, 2, id="bounded-backlog"),
        pytest.param(256, 64, 8, id="sustained-overload", marks=pytest.mark.stress),
    ],
)
def test_mixed_leases_survive_saturated_discovery_and_fail_closed_without_clock(
    tmp_path, monkeypatch, active_count, work_limit, arrivals_per_turn
):
    now = [datetime.now(timezone.utc).replace(microsecond=0)]
    elapsed = [0.0]

    class ModelDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(authority, "datetime", ModelDateTime)
    monkeypatch.setattr(lease, "datetime", ModelDateTime)
    monkeypatch.setattr(records, "datetime", ModelDateTime)
    monkeypatch.setattr(authority_work, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    paths = local_paths(cfg.runtime_root)
    supervisor = AuthoritySupervisor(cfg, work_limit=work_limit)
    supervisor.recover_startup()
    monkeypatch.setattr(supervisor._work, "_active_limit", active_count)
    attempts = []
    healthy_capability = scheduler.clock_capability
    unavailable = ClockCapability("unavailable", "injected_stale_clock")
    for number in range(active_count):
        mode = "bounded_lease" if number % 2 == 0 else "holder_bound"
        monkeypatch.setattr(
            scheduler, "clock_capability", healthy_capability if number % 2 == 0 else lambda *_args: unavailable
        )
        task = submit(cfg, ["echo", "overload"])
        attempt = scheduler.claim_task(cfg, task.task_id, [number])
        assert attempt is not None and attempt.authority_mode == mode
        assert scheduler.authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        process = {
            "protocol_version": 1,
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "fencing_token": attempt.current_fencing_token,
            "authority_mode": mode,
            "observed_state": "running",
            "lease_expires_at": claim.get("lease_expires_at"),
            "clock_error_bound_seconds": claim.get("clock_error_bound_seconds"),
        }
        atomic_replace(paths["processes"] / f"{attempt.attempt_id}.json", {"process": process})
        supervisor._work._remember(attempt.attempt_id)
        attempts.append(attempt)
    monkeypatch.setattr(scheduler, "clock_capability", healthy_capability)

    def failed_cleanup(_attempt_id):
        raise OSError("injected unavailable maintenance")

    monkeypatch.setattr(supervisor, "_remove_terminal_attempt_evidence", failed_cleanup)
    publications = {attempt.attempt_id: [] for attempt in attempts}
    previous_expiries = {}
    arrivals = []

    def observe_leases():
        for attempt in attempts:
            stored = read_json(attempt_path(cfg.shared_root, attempt.task_id, 1))["attempt"]
            assert stored["current_fencing_token"] == attempt.current_fencing_token
            assert stored["authority_mode"] == attempt.authority_mode
            if attempt.authority_mode == "holder_bound":
                assert stored["lease"]["expires_at"] is None
                assert stored["lease"]["clock_evidence"] is None
                continue
            expiry = stored["lease"]["expires_at"]
            assert datetime.fromisoformat(expiry.replace("Z", "+00:00")) - now[0] > timedelta(seconds=100)
            if expiry != previous_expiries.get(attempt.attempt_id):
                publications[attempt.attempt_id].append(elapsed[0])
                previous_expiries[attempt.attempt_id] = expiry

    try:
        observe_leases()
        for turn in range(1, 41):
            elapsed[0] = float(turn)
            now[0] += timedelta(seconds=1)
            # Arrivals exceed discovery's share of the selected per-tick work budget.
            for arrival in range(arrivals_per_turn):
                task = submit(cfg, ["echo", "arrival"])
                attempt = scheduler.claim_task(
                    cfg, task.task_id, [active_count + (turn - 1) * arrivals_per_turn + arrival]
                )
                assert attempt is not None
                assert scheduler.authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
                claim = load_task(cfg, task.task_id).claim_control["active_claim"]
                registration = {
                    "protocol_version": 1,
                    "task_id": task.task_id,
                    "attempt_id": attempt.attempt_id,
                    "fencing_token": attempt.current_fencing_token,
                    "machine_name": cfg.machine_name,
                    "authority_mode": attempt.authority_mode,
                    "process_created_at": records.utc_now(),
                    "lease_expires_at": claim.get("lease_expires_at"),
                    "clock_error_bound_seconds": claim.get("clock_error_bound_seconds"),
                }
                atomic_replace(
                    paths["registrations"] / f"{attempt.attempt_id}.json",
                    {"process_registration": registration},
                )
                arrivals.append(attempt)
            supervisor.tick()
            observe_leases()
        for attempt in attempts:
            if attempt.authority_mode == "bounded_lease":
                times = publications[attempt.attempt_id]
                assert len(times) >= 3
                assert all(later - earlier <= 18 for earlier, later in zip(times, times[1:])), (
                    attempt.attempt_id,
                    times,
                    supervisor.work_snapshot,
                )
                assert elapsed[0] - times[-1] <= 18
        assert supervisor.metrics["renewal.renewed"] > 0
        assert supervisor.metrics["renewal.not_required"] > 0
        assert supervisor.work_snapshot["active_cache_size"] == active_count
        assert supervisor.work_snapshot["cleanup_failures"] > 0
        assert supervisor.work_snapshot["active_admission_deferred"] > 0
        materialized = 0
        for attempt in arrivals:
            manifest = paths["processes"] / f"{attempt.attempt_id}.json"
            if manifest.exists():
                process = read_json(manifest)["process"]
                assert process["task_id"] == attempt.task_id
                assert process["fencing_token"] == attempt.current_fencing_token
                stored = read_json(attempt_path(cfg.shared_root, attempt.task_id, 1))["attempt"]
                assert stored["phase"] == "running"
                assert stored["current_fencing_token"] == attempt.current_fencing_token
                materialized += 1
        assert 0 < materialized < len(arrivals), "valid arrivals must progress while a discovery backlog remains"

        # Exceed the safe deadline with stale clock evidence. Actual renewal must
        # reject publication; local isolation is not a terminal or fencing change.
        before = {
            attempt.attempt_id: read_json(attempt_path(cfg.shared_root, attempt.task_id, 1)) for attempt in attempts
        }
        monkeypatch.setattr(scheduler, "clock_capability", lambda *_args: unavailable)
        elapsed[0] += 130
        now[0] += timedelta(seconds=130)
        for _ in range(8):
            supervisor.tick()
            elapsed[0] += 1
            now[0] += timedelta(seconds=1)
        for attempt in attempts:
            assert read_json(attempt_path(cfg.shared_root, attempt.task_id, 1)) == before[attempt.attempt_id]
            process = read_json(paths["processes"] / f"{attempt.attempt_id}.json")["process"]
            assert process["authority_state"] == (
                "isolated" if attempt.authority_mode == "bounded_lease" else "local_safe"
            )
    finally:
        supervisor.close()
