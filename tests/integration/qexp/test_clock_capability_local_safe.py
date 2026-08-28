from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.lease import ClockCapability, ClockObservation, LeasePolicy, clock_capability
from qqtools.plugins.qexp.runtime import availability as availability_runtime
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _observation(provider: str, lower: float, upper: float) -> ClockObservation:
    return ClockObservation(
        observation_id=f"{provider}-observation",
        provider=provider,
        observed_at="2026-08-06T00:00:00Z",
        monotonic_observed_at=1.0,
        boot_id="test-boot",
        lower_error_seconds=lower,
        upper_error_seconds=upper,
        max_drift_rate=0.0,
        provider_margin_seconds=0.0,
    )


def test_conflicting_provider_intervals_fail_closed(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.lease._provider_observations",
        lambda _policy: ([_observation("chrony", -0.2, 0.2),
                          _observation("linux_adjtimex", 0.5, 0.7)], []),
    )
    monkeypatch.setattr("qqtools.plugins.qexp.lease._boot_id", lambda: "test-boot")
    capability = clock_capability(cfg, LeasePolicy(clock_provider_margin_seconds=0.0))
    assert capability.status == "unhealthy"
    assert capability.reason == "provider_conflict"


def test_unqualified_clock_creates_holder_bound_claim(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.clock_capability",
        lambda *_args: ClockCapability("unavailable", "no_qualified_provider"),
    )
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert attempt.authority_mode == "holder_bound"
    assert attempt.lease["expires_at"] is None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    claim = load_task(cfg, task.task_id).claim_control["active_claim"]
    assert claim["authority_mode"] == "holder_bound"
    assert claim["lease_expires_at"] is None


def test_elapsed_offer_requires_conservative_two_host_proof(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="g", sharing_mode="private")
    start = datetime(2026, 8, 6, tzinfo=timezone.utc)
    creator = ClockObservation(
        observation_id="creator", provider="chrony", observed_at=start.isoformat(),
        monotonic_observed_at=1.0, boot_id="creator-boot", lower_error_seconds=-0.7,
        upper_error_seconds=0.7, max_drift_rate=0.0, provider_margin_seconds=0.0,
    )
    monkeypatch.setattr(availability_runtime, "clock_evidence", lambda _cfg: (creator, start, 2.0))
    task_commands.share(cfg, task.task_id, after_seconds=10)
    reader = ClockObservation(
        observation_id="reader", provider="chrony", observed_at=start.isoformat(),
        monotonic_observed_at=1.0, boot_id="reader-boot", lower_error_seconds=-0.8,
        upper_error_seconds=0.8, max_drift_rate=0.0, provider_margin_seconds=0.0,
    )
    monkeypatch.setattr(
        availability_runtime,
        "clock_evidence",
        lambda _cfg: (reader, start + timedelta(seconds=11), 2.0),
    )
    task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"
    monkeypatch.setattr(
        availability_runtime,
        "clock_evidence",
        lambda _cfg: (reader, start + timedelta(seconds=12), 2.0),
    )
    task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_elapsed_offer_ages_creator_evidence_through_long_delay(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="g", sharing_mode="private")
    start = datetime(2026, 8, 6, tzinfo=timezone.utc)
    creator = ClockObservation(
        observation_id="creator", provider="chrony", observed_at=start.isoformat(),
        monotonic_observed_at=1.0, boot_id="creator-boot", lower_error_seconds=-0.1,
        upper_error_seconds=0.1, max_drift_rate=0.1, provider_margin_seconds=0.0,
    )
    monkeypatch.setattr(availability_runtime, "clock_evidence", lambda _cfg: (creator, start, 2.0))
    task_commands.share(cfg, task.task_id, after_seconds=3600)
    proof = load_task(cfg, task.task_id).placement_runtime["offer_clock_evidence"]
    assert proof["creator_observation"]["max_drift_rate"] == 0.1
    assert proof["deadline_monotonic_at"] == 3602.0

    reader = ClockObservation(
        observation_id="reader", provider="chrony", observed_at=start.isoformat(),
        monotonic_observed_at=1.0, boot_id="reader-boot", lower_error_seconds=-0.1,
        upper_error_seconds=0.1, max_drift_rate=0.0, provider_margin_seconds=0.0,
    )
    monkeypatch.setattr(
        availability_runtime,
        "clock_evidence",
        lambda _cfg: (reader, start + timedelta(seconds=3601), 2.0),
    )
    task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"

    monkeypatch.setattr(
        availability_runtime,
        "clock_evidence",
        lambda _cfg: (reader, start + timedelta(seconds=3961), 2.0),
    )
    task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_submission_persists_raw_creator_observation_for_timed_offer(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    observation = _observation("chrony", -0.2, 0.2)
    monkeypatch.setattr(
        submission_runtime,
        "clock_capability",
        lambda _cfg: ClockCapability("healthy", "healthy", observation, ("chrony",)),
    )
    task = submit(cfg, ["echo", "ok"], group="g", sharing_mode="spillover", offer_after_seconds=3600)
    runtime = load_task(cfg, task.task_id).placement_runtime
    proof = runtime["offer_clock_evidence"]
    assert proof["creator_observation"] == observation.to_dict()
    assert proof["deadline_monotonic_at"] >= observation.monotonic_observed_at + 3600
