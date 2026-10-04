from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.layout import load_machine_registration, machine_state_path, save_machine_registration
from qqtools.plugins.qexp.lease import LeasePolicy, save_lease_policy
from qqtools.plugins.qexp.runtime import registration_authority
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import local_paths, task_path
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.version import __version__

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("ownership", ["current", "superseded", "other_generation"])
def test_explicit_reactivation_repairs_invalid_expiry_only_for_current_owner(tmp_path, ownership):
    runtime, cfg, binding, _revision, executor = _case(tmp_path)
    try:
        record = load_machine_registration(cfg)
        record["registration"]["eligibility_expires_at"] = "unreadable-date"
        if ownership == "superseded":
            record["registration"]["state"] = "superseded"
        elif ownership == "other_generation":
            record["registration"]["generation"] = "other-generation"
        save_machine_registration(cfg, record)
        before = load_machine_registration(cfg)
        assert runtime.reactivate_binding(binding) is (ownership == "current")
        after = load_machine_registration(cfg)
        if ownership == "current":
            assert after["registration"]["generation"] == before["registration"]["generation"]
            assert datetime.fromisoformat(
                after["registration"]["eligibility_expires_at"].replace("Z", "+00:00")
            ) > datetime.now(timezone.utc)
        else:
            assert after == before
    finally:
        executor.shutdown()


def _case(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    return runtime, cfg, binding, revision, executor


@pytest.mark.parametrize(
    "operation",
    ["registration_renew", "machine_snapshot_publish", "machine_stop_publish", "activation_consumer_register"],
)
def test_shared_worker_completes_while_controller_holds_local_locks(tmp_path: Path, operation: str):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    try:
        if operation == "registration_renew":
            request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=120.0)
        elif operation == "activation_consumer_register":
            request = executor.prepare_activation_consumer_register(binding, revision, process_fence="agent-a")
        else:
            request = executor.prepare_machine_snapshot_publish(
                binding,
                revision,
                instance_id="agent-a",
                pid=123,
                visible_gpu_ids=[0],
                reserved_gpu_ids=[],
                reservation_summaries=[],
                heartbeat_interval_seconds=5.0,
                started_at="2026-09-30T00:00:00Z",
                gpu_policy={"mode": "all", "warnings": []},
                stop_reason="stopped_by_signal" if operation == "machine_stop_publish" else None,
            )
        snapshot_lock = (
            local_paths(runtime.project_paths(binding.project_id)["root"])["locks"] / "machine-snapshot.lock"
        )
        with runtime.registry_guard(), runtime.binding_commit_guard(binding), exclusive(snapshot_lock):
            assert executor.start(request.request_id) is not None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                executor.poll()
                result = executor.consume(request.request_id, request)
                if result is not None:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("shared worker waited for a controller-owned local lock")
            assert result.status == "completed"
        expected = {
            "registration_renew": "eligible",
            "activation_consumer_register": "registered",
            "machine_snapshot_publish": "published",
            "machine_stop_publish": "published",
        }
        assert result.evidence["outcome"] == expected[operation]
        if operation in {"machine_snapshot_publish", "machine_stop_publish"}:
            agent = read_json(machine_state_path(cfg, "agent.json"))["agent"]
            assert agent["instance_id"] == "agent-a"
            assert agent["observed_state"] == ("stopped" if operation == "machine_stop_publish" else "idle")
            assert agent.get("stop_reason") == ("stopped_by_signal" if operation == "machine_stop_publish" else None)
    finally:
        executor.shutdown()


def test_registration_shared_guard_rechecks_identity_before_publication(tmp_path: Path, monkeypatch):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    try:
        registration = load_machine_registration(cfg)
        registration["registration"]["client_version"] = "0.0.1"
        save_machine_registration(cfg, registration)
        before = load_machine_registration(cfg)
        request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=0.0)
        real_expiry = registration_authority.lease_expiry

        def revoke_during_read(policy):
            expiry = real_expiry(policy)
            executor.fence_epoch()
            return expiry

        monkeypatch.setattr(registration_authority, "lease_expiry", revoke_during_read)
        possible = [False]
        with pytest.raises(worker._ExecutorEpochFenced):
            worker._registration_renew(request, runtime.root, executor.paths, possible)
        assert possible == [False]
        assert load_machine_registration(cfg) == before
    finally:
        executor.shutdown()


def test_ambiguous_version_refresh_replay_completes_directory_barrier(tmp_path: Path, monkeypatch):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    try:
        envelope = load_machine_registration(cfg)
        envelope["registration"]["client_version"] = "0.0.1"
        save_machine_registration(cfg, envelope)
        request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=0.0)
        real_save = registration_authority.save_machine_registration

        def commit_then_raise(root_config, value):
            real_save(root_config, value)
            raise OSError("injected post-replace failure")

        possible = [False]
        with monkeypatch.context() as interrupted:
            interrupted.setattr(registration_authority, "save_machine_registration", commit_then_raise)
            with pytest.raises(OSError, match="post-replace"):
                worker._registration_renew(request, runtime.root, executor.paths, possible)
        assert possible == [True]
        committed = load_machine_registration(cfg)["registration"]
        assert committed["client_version"] == __version__
        generation = committed["generation"]

        def fail_barrier(_durable_io, _path, _operation):
            raise OSError("injected replay barrier failure")

        with monkeypatch.context() as interrupted:
            interrupted.setattr(DurableIO, "sync_directory", fail_barrier)
            possible = [False]
            with pytest.raises(OSError, match="replay barrier"):
                worker._registration_renew(request, runtime.root, executor.paths, possible)
            assert possible == [True]

        barriers = []

        def record_barrier(_durable_io, path, operation):
            barriers.append((path, operation))

        monkeypatch.setattr(DurableIO, "sync_directory", record_barrier)
        possible = [False]
        evidence = worker._registration_renew(request, runtime.root, executor.paths, possible)

        assert evidence["outcome"] == "eligible"
        assert evidence["renewed"] is False
        assert possible == [False]
        assert barriers == [(cfg.shared_root / "machines" / cfg.machine_name, "registration_renewal_replay")]
        assert load_machine_registration(cfg)["registration"]["generation"] == generation
    finally:
        executor.shutdown()


def test_short_registration_lease_does_not_spend_half_its_life_in_cooldown(tmp_path: Path):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    try:
        save_lease_policy(
            cfg,
            LeasePolicy(
                ttl_seconds=4,
                renew_interval_seconds=0.5,
                max_clock_skew_seconds=0.1,
                renewal_commit_margin_seconds=0.1,
                retry_initial_seconds=0.1,
                retry_max_seconds=0.2,
            ),
        )
        registration = load_machine_registration(cfg)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=4)
        registration["registration"]["eligibility_expires_at"] = expires_at.isoformat()
        save_machine_registration(cfg, registration)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = controller.advance_registration_renewals([binding], revision, renewal_horizon_seconds=0.0)
            if binding.project_id in result:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("short registration proof did not complete")
        # The next service still needs time for queueing and worker startup.
        assert result[binding.project_id]["outcome"] == "eligible"
        assert all(100.0 <= due <= 101.0 for due in controller._registration_due.values())
        now[0] = 101.01
        controller.advance_registration_renewals([binding], revision, renewal_horizon_seconds=0.0)
        assert any(request.operation_kind == "registration_renew" for request in executor.unresolved_requests())
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "case", ["current", "superseded", "other_runtime", "other_generation", "disabled", "malformed"]
)
def test_registration_reactivation_requires_exact_nonsuperseded_ownership(tmp_path: Path, case: str):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    try:
        task = submit(cfg, ["true"], working_dir=tmp_path)
        registration = load_machine_registration(cfg)
        record = registration["registration"]
        record["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
        if case == "superseded":
            record["state"] = "superseded"
        elif case == "other_runtime":
            record["runtime_instance_id"] = "other-runtime"
        elif case == "other_generation":
            record["generation"] = "replacement-generation"
        elif case == "malformed":
            record["eligibility_expires_at"] = "not-a-time"
        save_machine_registration(cfg, registration)
        request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=0.0)
        if case == "disabled":
            runtime.set_enabled(binding.project_id, False)
        before = load_machine_registration(cfg)
        before_task = task_path(cfg.shared_root, task.task_id).read_bytes()
        assert executor.start(request.request_id) is not None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            executor.poll()
            result = executor.consume(request.request_id, request)
            if result is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("registration reactivation did not settle")
        assert result.status == "completed"
        if case == "current":
            assert result.evidence["outcome"] == "eligible"
            assert result.evidence["renewed"] is True
            renewed = load_machine_registration(cfg)["registration"]
            assert renewed["generation"] == record["generation"]
            assert datetime.fromisoformat(renewed["eligibility_expires_at"].replace("Z", "+00:00")) > datetime.now(
                timezone.utc
            )
        else:
            assert result.evidence["outcome"] == "stale"
            assert result.evidence["renewed"] is False
            assert load_machine_registration(cfg) == before
        assert task_path(cfg.shared_root, task.task_id).read_bytes() == before_task
    finally:
        executor.shutdown()
