from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp import notification_reconciliation as reconciliation
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.notification_cleanup import cleanup_credentials
from qqtools.plugins.qexp.notification_config import (
    shared_feishu_webhook_path,
    update_notifications,
    write_shared_feishu_webhook,
)
from qqtools.plugins.qexp.notification_policy import load_policy, policy_guard, replace_policy
from qqtools.plugins.qexp.notification_resolver import resolve_delivery
from qqtools.plugins.qexp.runtime.locks import machine_lock
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]
WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/private-import-fixture"


def _case(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    return runtime, cfg, binding, revision, executor


def _legacy(cfg):
    update_notifications(
        cfg,
        lambda _current: {
            "enabled": True,
            "providers": {"feishu": {"enabled": True, "credential_source": "shared_file"}},
        },
    )
    write_shared_feishu_webhook(cfg, WEBHOOK)


def _consume(executor, request):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("notification request did not finish")


def _run(executor, binding, revision):
    request = executor.prepare_notification_service(binding, revision)
    assert executor.start(request.request_id) is not None
    result = _consume(executor, request)
    assert result.status == "completed"
    return result


def test_notification_import_is_private_idempotent_and_independent_of_local_authority(tmp_path):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    try:
        with runtime.registry_guard(), runtime.binding_commit_guard(binding):
            assert _run(executor, binding, revision).evidence == {"state": "ready"}
        policy = load_policy(runtime.root, "project", binding.project_id)
        assert resolve_delivery(runtime.root, binding.project_id)["webhook"] == WEBHOOK
        assert policy["legacy"]["status"] == "ready"
        assert _run(executor, binding, revision).evidence == {"state": "ready"}
        assert load_policy(runtime.root, "project", binding.project_id) == policy
        for path in executor.paths["project_io_root"].rglob("*.json"):
            assert WEBHOOK.encode() not in path.read_bytes()
    finally:
        executor.shutdown()


def test_notification_conflict_and_invalid_source_preserve_public_migration_states(tmp_path):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    try:
        shared_feishu_webhook_path(cfg).unlink()
        assert _run(executor, binding, revision).evidence == {"state": "source_invalid"}
        invalid = load_policy(runtime.root, "project", binding.project_id)
        assert invalid["legacy"]["status"] == "source_invalid"
        assert _run(executor, binding, revision).evidence == {"state": "source_invalid"}
        assert load_policy(runtime.root, "project", binding.project_id) == invalid
        write_shared_feishu_webhook(cfg, WEBHOOK)
        assert _run(executor, binding, revision).evidence == {"state": "ready"}
        ready = load_policy(runtime.root, "project", binding.project_id)
        replace_policy(runtime.root, "project", ready["revision"], {"enabled": False}, binding.project_id)
        write_shared_feishu_webhook(cfg, WEBHOOK + "-changed")
        assert _run(executor, binding, revision).evidence == {"state": "conflict"}
        conflict = load_policy(runtime.root, "project", binding.project_id)
        assert conflict["override"] == {"enabled": False}
        assert conflict["legacy"]["status"] == "legacy_conflict"
    finally:
        executor.shutdown()


def test_notification_shared_capture_and_fence_do_not_read_shared_root_under_policy_lock(tmp_path, monkeypatch):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    held = [False]
    real_guard = reconciliation.policy_guard
    real_capture = reconciliation.capture_legacy
    real_current = worker._claim_binding_is_current

    @contextmanager
    def guard(*args, **kwargs):
        with real_guard(*args, **kwargs):
            held[0] = True
            try:
                yield
            finally:
                held[0] = False

    def capture(*args, **kwargs):
        assert not held[0]
        return real_capture(*args, **kwargs)

    def current(*args, **kwargs):
        assert not held[0]
        return real_current(*args, **kwargs)

    monkeypatch.setattr(reconciliation, "policy_guard", guard)
    monkeypatch.setattr(reconciliation, "capture_legacy", capture)
    monkeypatch.setattr(worker, "_claim_binding_is_current", current)
    try:
        request = executor.prepare_notification_service(binding, revision)
        assert worker._notification_service(request, runtime.root, executor.paths, [False]) == {"state": "ready"}
        assert resolve_delivery(runtime.root, binding.project_id)["webhook"] == WEBHOOK
    finally:
        executor.shutdown()


@pytest.mark.parametrize("revocation", ["epoch", "disabled"])
def test_notification_capture_revocation_prevents_private_publication(tmp_path, monkeypatch, revocation):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    real_capture = reconciliation.capture_legacy

    def capture(config):
        snapshot = real_capture(config)
        if revocation == "epoch":
            executor.fence_epoch()
        else:
            runtime.set_enabled(binding.project_id, False)
        return snapshot

    monkeypatch.setattr(reconciliation, "capture_legacy", capture)
    try:
        request = executor.prepare_notification_service(binding, revision)
        expected = worker._ExecutorEpochFenced if revocation == "epoch" else worker._BindingAuthorityChanged
        with pytest.raises(expected):
            worker._notification_service(request, runtime.root, executor.paths, [False])
        assert load_policy(runtime.root, "project", binding.project_id)["revision"] == 0
        assert not list((runtime.root / "notifications/credentials").glob("*.json"))
    finally:
        executor.shutdown()


@pytest.mark.parametrize("failure", ["worker_absence", "late_epoch"])
def test_notification_possible_private_write_remains_ambiguous(tmp_path, failure):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    try:
        request = executor.prepare_notification_service(binding, revision)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
        if failure == "late_epoch":
            executor.fence_epoch()
            assert worker._publish_result(
                runtime.root,
                executor.paths,
                request,
                process,
                "completed",
                None,
                {"state": "ready"},
                shared_write_possible=True,
            )
        else:
            executor.poll()
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
        if failure == "worker_absence":
            assert executor.reset_ambiguous_notification_service_for_retry(request.request_id, request)
            assert executor.start(request.request_id) is not None
            assert _consume(executor, request).evidence == {"state": "ready"}
    finally:
        executor.shutdown()


def test_notification_credential_publication_rechecks_private_fence(tmp_path, monkeypatch):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    real_stage = reconciliation.stage_webhook

    def revoke_before_stage(root, webhook):
        executor.fence_epoch()
        return real_stage(root, webhook)

    monkeypatch.setattr(reconciliation, "stage_webhook", revoke_before_stage)
    try:
        request = executor.prepare_notification_service(binding, revision)
        with pytest.raises(worker._ExecutorEpochFenced):
            worker._notification_service(request, runtime.root, executor.paths, [False])
        assert load_policy(runtime.root, "project", binding.project_id)["revision"] == 0
        assert not list((runtime.root / "notifications/credentials").glob("*.json"))
    finally:
        executor.shutdown()


def test_notification_blocked_project_does_not_delay_healthy_import(tmp_path, monkeypatch):
    runtime, cfg_blocked, blocked, _revision, executor = _case(tmp_path)
    cfg_healthy = init_shared_root(tmp_path / "healthy/.qexp", "gpu-1")
    healthy = runtime.add_binding(cfg_healthy.shared_root, cfg_healthy.machine_name)
    revision, bindings = runtime.load_registry()
    _legacy(cfg_healthy)
    controller = ProjectIOController(runtime, executor)
    blocked_path = cfg_blocked.shared_root / "schema/version.json"
    saved = blocked_path.with_suffix(".saved")
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            controller.advance_binding_validation(bindings, revision)
            if all(controller.validated_config(binding, revision) is not None for binding in bindings):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("notification bindings did not validate")
        monkeypatch.setattr(
            reconciliation, "reconcile_captured_legacy", lambda *args, **kwargs: pytest.fail("inline shared capture")
        )
        blocked_path.rename(saved)
        os.mkfifo(blocked_path)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            controller.advance_notification_maintenance(bindings, revision)
            if healthy.project_id in {identity.project_id for identity in controller._notification_due}:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("blocked source withheld the healthy import")
        assert resolve_delivery(runtime.root, healthy.project_id)["webhook"] == WEBHOOK
        requests = executor.unresolved_requests()
        assert len([request for request in requests if request.project_id == blocked.project_id]) == 1
        assert executor.status_view()["active_worker_count"] <= 4
    finally:
        if saved.exists():
            blocked_path.unlink()
            saved.rename(blocked_path)
        executor.shutdown()


def test_local_credential_cleanup_does_not_wait_for_private_worker_transaction(tmp_path):
    runtime, _cfg, _binding, _revision, executor = _case(tmp_path)
    try:
        with policy_guard(runtime.root):
            started = time.monotonic()
            result = cleanup_credentials(runtime.root, limit=8, blocking=False)
            assert time.monotonic() - started < 0.5
            assert result["deleted"] == 0
            assert result["diagnostic"] is not None
    finally:
        executor.shutdown()


@pytest.mark.parametrize("lock", ["shared", "private"])
def test_notification_busy_transaction_returns_blocked_without_waiting(tmp_path, lock):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    try:
        guard = machine_lock(cfg.shared_root, cfg.machine_name) if lock == "shared" else policy_guard(runtime.root)
        with guard:
            assert _run(executor, binding, revision).evidence == {"state": "blocked"}
        assert load_policy(runtime.root, "project", binding.project_id)["revision"] == 0
        assert _run(executor, binding, revision).evidence == {"state": "ready"}
    finally:
        executor.shutdown()


def test_notification_partial_policy_failure_is_unknown_and_replays_idempotently(tmp_path, monkeypatch):
    runtime, cfg, binding, revision, executor = _case(tmp_path)
    _legacy(cfg)
    request = executor.prepare_notification_service(binding, revision)
    real_publish = reconciliation.replace_policy_unlocked
    try:
        # Exercise the real worker error classifier with its exact handshake.
        pid = os.getpid()
        ticks = worker._process_start_time_ticks(pid)
        assert ticks is not None
        process = ProjectIOProcess(request, pid, ticks, request.prepared_at, "running")
        process_path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
        atomic_replace(process_path, process.to_dict())

        def fail_after_publish(*args, **kwargs):
            real_publish(*args, **kwargs)
            raise RuntimeError("injected failure after private publication")

        with monkeypatch.context() as patch:
            patch.setattr(reconciliation, "replace_policy_unlocked", fail_after_publish)
            assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert result.evidence == {}
        before = load_policy(runtime.root, "project", binding.project_id)
        assert before["legacy"]["status"] == "ready"
        assert executor.consume(request.request_id, request) is None
        # The in-process test worker is alive, so replace only its synthetic
        # process witness with positively absent identity before retrying.
        atomic_replace(
            process_path, ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running").to_dict()
        )
        assert executor.reset_ambiguous_notification_service_for_retry(request.request_id, request)
        assert executor.start(request.request_id) is not None
        assert _consume(executor, request).evidence == {"state": "ready"}
        assert load_policy(runtime.root, "project", binding.project_id) == before
    finally:
        # Never leave the pytest process recorded as a child shutdown target.
        if (process_path := executor.paths["project_io_processes"] / f"{request.request_id}.json").exists():
            atomic_replace(
                process_path, ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running").to_dict()
            )
        executor.shutdown()


@pytest.mark.parametrize("held_workers", [0, 2])
def test_notification_roster_beyond_intent_window_progresses_with_new_arrivals(tmp_path, held_workers):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    healthy = []
    blocked_configs = []
    for index in range(65 + held_workers):
        cfg = init_shared_root(tmp_path / f"project-{index}/.qexp", "gpu-1")
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        if index < held_workers:
            blocked_configs.append(cfg)
        else:
            healthy.append(binding)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    restored = []
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            controller.advance_binding_validation(bindings, revision)
            if all(controller.validated_config(binding, revision) is not None for binding in bindings):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("large notification roster did not validate")
        for cfg in blocked_configs:
            path = cfg.shared_root / "schema/version.json"
            saved = path.with_suffix(".saved")
            path.rename(saved)
            os.mkfifo(path)
            restored.append((path, saved))
        if held_workers:
            blocked_bindings = [binding for binding in bindings if binding not in healthy]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                controller.advance_notification_maintenance(blocked_bindings, revision)
                if executor.status_view()["active_worker_count"] == held_workers:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("the qualification did not establish both held workers")
        arrivals_added = False
        ready_ids = set()
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            controller.advance_notification_maintenance(bindings, revision)
            observed = {identity.project_id for identity in controller._notification_due}
            for project_id in observed - ready_ids:
                if load_policy(runtime.root, "project", project_id).get("legacy", {}).get("status") == "ready":
                    ready_ids.add(project_id)
            if not arrivals_added and len(ready_ids) >= 4:
                for index in range(3):
                    cfg = init_shared_root(tmp_path / f"arrival-{index}/.qexp", "gpu-1")
                    healthy.append(runtime.add_binding(cfg.shared_root, cfg.machine_name))
                revision, bindings = runtime.load_registry()
                arrivals_added = True
            if arrivals_added:
                controller.advance_binding_validation(healthy, revision)
            status = executor.status_view()
            assert status["active_worker_count"] <= 4
            if arrivals_added and {binding.project_id for binding in healthy}.issubset(ready_ids):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("notification incumbents beyond the intent window were starved")
        for binding in healthy:
            assert load_policy(runtime.root, "project", binding.project_id)["legacy"]["status"] == "ready"
        live = executor.unresolved_requests()
        blocked_ids = {
            binding.project_id
            for binding in bindings
            if binding.project_id not in {item.project_id for item in healthy}
        }
        assert all(sum(request.project_id == project_id for request in live) == 1 for project_id in blocked_ids)
        assert status["overdue_worker_count"] <= held_workers
    finally:
        for path, saved in restored:
            path.unlink()
            saved.rename(path)
        executor.shutdown()
