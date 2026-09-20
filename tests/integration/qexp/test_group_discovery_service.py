"""Background discovery, post-bootstrap debt, and restart collaboration."""

import os
import time

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.service import (
    GroupDiscoveryService,
    MachineGroupDiscoveryWorker,
    publish_submission_debt,
)
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.store import read_json
from tests.helpers.qexp_discovery import isolated_group, set_tail, source_file

pytestmark = pytest.mark.integration


def close(service):
    service.request_close()
    for _ in range(10000):
        if service.is_closed:
            return
        service.advance()
    raise AssertionError("background service retained source resources")


def finish(service):
    for _ in range(20000):
        result = service.advance()
        state = result.get("state") if isinstance(result, dict) else result.state
        if state == "complete":
            return result
    raise AssertionError(f"background service did not converge: {result}")


def test_background_bootstrap_and_restart_do_not_reparse_confirmed_history(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=2)
    source_file(
        submission_path(cfg.shared_root, "batch"), operation="batch", tasks=["task-a", "task-b"], sequences=[1, 2]
    )
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        finish(service)
        assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
    finally:
        close(service)

    scandir = os.scandir

    def guarded(path):
        assert isinstance(path, int) or os.fspath(path) != str(cfg.shared_root / "operations/submissions"), (
            "healthy restart scanned source history"
        )
        return scandir(path)

    def forbidden(*args, **kwargs):
        pytest.fail("healthy restart parsed confirmed history")

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", guarded)
        guard.setattr(ProjectionDriver, "advance", forbidden)
        resumed = GroupDiscoveryService(cfg.shared_root, "experiment")
        try:
            finish(resumed)
        finally:
            close(resumed)


def test_new_debt_extends_coverage_without_historical_enumeration(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=1)
    source_file(submission_path(cfg.shared_root, "first"), operation="first")
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        finish(service)
    finally:
        close(service)
    source_file(submission_path(cfg.shared_root, "later"), operation="later", tasks=["task-b"], sequences=[2])
    publish_submission_debt(cfg.shared_root, "experiment", "later")
    set_tail(cfg, 2)
    scandir = os.scandir

    def guarded(path):
        assert isinstance(path, int) or os.fspath(path) != str(cfg.shared_root / "operations/submissions")
        return scandir(path)

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", guarded)
        resumed = GroupDiscoveryService(cfg.shared_root, "experiment")
        try:
            finish(resumed)
            status = GroupCoverage(cfg.shared_root, "experiment").status()
            assert (status.prefix, status.tail, status.is_complete) == (2, 2, True)
        finally:
            close(resumed)


def test_submission_publishes_debt_before_clearing_group_pending(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import submission as submission_module

    cfg = isolated_group(tmp_path, tail=0)
    original = submission_module._write_group_record
    saw_finalization = False

    def guarded(config, path, record):
        nonlocal saw_finalization
        if record["group"].get("next_membership_sequence", 1) > 1 and not record["group"].get(
            "pending_submission_commit"
        ):
            debts = list((cfg.shared_root / "operations/group-discovery/active").glob("*/*.json"))
            assert debts, "pending locator disappeared before discoverable coverage debt"
            operation_ids = {read_json(debt).get("operation_id") for debt in debts}
            # The pending finalizer has one currently committed source in this isolated fixture.
            sources = list((cfg.shared_root / "operations/submissions").glob("*.json"))
            assert any(read_json(source)["submission"]["operation_id"] in operation_ids for source in sources)
            saw_finalization = True
        return original(config, path, record)

    monkeypatch.setattr(submission_module, "_write_group_record", guarded)
    task = submit(cfg, ["true"], group="experiment")
    assert task is not None
    assert saw_finalization


def test_directory_eof_with_missing_membership_stays_unavailable(tmp_path):
    cfg = isolated_group(tmp_path, tail=2)
    source_file(submission_path(cfg.shared_root, "later"), operation="later", tasks=["task-b"], sequences=[2])
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        for _ in range(600):
            service.advance()
        status = GroupCoverage(cfg.shared_root, "experiment").status()
        assert status.prefix == 0
        assert not status.is_complete
    finally:
        close(service)


def test_real_background_worker_advances_registered_group(tmp_path):
    cfg = isolated_group(tmp_path, tail=1)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    worker = MachineGroupDiscoveryWorker(runtime)
    worker.start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = GroupCoverage(cfg.shared_root, "experiment").status()
            if status.is_complete:
                break
            time.sleep(0.02)
        assert status.is_complete
    finally:
        worker.stop()
    assert not worker.is_alive


def test_failed_debt_publication_retains_pending_and_recovery_completes(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import submission as submission_module
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = isolated_group(tmp_path, tail=0)

    def fail(*args, **kwargs):
        raise OSError("debt publication interrupted")

    with monkeypatch.context() as crashing:
        crashing.setattr(service_module, "publish_submission_debt", fail)
        with pytest.raises(OSError, match="debt publication interrupted"):
            submit(cfg, ["true"], group="experiment", idempotency_key="crashed-finalizer")
    group = read_json(group_path(cfg.shared_root, "experiment"))["group"]
    assert group["pending_submission_commit"]
    operation = group["pending_submission_commit"]["operation_id"]
    source = read_json(submission_path(cfg.shared_root, operation))["submission"]
    assert source["state"] == "committed"
    submission_module.finalize_submission_group(cfg, source)
    assert not read_json(group_path(cfg.shared_root, "experiment"))["group"]["pending_submission_commit"]
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        finish(service)
        assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
    finally:
        close(service)


def test_preparing_source_revision_does_not_hide_later_commit(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    path = submission_path(cfg.shared_root, "changing")
    source_file(path, operation="changing", state="preparing")
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        finish(service)
    finally:
        close(service)
    replacement = source_file(path.with_suffix(".new"), operation="changing")
    replacement.replace(path)
    publish_submission_debt(cfg.shared_root, "experiment", "changing")
    set_tail(cfg, 1)
    resumed = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        finish(resumed)
        assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
    finally:
        close(resumed)


def test_agent_lifecycle_starts_discovery_and_still_exits_when_idle(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    cfg = isolated_group(tmp_path, tail=100)
    source_file(
        submission_path(cfg.shared_root, "large"),
        operation="large",
        tasks=[f"t{i}" for i in range(100)],
        sequences=list(range(1, 101)),
    )
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    program = r"""
import sys
from pathlib import Path
from qqtools.plugins.qexp.agent import lifecycle
original = lifecycle.MachineGroupDiscoveryWorker
marker = Path(sys.argv[2])
class ObservedWorker(original):
    def start(self):
        super().start()
        marker.with_suffix(".started").write_text("started")
    def stop(self):
        super().stop()
        if self.is_alive:
            raise RuntimeError("discovery outlived idle shutdown")
        marker.with_suffix(".stopped").write_text("stopped")
lifecycle.MachineGroupDiscoveryWorker = ObservedWorker
lifecycle.run_machine_agent_loop(sys.argv[1], loop_interval=0.05, available_gpus=[])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    marker = tmp_path / "worker"
    child = subprocess.run(
        [sys.executable, "-c", program, str(runtime.root), str(marker)],
        env=environment,
        capture_output=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr.decode()
    assert marker.with_suffix(".started").exists()
    assert marker.with_suffix(".stopped").exists()
    assert not runtime.paths["pid"].exists()


@pytest.mark.parametrize("cut", ["before_marker", "after_marker"])
def test_new_commit_at_bootstrap_handoff_remains_discoverable(tmp_path, monkeypatch, cut):
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module

    cfg = isolated_group(tmp_path, tail=1)
    source_file(submission_path(cfg.shared_root, "first"), operation="first")
    original = service_module.atomic_replace
    has_cut = False

    def arrive():
        source_file(submission_path(cfg.shared_root, "later"), operation="later", tasks=["task-b"], sequences=[2])
        publish_submission_debt(cfg.shared_root, "experiment", "later")
        set_tail(cfg, 2)

    def interrupted(path, record):
        nonlocal has_cut
        is_boundary = path.name == "background.json" and record.get("bootstrap_complete") and not has_cut
        if is_boundary:
            has_cut = True
            if cut == "before_marker":
                arrive()
        original(path, record)
        if is_boundary:
            if cut == "after_marker":
                arrive()
            raise OSError("bootstrap marker durable")

    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        with monkeypatch.context() as crashing:
            crashing.setattr(service_module, "atomic_replace", interrupted)
            for _ in range(10000):
                try:
                    service.advance()
                except OSError as exc:
                    assert "bootstrap marker durable" in str(exc)
                if has_cut:
                    break
        assert has_cut
    finally:
        close(service)
    scandir = os.scandir

    def guarded(path):
        assert isinstance(path, int) or os.fspath(path) != str(cfg.shared_root / "operations/submissions")
        return scandir(path)

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", guarded)
        resumed = GroupDiscoveryService(cfg.shared_root, "experiment")
        try:
            finish(resumed)
            assert GroupCoverage(cfg.shared_root, "experiment").status().prefix == 2
        finally:
            close(resumed)


def test_empty_debt_directory_finishes_bootstrap(tmp_path):
    import hashlib

    cfg = isolated_group(tmp_path, tail=0)
    debt_dir = cfg.shared_root / "operations/group-discovery/active" / hashlib.sha256(b"experiment").hexdigest()
    debt_dir.mkdir(parents=True)
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        for _ in range(200):
            result = service.advance()
            if result["state"] == "complete":
                break
        assert result["state"] == "complete", result
    finally:
        close(service)
