from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.qexp.lifecycle import (
    LifecycleBranch,
    LifecycleDeadline,
    LifecycleLab,
    LifecycleWaitError,
    wait_all,
    wait_until,
)


def test_wait_all_observes_every_condition_under_one_deadline() -> None:
    state = {"first": False, "second": False}
    calls = {"first": 0, "second": 0}

    def ready(name: str, threshold: int) -> bool:
        calls[name] += 1
        state[name] = calls[name] >= threshold
        return state[name]

    wait_all(
        {
            "first": lambda: ready("first", 1),
            "second": lambda: ready("second", 2),
        },
        timeout=0.1,
        poll_interval=0.001,
    )

    assert calls == {"first": 1, "second": 2}


def test_wait_all_reports_only_pending_condition_names() -> None:
    with pytest.raises(AssertionError, match=r"\['blocked'\]"):
        wait_all(
            {"ready": lambda: True, "blocked": lambda: False},
            timeout=0.001,
            poll_interval=0.0001,
        )


def test_wait_all_reports_stage_and_snapshot_for_timeout() -> None:
    with pytest.raises(LifecycleWaitError) as caught:
        wait_all(
            {"blocked": lambda: False},
            deadline=LifecycleDeadline.after(0.001),
            poll_interval=0.0001,
            stage="evidence",
            on_timeout=lambda: {"attempt": "a1"},
        )

    assert caught.value.report.name == "evidence"
    assert caught.value.report.pending == ("blocked",)
    assert caught.value.report.snapshot == {"attempt": "a1"}


def test_wait_all_collects_predicate_failures_without_hiding_other_pending() -> None:
    with pytest.raises(LifecycleWaitError) as caught:
        wait_all(
            {"broken": lambda: 1 / 0, "blocked": lambda: False},
            timeout=0.001,
            poll_interval=0.0001,
            stage="diagnostics",
        )

    assert caught.value.report.pending == ("blocked",)
    assert "ZeroDivisionError" in caught.value.report.failures["broken"]


def test_wait_until_returns_named_stage_report() -> None:
    report = wait_until("ready", lambda: True, stage="registration")

    assert report.name == "registration"
    assert report.completed == ("ready",)


def test_wait_until_honors_shared_deadline() -> None:
    with pytest.raises(LifecycleWaitError) as caught:
        wait_until(
            "ready",
            lambda: False,
            timeout=10.0,
            deadline=LifecycleDeadline.after(0.001),
            poll_interval=0.0001,
        )

    assert caught.value.report.pending == ("ready",)
    assert caught.value.report.duration_seconds < 1.0


def test_wait_all_preserves_wait_failure_when_snapshot_fails() -> None:
    def broken_snapshot():
        raise RuntimeError("snapshot failed")

    with pytest.raises(LifecycleWaitError) as caught:
        wait_all(
            {"blocked": lambda: False},
            timeout=0.001,
            poll_interval=0.0001,
            on_timeout=broken_snapshot,
        )

    assert caught.value.report.pending == ("blocked",)
    assert caught.value.report.snapshot == {"snapshot_failure": "RuntimeError: snapshot failed"}


def test_lifecycle_lab_rejects_duplicate_branch_names() -> None:
    lab = LifecycleLab(runtime=SimpleNamespace(), available_gpus=[])
    branch = LifecycleBranch("same", SimpleNamespace(), SimpleNamespace(task_id="task"), Path("marker"))
    lab.add_branch(branch)
    with pytest.raises(ValueError, match="must be unique"):
        lab.add_branch(branch)


def test_lifecycle_lab_terminal_wait_rejects_unknown_branch() -> None:
    lab = LifecycleLab(runtime=SimpleNamespace(), available_gpus=[])
    with pytest.raises(ValueError, match="unknown lifecycle branches"):
        lab.wait_terminal({"missing": "succeeded"})


def test_lifecycle_lab_cleanup_reports_every_process_error(monkeypatch) -> None:
    class BrokenProcess:
        def poll(self):
            return None

        def kill(self):
            raise OSError("kill failed")

        def wait(self, timeout):
            raise AssertionError("wait must not run after kill failure")

    lab = LifecycleLab(runtime=SimpleNamespace(), available_gpus=[])
    lab._processes.extend((BrokenProcess(), BrokenProcess()))
    monkeypatch.setattr("tests.helpers.qexp.lifecycle.get_machine_agent_status", lambda _runtime: {"is_running": False})

    with pytest.raises(ExceptionGroup) as caught:
        lab.close()

    assert len(caught.value.exceptions) == 2
