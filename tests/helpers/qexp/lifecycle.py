"""Shared wall-clock wait primitives for real qexp lifecycle labs."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
from qqtools.plugins.qexp.runtime.tasks import load_task


@dataclass(frozen=True)
class LifecycleDeadline:
    """Share one monotonic deadline across related lifecycle waits."""

    expires_at: float

    @classmethod
    def after(cls, timeout: float) -> LifecycleDeadline:
        return cls(time.monotonic() + timeout)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


@dataclass(frozen=True)
class LifecycleStageReport:
    """Record the outcome of one named lifecycle wait stage."""

    name: str
    duration_seconds: float
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    failures: Mapping[str, str]
    snapshot: Any = None


class LifecycleWaitError(AssertionError):
    """Expose all failures and pending branches from a lifecycle stage."""

    def __init__(self, report: LifecycleStageReport):
        self.report = report
        details = [f"lifecycle conditions did not converge in stage {report.name!r}"]
        if report.pending:
            details.append(f"pending={list(report.pending)!r}")
        if report.failures:
            details.append(f"failures={dict(report.failures)!r}")
        if report.snapshot is not None:
            details.append(f"snapshot={report.snapshot!r}")
        super().__init__("; ".join(details))


def wait_until(
    name: str,
    predicate: Callable[[], bool],
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.01,
    stage: str = "lifecycle",
    deadline: LifecycleDeadline | None = None,
    on_timeout: Callable[[], Any] | None = None,
) -> LifecycleStageReport:
    """Wait for one named condition with the same diagnostics as a lab stage."""
    return wait_all(
        {name: predicate},
        timeout=timeout,
        poll_interval=poll_interval,
        stage=stage,
        deadline=deadline,
        on_timeout=on_timeout,
    )


def wait_all(
    conditions: Mapping[str, Callable[[], bool]],
    *,
    timeout: float = 15.0,
    poll_interval: float = 0.01,
    stage: str = "lifecycle",
    deadline: LifecycleDeadline | None = None,
    on_timeout: Callable[[], Any] | None = None,
) -> LifecycleStageReport:
    """Poll independent conditions under one shared deadline."""
    pending = dict(conditions)
    started_at = time.monotonic()
    active_deadline = deadline or LifecycleDeadline.after(timeout)
    completed: list[str] = []
    failures: dict[str, str] = {}
    while pending and active_deadline.remaining() > 0:
        completed_now = []
        for name, predicate in pending.items():
            try:
                if predicate():
                    completed_now.append(name)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
                completed_now.append(name)
        for name in completed_now:
            del pending[name]
            if name not in failures:
                completed.append(name)
        if pending:
            time.sleep(min(poll_interval, active_deadline.remaining()))
    snapshot = None
    if (pending or failures) and on_timeout is not None:
        try:
            snapshot = on_timeout()
        except Exception as error:
            snapshot = {"snapshot_failure": f"{type(error).__name__}: {error}"}
    report = LifecycleStageReport(
        name=stage,
        duration_seconds=time.monotonic() - started_at,
        completed=tuple(sorted(completed)),
        pending=tuple(sorted(pending)),
        failures=failures,
        snapshot=snapshot,
    )
    if pending or failures:
        raise LifecycleWaitError(report)
    return report


@dataclass(frozen=True)
class LifecycleBranch:
    """Describe one independently asserted task inside a shared lifecycle lab."""

    name: str
    cfg: Any
    task: Any
    marker: Path | None = None

    @property
    def task_id(self) -> str:
        return self.task.task_id


@dataclass
class LifecycleLab:
    """Own agent processes and named task branches for one isolated machine lab."""

    runtime: MachineRuntime
    available_gpus: list[int]
    loop_interval: float = 0.1
    branches: list[LifecycleBranch] = field(default_factory=list)
    process: subprocess.Popen | None = None
    _processes: list[subprocess.Popen] = field(default_factory=list, init=False)
    reports: list[LifecycleStageReport] = field(default_factory=list, init=False)

    def add_branch(self, branch: LifecycleBranch) -> LifecycleBranch:
        if any(existing.name == branch.name for existing in self.branches):
            raise ValueError(f"lifecycle branch name must be unique: {branch.name!r}")
        self.branches.append(branch)
        return branch

    def adopt_process(self, process: subprocess.Popen) -> subprocess.Popen:
        """Make a custom agent process part of this lab's cleanup boundary."""
        self.process = process
        self._processes.append(process)
        return process

    def start_agent(self) -> subprocess.Popen:
        return self.adopt_process(
            start_machine_agent(self.runtime, available_gpus=self.available_gpus, loop_interval=self.loop_interval)
        )

    def stop_agent(self, *, timeout: float = 10.0) -> None:
        stop_machine_agent(self.runtime, timeout=timeout)
        if self.process is not None:
            self.process.wait(timeout=timeout)

    def restart_agent(self) -> subprocess.Popen:
        self.stop_agent()
        return self.start_agent()

    def wait_running(
        self,
        *,
        timeout: float = 15.0,
        deadline: LifecycleDeadline | None = None,
        on_timeout: Callable[[], Any] | None = None,
    ) -> LifecycleStageReport:
        report = wait_all(
            {
                f"running:{branch.name}": lambda branch=branch: (
                    load_task(branch.cfg, branch.task_id).state["projection"] == "running"
                    and (branch.marker is None or branch.marker.exists())
                )
                for branch in self.branches
            },
            timeout=timeout,
            stage="running",
            deadline=deadline,
            on_timeout=on_timeout,
        )
        self.reports.append(report)
        return report

    def wait_terminal(
        self,
        expected: Mapping[str, str],
        *,
        timeout: float = 15.0,
        deadline: LifecycleDeadline | None = None,
        on_timeout: Callable[[], Any] | None = None,
    ) -> LifecycleStageReport:
        unknown = sorted(set(expected) - {branch.name for branch in self.branches})
        if unknown:
            raise ValueError(f"terminal expectations reference unknown lifecycle branches: {unknown}")
        report = wait_all(
            {
                f"terminal:{branch.name}:{expected[branch.name]}": lambda branch=branch: (
                    load_task(branch.cfg, branch.task_id).state["projection"] == expected[branch.name]
                )
                for branch in self.branches
                if branch.name in expected
            },
            timeout=timeout,
            stage="terminal",
            deadline=deadline,
            on_timeout=on_timeout,
        )
        self.reports.append(report)
        return report

    def close(self) -> None:
        errors: list[Exception] = []
        try:
            if get_machine_agent_status(self.runtime)["is_running"]:
                stop_machine_agent(self.runtime, timeout=5.0)
        except (OSError, RuntimeError, TimeoutError) as error:
            errors.append(error)
        for process in reversed(self._processes):
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("lifecycle lab cleanup failed", errors)
