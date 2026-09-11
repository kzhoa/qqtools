"""Machine control plane."""
from __future__ import annotations
import threading
import time
from ..authority import AuthoritySupervisor
from ..config_types import RootConfig
from ..runtime.paths import local_paths
from ..runtime.resources.reservations import reservation_snapshot
from ..runtime.store import iter_json
from .context import MachineRuntime, ProjectBinding
from .helpers import _publish_project_snapshots, _read_pid
from . import helpers as _helpers
from ..legacy_agent import _visible_gpus
from ..authority import AuthoritySupervisor as _AuthoritySupervisor

class _MachineControlPlane:
    """Run deadline-sensitive heartbeats and authority renewal outside dispatch scans."""

    def __init__(
        self,
        runtime: MachineRuntime,
        *,
        instance_id: str,
        loop_interval: float,
        started_at: str,
        available_gpus: list[int] | None,
        scheduler_wakeup: threading.Event | None = None,
    ) -> None:
        self._runtime = runtime
        self._instance_id = instance_id
        self._loop_interval = loop_interval
        self._started_at = started_at
        self._visible_gpus = list(available_gpus) if available_gpus is not None else None
        self._scheduler_wakeup = scheduler_wakeup
        self._registry_revision: int | None = None
        self._stop_event = threading.Event()
        self._supervisors: dict[str, AuthoritySupervisor] = {}
        self._supervisor_generations: dict[str, str | None] = {}
        self._authority_thread = threading.Thread(
            target=self._run_authority_loop,
            name="qexp-machine-authority",
            daemon=True,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_loop,
            name="qexp-machine-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        """Publish initial liveness before starting the control loops."""
        self._refresh_visible_gpus()
        self._publish_heartbeat()
        self._authority_thread.start()
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop control loops before publishing the terminal machine snapshot."""
        self._stop_event.set()
        for thread in (self._authority_thread, self._heartbeat_thread):
            if thread.is_alive():
                thread.join()

    def _supervised_bindings(self) -> list[ProjectBinding]:
        revision, registered = self._runtime.load_registry()
        if (
            self._registry_revision is not None
            and revision != self._registry_revision
            and self._scheduler_wakeup is not None
        ):
            self._scheduler_wakeup.set()
        self._registry_revision = revision
        return [
            binding
            for binding in registered
            if (binding.enabled or self._runtime.binding_state(binding) == "draining")
            and self._runtime.registration_status(binding)["state"] != "superseded"
        ]

    def _run_authority_cycle(self) -> float:
        authority_interval = self._loop_interval
        reserved_before = {
            gpu_id
            for item in reservation_snapshot(self._runtime.root).reservations
            for gpu_id in item.get("gpu_ids", [])
        }
        try:
            bindings = self._supervised_bindings()
        except (OSError, RuntimeError, ValueError):
            return authority_interval
        supervised_ids = {binding.project_id for binding in bindings}
        for project_id in set(self._supervisors) - supervised_ids:
            del self._supervisors[project_id]
            self._supervisor_generations.pop(project_id, None)
        for binding in bindings:
            try:
                cfg = _helpers._binding_config(self._runtime, binding)
                is_eligible = self._runtime.binding_write_eligible(binding, renew=True)
                has_local_process = any(iter_json(local_paths(cfg.runtime_root)["processes"]))
                if not is_eligible and not has_local_process:
                    continue
                supervisor = self._supervisors.get(binding.project_id)
                if supervisor is not None and self._supervisor_generations.get(binding.project_id) != (
                    binding.registration_generation
                ):
                    del self._supervisors[binding.project_id]
                    supervisor = None
                if supervisor is None:
                    supervisor = _AuthoritySupervisor(cfg, reservation_runtime_root=self._runtime.root)
                    supervisor.recover_startup()
                    self._supervisors[binding.project_id] = supervisor
                    self._supervisor_generations[binding.project_id] = binding.registration_generation
                if not is_eligible and not self._runtime.reactivate_binding(binding):
                    continue
                supervisor.tick()
                authority_interval = min(authority_interval, supervisor.renewal_interval_seconds)
            except OSError:
                try:
                    local_cfg = RootConfig(
                        binding.shared_root,
                        binding.shared_root.parent,
                        binding.machine_name,
                        self._runtime.project_paths(binding.project_id)["root"],
                    )
                    local_supervisor = _AuthoritySupervisor(local_cfg, reservation_runtime_root=self._runtime.root)
                    local_supervisor.reconcile_local_exit_evidence()
                except (OSError, RuntimeError, ValueError):
                    continue
            except (RuntimeError, ValueError):
                continue
        reserved_after = {
            gpu_id
            for item in reservation_snapshot(self._runtime.root).reservations
            for gpu_id in item.get("gpu_ids", [])
        }
        if reserved_after < reserved_before and self._scheduler_wakeup is not None:
            self._scheduler_wakeup.set()
        return authority_interval

    def _refresh_visible_gpus(self) -> None:
        if self._visible_gpus is not None:
            return
        try:
            bindings = self._supervised_bindings()
            if bindings:
                self._visible_gpus = _visible_gpus(_helpers._binding_config(self._runtime, bindings[0]))
        except (OSError, RuntimeError, ValueError):
            return

    def _publish_heartbeat(self) -> None:
        self._refresh_visible_gpus()
        try:
            bindings = self._supervised_bindings()
        except (OSError, RuntimeError, ValueError):
            return
        readable: dict[str, RootConfig] = {}
        readable_bindings: dict[str, ProjectBinding] = {}
        for binding in bindings:
            try:
                if not self._runtime.binding_write_eligible(binding, renew=True):
                    continue
                readable[binding.project_id] = _helpers._binding_config(self._runtime, binding)
                readable_bindings[binding.project_id] = binding
            except (OSError, RuntimeError, ValueError):
                continue
        try:
            _publish_project_snapshots(
                readable,
                instance_id=self._instance_id,
                pid=_read_pid(self._runtime),
                visible=self._visible_gpus or [],
                reservations=list(reservation_snapshot(self._runtime.root).reservations),
                heartbeat_interval_seconds=self._loop_interval,
                started_at=self._started_at,
                write_guard=lambda project_id: self._runtime.binding_write_guard(readable_bindings[project_id]),
            )
        except (OSError, RuntimeError, ValueError):
            return

    def _run_authority_loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            deadline = time.monotonic() + self._run_authority_cycle()

    def _run_heartbeat_loop(self) -> None:
        self._run_deadline_loop(self._publish_heartbeat, initial_delay=self._loop_interval)

    def _run_deadline_loop(self, operation, *, initial_delay: float = 0.0) -> None:
        """Run an operation on monotonic deadlines without adding execution time to the period."""
        deadline = time.monotonic() + initial_delay
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            operation()
            deadline += self._loop_interval
            now = time.monotonic()
            if deadline <= now:
                missed_intervals = int((now - deadline) // self._loop_interval) + 1
                deadline += missed_intervals * self._loop_interval
