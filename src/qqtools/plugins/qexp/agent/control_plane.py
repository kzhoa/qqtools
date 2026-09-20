"""Machine control plane."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..authority import AuthoritySupervisor
from ..authority import AuthoritySupervisor as _AuthoritySupervisor
from ..config_types import RootConfig
from ..lease import load_lease_policy
from ..legacy_agent import _visible_gpus
from ..runtime.authority_scan import EvidenceScan, is_path_present
from ..runtime.paths import local_paths
from ..runtime.records import utc_now
from ..runtime.resources.reservations import reservation_snapshot
from ..runtime.responsibility_completion import COMPLETION_FILE
from ..runtime.store import atomic_replace
from ..runtime.work_budget import RuntimeDiagnostics, activate_diagnostics, diagnostic_increment
from . import helpers as _helpers
from .context import MachineRuntime, ProjectBinding
from .deadlines import advance_deadline
from .helpers import _publish_project_snapshots, _read_pid
from .progress_loop import ProgressObservationLoop
from .recovery_capture import recovery_owner

STARTUP_AUTHORITY_POLL_SECONDS = 1.0


@contextmanager
def _measure_authority_phase(sample: dict[str, Any], phase: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        sample["phase_seconds"][phase] = max(0.0, time.monotonic() - started)


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
        if not math.isfinite(loop_interval) or loop_interval <= 0:
            raise ValueError("control-plane interval must be finite and positive")
        self._runtime = runtime
        self._runtime.authority_ready_generations = {}
        self._progress_loop = ProgressObservationLoop(runtime)
        self._instance_id = instance_id
        self._loop_interval = loop_interval
        self._started_at = started_at
        self._visible_gpus = list(available_gpus) if available_gpus is not None else None
        self._scheduler_wakeup = scheduler_wakeup
        self._registry_revision: int | None = None
        self._stop_event = threading.Event()
        self._supervisors: dict[str, AuthoritySupervisor] = {}
        self._supervisor_generations: dict[str, str | None] = {}
        self._eligibility_scans: dict[tuple[str, str], EvidenceScan] = {}
        self._outage_supervisors: dict[str, AuthoritySupervisor] = {}
        self._next_authority_project: str | None = None
        self._project_service_times: dict[tuple[str, str | None], tuple[float, float]] = {}
        self._authority_snapshot: dict[str, Any] | None = None
        self._heartbeat_snapshot: dict[str, Any] | None = None
        self._heartbeat_cycle_count = 0
        self._authority_cycle_count = 0
        self._authority_skipped_intervals = 0
        self._last_authority_diagnostics_at: float | None = None
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
        self._progress_loop.start()

    def stop(self) -> None:
        """Stop control loops before publishing the terminal machine snapshot."""
        self._stop_event.set()
        self._progress_loop.stop()
        for thread in (self._authority_thread, self._heartbeat_thread):
            if thread.is_alive():
                thread.join()
        for supervisor in self._supervisors.values():
            supervisor.close()
        for scan in self._eligibility_scans.values():
            scan.close()
        for supervisor in self._outage_supervisors.values():
            supervisor.close()

    def _supervised_bindings(self) -> list[ProjectBinding]:
        revision, registered = self._runtime.load_registry()
        if (
            self._registry_revision is not None
            and revision != self._registry_revision
            and self._scheduler_wakeup is not None
        ):
            self._scheduler_wakeup.set()
        self._registry_revision = revision
        # Registry discovery is machine-local. Shared status is checked inside
        # each project's turn, so it cannot delay selection of all projects.
        return registered

    def _has_local_evidence(self, binding: ProjectBinding, cfg: RootConfig, names: tuple[str, ...]) -> bool:
        for name in names:
            key = (binding.project_id, name)
            scan = self._eligibility_scans.get(key)
            if scan is None:
                scan = EvidenceScan(local_paths(cfg.runtime_root)[name], directories=name == "termination_decisions")
                self._eligibility_scans[key] = scan
            if scan.take(1).paths:
                return True
        return False

    def _has_local_process(self, binding: ProjectBinding, cfg: RootConfig) -> bool:
        return self._has_local_evidence(binding, cfg, ("processes",))

    def _has_local_convergence_evidence(self, binding: ProjectBinding, cfg: RootConfig) -> bool:
        return self._has_local_evidence(
            binding,
            cfg,
            ("registrations", "observations", "launch_intents", "termination_decisions"),
        )

    def _ordered_authority_bindings(self, bindings: list[ProjectBinding]) -> list[ProjectBinding]:
        """Rotate the leading project without treating order as write eligibility.

        Every selected binding is still visited each cycle. This reduces fixed
        positional disadvantage; it is NOT a record budget or I/O time limit.
        """
        if not bindings:
            self._next_authority_project = None
            return []
        project_ids = [binding.project_id for binding in bindings]
        try:
            start = project_ids.index(self._next_authority_project)
        except ValueError:
            start = 0
        ordered = bindings[start:] + bindings[:start]
        self._next_authority_project = ordered[(16 if len(ordered) > 16 else 1) % len(ordered)].project_id
        return ordered

    def _run_authority_cycle(self) -> float:
        started = time.monotonic()
        self._authority_cycle_count += 1
        sample: dict[str, Any] = {
            "protocol_version": 1,
            "diagnostic_only": True,
            "instance_id": self._instance_id,
            "sequence": self._authority_cycle_count,
            "observation_status": "completed",
            "project_count": None,
            "phase_seconds": {},
            "projects": [],
            "schedule": None,
        }
        authority_interval = self._loop_interval
        with _measure_authority_phase(sample, "reservation_before"):
            reserved_before = self._reserved_gpu_ids()
        try:
            with _measure_authority_phase(sample, "registry"):
                bindings = self._ordered_authority_bindings(self._supervised_bindings())
        except (OSError, RuntimeError, ValueError) as exc:
            self._runtime.authority_ready_generations.clear()
            for supervisor in self._supervisors.values():
                supervisor.cancel_pending_control()
            sample["observation_status"] = "registry_unavailable"
            sample["error_type"] = type(exc).__name__
            self._finish_authority_sample(sample, started, authority_interval)
            return authority_interval
        sample["project_count"] = len(bindings)
        supervised_ids = {binding.project_id for binding in bindings}
        generations = {(binding.project_id, binding.registration_generation) for binding in bindings}
        self._project_service_times = {
            key: value for key, value in self._project_service_times.items() if key in generations
        }
        for project_id in set(self._supervisors) - supervised_ids:
            self._supervisors.pop(project_id).close()
            self._supervisor_generations.pop(project_id, None)
            self._runtime.authority_ready_generations.pop(project_id, None)
        for key in list(self._eligibility_scans):
            if key[0] not in supervised_ids:
                self._eligibility_scans.pop(key).close()
        for project_id in set(self._outage_supervisors) - supervised_ids:
            self._outage_supervisors.pop(project_id).close()
        for binding in bindings[:16]:
            project_started = time.monotonic()
            eligibility_diagnostics = RuntimeDiagnostics()
            service_key = (binding.project_id, binding.registration_generation)
            previous_service = self._project_service_times.get(service_key)
            service_gap = None if previous_service is None else max(0.0, project_started - previous_service[0])
            maximum_gap = 0.0 if previous_service is None else max(previous_service[1], service_gap)
            self._project_service_times[service_key] = (project_started, maximum_gap)
            project_sample: dict[str, Any] = {
                "project_id": binding.project_id,
                "registration_generation": binding.registration_generation,
                "service_gap_seconds": service_gap,
                "maximum_service_gap_seconds": maximum_gap,
                "observation_status": "not_ticked",
                "phase_seconds": {},
                "eligibility_inventory_checks": 0,
            }
            try:
                with _measure_authority_phase(project_sample, "registration_status"):
                    registration_status = self._runtime.registration_status(binding)
                if registration_status["state"] == "superseded":
                    project_sample["observation_status"] = "superseded"
                    old_supervisor = self._supervisors.pop(binding.project_id, None)
                    if old_supervisor is not None:
                        old_supervisor.close()
                    self._supervisor_generations.pop(binding.project_id, None)
                    self._runtime.authority_ready_generations.pop(binding.project_id, None)
                    continue
                with _measure_authority_phase(project_sample, "configuration"):
                    cfg = _helpers._binding_config(self._runtime, binding)
                    # Idle supervisors may cache an older policy. Schedule from
                    # current shared policy before any guard shortens eligibility.
                    try:
                        policy = load_lease_policy(cfg)
                    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
                        # Cadence discovery must not bypass cached-policy local recovery.
                        policy = None
                    if policy is not None:
                        authority_interval = min(
                            authority_interval, policy.renew_interval_seconds, policy.ttl_seconds / 4
                        )
                has_capture = is_path_present(cfg.runtime_root / COMPLETION_FILE)
                if (
                    not binding.enabled
                    and not has_capture
                    and not self._has_local_process(binding, cfg)
                    and not self._has_local_convergence_evidence(binding, cfg)
                ):
                    project_sample["observation_status"] = "disabled_no_local_process"
                    continue
                with (
                    _measure_authority_phase(project_sample, "eligibility"),
                    activate_diagnostics(eligibility_diagnostics),
                ):
                    is_eligible = self._runtime.binding_write_eligible(binding, renew=True)
                # An eligible binding is supervised regardless of local inventory.
                # Do not enumerate that inventory merely to discard the answer.
                if not is_eligible:
                    with _measure_authority_phase(project_sample, "eligibility_inventory"):
                        project_sample["eligibility_inventory_checks"] = 1
                        has_local_process = has_capture or self._has_local_process(binding, cfg)
                        can_reactivate_idle = False
                        if not has_local_process and registration_status["state"] == "expired":
                            can_reactivate_idle = not self._has_local_convergence_evidence(binding, cfg)
                    if not has_local_process and not can_reactivate_idle:
                        project_sample["observation_status"] = "ineligible_no_local_process"
                        self._reconcile_local_exits(binding, project_sample)
                        continue
                supervisor = self._supervisors.get(binding.project_id)
                if supervisor is not None and self._supervisor_generations.get(binding.project_id) != (
                    binding.registration_generation
                ):
                    self._supervisors.pop(binding.project_id).close()
                    self._runtime.authority_ready_generations.pop(binding.project_id, None)
                    self._supervisor_generations.pop(binding.project_id, None)
                    supervisor = None
                if supervisor is None:
                    with _measure_authority_phase(project_sample, "startup_recovery"):
                        supervisor = _AuthoritySupervisor(
                            cfg,
                            reservation_runtime_root=self._runtime.root,
                            work_limit=256,
                            recovery_owner=recovery_owner(self._runtime, binding),
                        )
                        supervisor.recover_startup()
                    self._supervisors[binding.project_id] = supervisor
                    self._supervisor_generations[binding.project_id] = binding.registration_generation
                if not is_eligible:
                    with (
                        _measure_authority_phase(project_sample, "reactivation"),
                        activate_diagnostics(eligibility_diagnostics),
                    ):
                        reactivated = self._runtime.reactivate_binding(binding)
                    if not reactivated:
                        project_sample["observation_status"] = "reactivation_unavailable"
                        self._reconcile_local_exits(binding, project_sample)
                        continue
                with _measure_authority_phase(project_sample, "tick"):
                    project_sample["step_limit"] = supervisor.work_limit
                    supervisor.tick()
                    project_sample["work"] = supervisor.work_snapshot
                    if supervisor.work_snapshot.get("startup_complete"):
                        supervisor.work_limit = 64
                        previous = self._runtime.authority_ready_generations.get(binding.project_id)
                        self._runtime.authority_ready_generations[binding.project_id] = binding.registration_generation
                        if previous != binding.registration_generation and self._scheduler_wakeup is not None:
                            self._scheduler_wakeup.set()
                    else:
                        self._runtime.authority_ready_generations.pop(binding.project_id, None)
                        if supervisor.work_snapshot.get("discovery_mode") == "primary":
                            authority_interval = min(authority_interval, STARTUP_AUTHORITY_POLL_SECONDS)
                recovered_outage = self._outage_supervisors.pop(binding.project_id, None)
                if recovered_outage is not None:
                    recovered_outage.close()
                authority_interval = min(authority_interval, supervisor.renewal_interval_seconds)
                # tick() may handle a storage error internally. Returning is an
                # observation, not evidence that this project's authority is healthy.
                project_sample["observation_status"] = "tick_returned"
            except OSError as exc:
                project_sample["observation_status"] = "storage_error"
                project_sample["error_type"] = type(exc).__name__
                self._reconcile_local_exits(binding, project_sample)
            except (RuntimeError, ValueError) as exc:
                project_sample["observation_status"] = "runtime_error"
                project_sample["error_type"] = type(exc).__name__
            finally:
                if project_sample["observation_status"] != "tick_returned":
                    self._runtime.authority_ready_generations.pop(binding.project_id, None)
                    pending_supervisor = self._supervisors.get(binding.project_id)
                    if pending_supervisor is not None:
                        pending_supervisor.cancel_pending_control()
                project_sample["eligibility_operations"] = eligibility_diagnostics.snapshot()
                project_sample["work_seconds"] = max(0.0, time.monotonic() - project_started)
                sample["projects"].append(project_sample)
        with _measure_authority_phase(sample, "reservation_after"):
            reserved_after = self._reserved_gpu_ids()
        if (
            reserved_before is not None
            and reserved_after is not None
            and reserved_after < reserved_before
            and self._scheduler_wakeup is not None
        ):
            self._scheduler_wakeup.set()
        self._finish_authority_sample(sample, started, authority_interval)
        return authority_interval

    def _reconcile_local_exits(self, binding: Any, project_sample: dict[str, Any]) -> None:
        """Free verified finished local capacity without granting shared write eligibility."""
        try:
            with _measure_authority_phase(project_sample, "local_exit_reconciliation"):
                local_cfg = RootConfig(
                    binding.shared_root,
                    binding.shared_root.parent,
                    binding.machine_name,
                    self._runtime.project_paths(binding.project_id)["root"],
                )
                local_supervisor = self._outage_supervisors.get(binding.project_id)
                if local_supervisor is None:
                    local_supervisor = _AuthoritySupervisor(local_cfg, reservation_runtime_root=self._runtime.root)
                    self._outage_supervisors[binding.project_id] = local_supervisor
                local_supervisor.reconcile_local_exit_evidence(limit=8)
            project_sample["local_reconciliation"] = "returned"
        except (OSError, RuntimeError, ValueError) as local_exc:
            project_sample["local_reconciliation"] = "unavailable"
            project_sample["local_error_type"] = type(local_exc).__name__

    def _finish_authority_sample(self, sample: dict[str, Any], started: float, interval: float) -> None:
        sample["sampled_at"] = utc_now()
        sample["work_seconds"] = max(0.0, time.monotonic() - started)
        sample["interval_seconds"] = interval
        # Publish only complete snapshots. The heartbeat thread never observes
        # lists or dictionaries while the authority thread is mutating them.
        self._authority_snapshot = sample

    def _publish_authority_diagnostics(self) -> None:
        """Best-effort local diagnostics, never consulted for authority decisions."""
        snapshot = self._authority_snapshot
        if snapshot is None:
            return
        now = time.monotonic()
        previous = self._last_authority_diagnostics_at
        if previous is not None and now - previous < max(1.0, self._loop_interval):
            return
        # Rate-limit failed writes too, avoiding a new storage-outage hot loop.
        self._last_authority_diagnostics_at = now
        try:
            atomic_replace(
                self._runtime.root / "authority_control_plane.json",
                {"authority_control_plane": {**snapshot, "heartbeat": self._heartbeat_snapshot}},
            )
        except OSError:
            pass

    def _reserved_gpu_ids(self) -> set[int] | None:
        """Read occupancy without allowing a transient storage error to stop supervision."""
        try:
            return {
                gpu_id
                for item in reservation_snapshot(self._runtime.root).reservations
                for gpu_id in item.get("gpu_ids", [])
            }
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return None

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
        diagnostics = RuntimeDiagnostics()
        self._heartbeat_cycle_count += 1
        started = time.monotonic()
        status = "raised"
        try:
            with activate_diagnostics(diagnostics):
                status = self._publish_project_heartbeats()
        finally:
            self._heartbeat_snapshot = {
                "sequence": self._heartbeat_cycle_count,
                "sampled_at": utc_now(),
                "work_seconds": max(0.0, time.monotonic() - started),
                "observation_status": status,
                "operations": diagnostics.snapshot(),
            }
            # Publish liveness first. Optional local diagnostics must not sit
            # ahead of this cycle's project heartbeat writes.
            self._publish_authority_diagnostics()

    def _publish_project_heartbeats(self) -> str:
        self._refresh_visible_gpus()
        try:
            bindings = self._supervised_bindings()
        except (OSError, RuntimeError, ValueError):
            return "registry_unavailable"
        readable: dict[str, RootConfig] = {}
        readable_bindings: dict[str, ProjectBinding] = {}
        for binding in bindings:
            try:
                if not self._runtime.binding_write_eligible(binding, renew=True):
                    diagnostic_increment("heartbeat.ineligible_projects")
                    continue
                readable[binding.project_id] = _helpers._binding_config(self._runtime, binding)
                readable_bindings[binding.project_id] = binding
            except (OSError, RuntimeError, ValueError):
                diagnostic_increment("heartbeat.unavailable_projects")
                continue

        @contextmanager
        def write_guard(project_id: str) -> Iterator[bool]:
            with self._runtime.binding_write_guard(readable_bindings[project_id]) as is_eligible:
                if not is_eligible:
                    diagnostic_increment("heartbeat.write_guard_rejected_projects")
                yield is_eligible

        try:
            reservations = list(reservation_snapshot(self._runtime.root).reservations)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            # Persisted capacity records can have invalid container/field shapes.
            # Retain them and retry next cycle; do not advertise empty capacity.
            return "capacity_unavailable"
        try:
            _publish_project_snapshots(
                readable,
                instance_id=self._instance_id,
                pid=_read_pid(self._runtime),
                visible=self._visible_gpus or [],
                reservations=reservations,
                heartbeat_interval_seconds=self._loop_interval,
                started_at=self._started_at,
                write_guard=write_guard,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return "publication_unavailable"
        return "returned"

    def _run_authority_loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop_event.is_set():
            wait_started = time.monotonic()
            remaining = max(0.0, deadline - wait_started)
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            cycle_started = time.monotonic()
            interval = self._run_authority_cycle()
            finished = time.monotonic()
            next_deadline, skipped = advance_deadline(deadline, interval, finished)
            self._authority_skipped_intervals += skipped
            self._authority_snapshot = {
                **(self._authority_snapshot or {}),
                "schedule": {
                    "requested_wait_seconds": remaining,
                    "observed_wait_seconds": max(0.0, cycle_started - wait_started),
                    "start_lateness_seconds": max(0.0, cycle_started - deadline),
                    "cycle_seconds": max(0.0, finished - cycle_started),
                    "skipped_intervals": skipped,
                    "skipped_intervals_total": self._authority_skipped_intervals,
                },
            }
            deadline = next_deadline

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
            deadline, _ = advance_deadline(deadline, self._loop_interval, time.monotonic())
