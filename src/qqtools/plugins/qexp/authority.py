"""Agent-owned runtime authority for locally registered qexp processes."""

from __future__ import annotations

import copy
import shutil
import uuid
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config_types import RootConfig
from .lease import AuthorityResolutionOutcome, LeasePolicy, LeaseRenewalOutcome, holder_safe_deadline, load_lease_policy
from .lifecycle import TerminalTransition, commit_terminal_transition_locked, dispatch_task_lifecycle_hooks_noexcept
from .runtime.authority_scan import EvidenceScan
from .runtime.claims import archive_claim, reconcile_claim_archives
from .runtime.paths import attempt_path, local_paths
from .runtime.records import AttemptRecord, utc_now
from .runtime.resources.reservations import (
    ReservationIdentity,
    has_reservation,
    release,
    release_if_matches,
    reservation_snapshot,
)
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.tasks import load_task, save_task
from .runtime.termination import (
    advance_signals,
    attempt_control_lock,
    commit_local_unavailable,
    commit_signal,
    create_decision,
    send_signals,
    termination_check_steps,
    update_decision,
)
from .scheduler import authority_locks, commit_shared_termination, renew_attempt_lease, resolve_execution_authority


class AuthoritySupervisor:
    """The sole local writer of live-process authority, termination, and terminal truth."""

    def __init__(
        self,
        cfg: RootConfig,
        *,
        reservation_runtime_root: Path | None = None,
        work_limit: int | None = None,
    ) -> None:
        if work_limit is not None and (type(work_limit) is not int or not 1 <= work_limit <= 256):
            raise ValueError("authority work limit must be an integer from 1 to 256")
        self.work_limit = work_limit
        self.work_snapshot: dict[str, object] = {}
        self.metrics: dict[str, int | float] = {}
        self.cfg = cfg
        self.reservation_runtime_root = reservation_runtime_root or cfg.runtime_root
        self._last_renewal: dict[str, float] = {}
        self._termination_deadlines: dict[tuple[str, str], float] = {}
        self._failures: dict[str, int] = {}
        self._states: dict[str, str] = {}
        self._lease_expiries: dict[str, str] = {}
        self._policy: LeasePolicy | None = None
        self._policy_load_failed = False
        self._refresh_policy()
        self._work = None
        capacity_paths = local_paths(self.reservation_runtime_root)
        self._capacity_scans = (EvidenceScan(capacity_paths["active"]), EvidenceScan(capacity_paths["cpu_active"]))
        self._exit_scan = EvidenceScan(local_paths(cfg.runtime_root)["observations"])

    def tick_bounded(self, limit: int = 64) -> dict[str, object]:
        """Advance resumable discovery and supervision without a full inventory pass."""
        if self._work is None:
            from .authority_work import AuthorityWork

            self._work = AuthorityWork(self)
        return self._work.tick(limit)

    def cancel_pending_control(self) -> None:
        """Drop a locked inventory proof when this binding cannot be serviced."""
        if self._work is not None:
            self._work.cancel_pending_control()

    def close(self) -> None:
        """Release advisory cursors after this supervisor is no longer scheduled."""
        if self._work is not None:
            self._work.close()
        self._exit_scan.close()
        for scan in self._capacity_scans:
            scan.close()

    def recover_startup(self) -> None:
        if self.work_limit is not None:
            # Startup uses the same resumable evidence lanes as ordinary service.
            # Admission waits for their first successful complete sweeps.
            from .authority_work import AuthorityWork

            if self._work is None:
                self._work = AuthorityWork(self)
            return
        try:
            self._materialize_registrations()
        except OSError:
            self.reconcile_local_exit_evidence()
            raise
        for path in iter_json(local_paths(self.cfg.runtime_root)["processes"]):
            process = read_json(path).get("process", {})
            if process.get("protocol_version") != 1:
                continue
            task_id = process.get("task_id")
            if isinstance(task_id, str):
                self._reconcile_orphaned_process(process, load_task(self.cfg, task_id))
        for directory in local_paths(self.cfg.runtime_root)["termination_decisions"].glob("*"):
            if directory.is_dir():
                for decision in iter_json(directory):
                    value = read_json(decision).get("termination_decision", {})
                    if value.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent"}:
                        with attempt_control_lock(self.cfg, value["attempt_id"]):
                            self._send_signals(value["attempt_id"], value["decision_id"])

    def tick(self) -> None:
        if self.work_limit is not None:
            self.work_snapshot = self.tick_bounded(self.work_limit)
            return
        try:
            self._remove_terminal_evidence()
            self._materialize_registrations()
        except OSError:
            self.reconcile_local_exit_evidence()
            return
        for path in iter_json(local_paths(self.cfg.runtime_root)["processes"]):
            process = read_json(path).get("process", {})
            if process.get("protocol_version") != 1:
                continue
            try:
                self._supervise(process)
            except OSError as exc:
                self._record_diagnostic(process, "shared_storage_unavailable", exc)
                self._mark_shared_unavailable(process)
                self._release_finished_local_capacity(process)

    def reconcile_local_exit_evidence(self, *, limit: int | None = None) -> None:
        """Release verified finished local occupancy without reading shared Task truth."""
        observations = (
            iter_json(local_paths(self.cfg.runtime_root)["observations"])
            if limit is None
            else self._exit_scan.take(limit).paths
        )
        for observation in observations:
            self._reconcile_local_exit_observation(observation, is_bounded=limit is not None)

    def _reconcile_local_exit_observation(
        self,
        observation: Path,
        *,
        is_bounded: bool = False,
        reservation_identity: ReservationIdentity | None = None,
    ) -> None:
        paths = local_paths(self.cfg.runtime_root)
        attempt_id = observation.stem
        try:
            manifest = paths["processes"] / observation.name
            if manifest.exists():
                process = read_json(manifest)["process"]
            else:
                process = read_json(paths["registrations"] / observation.name)["process_registration"]
            if not isinstance(process, dict):
                raise ValueError("local process evidence must be an object")
            if process.get("attempt_id") == attempt_id:
                self._release_finished_local_capacity(
                    process, is_bounded=is_bounded, reservation_identity=reservation_identity
                )
        except (OSError, KeyError, TypeError, ValueError):
            self._record_diagnostic({"attempt_id": attempt_id}, "local_capacity_reconciliation_unavailable")

    def _release_finished_local_capacity(
        self,
        process: dict[str, object],
        *,
        is_bounded: bool = False,
        reservation_identity: ReservationIdentity | None = None,
    ) -> None:
        """Retain recovery evidence while releasing an identity-verified absent process."""
        from .scheduler import _is_process_group_alive, _process_start_time_ticks

        task_id, attempt_id = process.get("task_id"), process.get("attempt_id")
        if not isinstance(task_id, str) or not isinstance(attempt_id, str):
            return
        paths = local_paths(self.cfg.runtime_root)
        try:
            registration = read_json(paths["registrations"] / f"{attempt_id}.json")["process_registration"]
            if not isinstance(registration, dict):
                raise ValueError("local process registration must be an object")
            for key in ("task_id", "attempt_id", "process_group_id", "process_group_start_time_ticks"):
                if registration.get(key) is None or registration.get(key) != process.get(key):
                    return
            group = registration["process_group_id"]
            if not isinstance(group, int) or group <= 0:
                return
            if _process_start_time_ticks(group) is not None or _is_process_group_alive(group):
                return
            is_valid, _code = self._read_exit_observation(
                paths["observations"] / f"{attempt_id}.json", task_id, attempt_id, process
            )
            if not is_valid:
                return
            if reservation_identity is None and (is_bounded or self._work is not None):
                for identity in self._local_capacity_page(attempt_id):
                    if identity.project_id not in {None, self.cfg.runtime_root.name}:
                        continue
                    candidate_id = identity.attempt_id
                    if not candidate_id or candidate_id in {".", ".."} or "/" in candidate_id or "\\" in candidate_id:
                        continue
                    self._reconcile_local_exit_observation(
                        paths["observations"] / f"{candidate_id}.json", reservation_identity=identity
                    )
                return
            identities = (
                (reservation_identity,)
                if reservation_identity is not None
                else (
                    ReservationIdentity.from_record(record)
                    for record in reservation_snapshot(self.reservation_runtime_root).reservations
                )
            )
            for identity in identities:
                if identity.project_id is not None and identity.project_id != self.cfg.runtime_root.name:
                    continue
                if (
                    identity.task_id == task_id
                    and identity.attempt_id == attempt_id
                    and identity.fencing_token == process.get("fencing_token")
                ):
                    release_if_matches(self.reservation_runtime_root, identity, "local_process_exited")
        except (OSError, KeyError, TypeError, ValueError):
            self._record_diagnostic(process, "local_capacity_reconciliation_unavailable")

    def _local_capacity_page(self, attempt_id: str) -> Iterable[ReservationIdentity]:
        for scan in self._capacity_scans:
            try:
                page = scan.take(4)
            except OSError:
                self._record_diagnostic({"attempt_id": attempt_id}, "local_reservation_unreadable")
                continue
            for path in page.paths:
                try:
                    reservation = read_json(path)["reservation"]
                    if not isinstance(reservation, dict):
                        raise ValueError("local reservation must be an object")
                    yield ReservationIdentity.from_record(reservation)
                except (OSError, KeyError, TypeError, ValueError):
                    self._record_diagnostic({"attempt_id": attempt_id}, "local_reservation_unreadable")

    @property
    def renewal_interval_seconds(self) -> float:
        """Return the current policy interval used to schedule the next supervision pass."""
        policy = self._policy or self._refresh_policy()
        return policy.renew_interval_seconds if policy is not None else 1.0

    def _replace_local_if_changed(self, path: Path, value: dict[str, object]) -> None:
        try:
            if read_json(path) == value:
                self.metrics["equivalent_local_writes_avoided"] = (
                    self.metrics.get("equivalent_local_writes_avoided", 0) + 1
                )
                return
        except (OSError, ValueError):
            pass
        atomic_replace(path, value)

    def _observe_latency(self, name: str, started_at: object) -> None:
        if not isinstance(started_at, str):
            return
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if started.tzinfo is None:
                return
            seconds = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        except (ValueError, OverflowError):
            return
        self.metrics[f"{name}.count"] = self.metrics.get(f"{name}.count", 0) + 1
        self.metrics[f"{name}.last_seconds"] = seconds
        self.metrics[f"{name}.maximum_seconds"] = max(self.metrics.get(f"{name}.maximum_seconds", 0), seconds)

    def _refresh_policy(self) -> LeasePolicy | None:
        try:
            self._policy = load_lease_policy(self.cfg)
            self._policy_load_failed = False
            policy_value = asdict(self._policy)
            policy_value["clock_provider_priority"] = list(self._policy.clock_provider_priority)
            self._replace_local_if_changed(
                local_paths(self.cfg.runtime_root)["lease_policy_cache"],
                {
                    "lease_policy": policy_value,
                },
            )
        except (OSError, RuntimeError, ValueError):
            self._policy_load_failed = True
            try:
                cached = read_json(local_paths(self.cfg.runtime_root)["lease_policy_cache"])
                self._policy = LeasePolicy(**cached["lease_policy"])
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                pass
        return self._policy

    def _record_diagnostic(self, process: dict[str, object], reason: str, error: Exception | None = None) -> None:
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        value: dict[str, object] = {"attempt_id": attempt_id, "reason": reason, "at": utc_now()}
        if error is not None:
            value["error_type"] = type(error).__name__
            value["error"] = str(error)
        try:
            atomic_replace(
                local_paths(self.cfg.runtime_root)["authority_diagnostics"] / f"{attempt_id}.json",
                {"authority_diagnostic": value},
            )
        except OSError:
            pass

    def _set_authority_state(self, process: dict[str, object], state: str) -> None:
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        self._states[attempt_id] = state
        process["authority_state"] = state
        try:
            self._replace_local_if_changed(
                local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json", {"process": process}
            )
        except OSError:
            pass

    def _mark_shared_unavailable(self, process: dict[str, object]) -> None:
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        if process.get("authority_mode") == "holder_bound":
            self._set_authority_state(process, "local_safe")
            return
        policy = self._policy
        expires = self._lease_expiries.get(attempt_id) or process.get("lease_expires_at")
        holder_bound = process.get("clock_error_bound_seconds")
        if (
            policy
            and isinstance(expires, str)
            and isinstance(holder_bound, (int, float))
            and datetime.now(timezone.utc) >= holder_safe_deadline(expires, holder_bound)
        ):
            self._set_authority_state(process, "isolated")
        else:
            self._set_authority_state(process, "suspect")

    def _publish_running(self, registration: dict[str, object], manifest: Path) -> None:
        task_id = registration.get("task_id")
        attempt_id = registration.get("attempt_id")
        token = registration.get("fencing_token")
        if not isinstance(task_id, str) or not isinstance(attempt_id, str) or not isinstance(token, int):
            return
        with attempt_control_lock(self.cfg, attempt_id):
            task = load_task(self.cfg, task_id)
            with authority_locks(self.cfg, task):
                task = load_task(self.cfg, task_id)
                claim = task.claim_control.get("active_claim") or {}
                number = task.attempt_control.get("current_attempt_number")
                if not isinstance(number, int):
                    return
                try:
                    attempt_file = attempt_path(self.cfg.shared_root, task_id, number)
                    stored_attempt = read_json(attempt_file)
                    attempt = AttemptRecord.from_dict(stored_attempt)
                    original_attempt_value = copy.deepcopy(stored_attempt)
                except (FileNotFoundError, KeyError, ValueError):
                    return
                if (
                    claim.get("attempt_id") != attempt_id
                    or claim.get("fencing_token") != token
                    or claim.get("machine_name") != self.cfg.machine_name
                    or attempt.attempt_id != attempt_id
                    or attempt.current_fencing_token != token
                    or attempt.machine_name != self.cfg.machine_name
                ):
                    return
                if claim.get("launch_state") not in {"starting", "running"} or attempt.phase not in {
                    "starting",
                    "running",
                }:
                    return
                for key in (
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                ):
                    value = registration.get(key)
                    if value is not None:
                        existing = attempt.process.get(key)
                        if existing is not None and existing != value:
                            return
                        attempt.process[key] = value
                attempt.process["local_process_manifest"] = str(manifest)
                created_at = registration.get("process_created_at")
                if not isinstance(created_at, str):
                    return
                existing_created = attempt.timestamps.get("process_created_at")
                if existing_created is not None and existing_created != created_at:
                    return
                attempt.timestamps["process_created_at"] = created_at
                if attempt.timestamps.get("running_at") is None:
                    attempt.timestamps["running_at"] = utc_now()
                was_running = attempt.phase == "running"
                attempt.phase = "running"
                attempt_value = attempt.to_dict()
                if attempt_value != original_attempt_value:
                    atomic_replace(attempt_file, attempt_value)
                if not was_running:
                    self._observe_latency("registration_to_running", registration.get("process_created_at"))
                if claim.get("launch_state") != "running":
                    claim["launch_state"] = "running"
                    task.meta["revision"] += 1
                    task.meta["updated_at"] = utc_now()
                    save_task(self.cfg, task)

    def _materialize_registrations(
        self,
        *,
        registration_paths: Iterable[Path] | None = None,
        should_materialize_intents: bool = True,
    ) -> None:
        if should_materialize_intents:
            self._materialize_unverified_intents()
        paths = registration_paths
        if paths is None:
            paths = iter_json(local_paths(self.cfg.runtime_root)["registrations"])
        for path in paths:
            self._materialize_registration(path)

    def _materialize_registration(self, path: Path) -> None:
        registration = read_json(path).get("process_registration", {})
        if not isinstance(registration, dict):
            raise ValueError("local process registration must be an object")
        if registration.get("protocol_version") != 1:
            return
        attempt_id = registration.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        if self._work is not None and self._work.is_control_pending(attempt_id):
            return
        manifest = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
        if not manifest.exists():
            value = dict(registration)
            value.update(
                {
                    "observed_state": "running",
                    "supervisor": "agent",
                    "authority_state": "healthy",
                    "created_by": "agent",
                }
            )
            atomic_replace(manifest, {"process": value})
            self._observe_latency("registration_to_manifest", registration.get("process_created_at"))
        self._publish_running(registration, manifest)

    def _materialize_unverified_intents(self) -> None:
        for path in iter_json(local_paths(self.cfg.runtime_root)["launch_intents"]):
            self._materialize_unverified_intent(path)

    def _materialize_unverified_intent(self, path: Path) -> None:
        intent = read_json(path).get("launch_intent", {})
        if not isinstance(intent, dict):
            raise ValueError("local launch intent must be an object")
        if intent.get("protocol_version") != 1:
            return
        attempt_id = intent.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        if self._work is not None and self._work.is_control_pending(attempt_id):
            return
        registration = local_paths(self.cfg.runtime_root)["registrations"] / f"{attempt_id}.json"
        manifest = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
        if manifest.exists():
            return
        if registration.exists():
            if self._work is not None:
                self._materialize_registration(registration)
            return
        if self._wrapper_matches(intent):
            return
        value = dict(intent)
        value.update(
            {
                "observed_state": "launch_unverifiable",
                "supervisor": "agent",
                "authority_state": "isolated",
                "created_by": "agent",
            }
        )
        atomic_replace(manifest, {"process": value})
        self._record_diagnostic(value, "launch_registration_missing")

    @staticmethod
    def _wrapper_matches(intent: dict[str, object]) -> bool:
        pid = intent.get("wrapper_pid")
        start = intent.get("wrapper_start_time_ticks")
        if not isinstance(pid, int) or not isinstance(start, int):
            return False
        try:
            from .scheduler import _process_start_time_ticks

            return _process_start_time_ticks(pid) == start
        except (FileNotFoundError, OSError, ValueError):
            return False

    def _supervise(self, process: dict[str, object]) -> None:
        task_id = process.get("task_id")
        attempt_id = process.get("attempt_id")
        token = process.get("fencing_token")
        if not isinstance(task_id, str) or not isinstance(attempt_id, str) or not isinstance(token, int):
            return
        if self._has_terminal_attempt(task_id, attempt_id):
            self._remove_attempt_evidence(attempt_id)
            return
        if self._reconcile_terminal_accounting(task_id, attempt_id):
            if self._has_terminal_attempt(task_id, attempt_id):
                self._remove_attempt_evidence(attempt_id)
            return
        if process.get("observed_state") == "launch_unverifiable":
            self._record_diagnostic(process, "launch_unverifiable")
            return
        task = load_task(self.cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != token:
            self._reconcile_orphaned_process(process, task)
            return
        if claim.get("machine_name") != self.cfg.machine_name:
            self._supervise_renamed_execution(process, task_id, attempt_id, token)
            return
        if self._work is not None:
            self._work.supervise(process)
            return
        decision = None
        for decision_file in iter_json(local_paths(self.cfg.runtime_root)["termination_decisions"] / attempt_id):
            value = read_json(decision_file).get("termination_decision", {})
            if not isinstance(value, dict):
                raise ValueError("local termination decision must be an object")
            if value.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent", "confirmed"}:
                decision = value
                break
        result = self._supervision_result(process, decision)
        if result is not None:
            exit_code, was_terminated = result
            self._finalize(task_id, attempt_id, token, exit_code, was_terminated=was_terminated)

    def supervision_steps(self, process: dict[str, object]) -> Generator[None, None, None]:
        """Retain a termination fence across bounded inventory pages and renewal."""
        task_id, attempt_id, token = process["task_id"], process["attempt_id"], process["fencing_token"]
        with attempt_control_lock(self.cfg, attempt_id, blocking=False) as acquired:
            if not acquired:
                self._record_diagnostic(process, "supervision_lock_busy")
                return
            decision = yield from termination_check_steps(self.cfg, attempt_id)
            path = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
            try:
                current = read_json(path).get("process")
            except FileNotFoundError:
                return
            if not isinstance(current, dict):
                raise ValueError("local process manifest must be an object")
            if any(current.get(key) != process.get(key) for key in ("task_id", "attempt_id", "fencing_token")):
                return
            task = load_task(self.cfg, task_id)
            claim = task.claim_control.get("active_claim") or {}
            if (
                claim.get("attempt_id") != attempt_id
                or claim.get("fencing_token") != token
                or claim.get("machine_name") != self.cfg.machine_name
            ):
                return
            result = self._supervision_result(current, decision, has_attempt_lock=True)
        # Terminal hooks may invoke qexp again. They must not inherit our local
        # control lock; _finalize independently fences the shared transition.
        if result is not None:
            exit_code, was_terminated = result
            self._finalize(task_id, attempt_id, token, exit_code, was_terminated=was_terminated)

    def _supervision_result(
        self,
        process: dict[str, object],
        decision: dict[str, object] | None,
        *,
        has_attempt_lock: bool = False,
    ) -> tuple[int | None, bool] | None:
        task_id, attempt_id, token = process["task_id"], process["attempt_id"], process["fencing_token"]
        observation = local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json"
        if decision is not None:
            with nullcontext() if has_attempt_lock else attempt_control_lock(self.cfg, attempt_id):
                if decision.get("state") == "pending":
                    # A crash after commitment but before signal_committed must
                    # complete termination rather than permit a fresh renewal.
                    commit_signal(self.cfg, attempt_id, decision["decision_id"])
                result = (
                    decision
                    if decision.get("state") == "confirmed"
                    else self._send_signals(attempt_id, decision["decision_id"])
                )
            if result.get("state") != "confirmed":
                return None
            exit_code = None
            if observation.exists():
                is_valid, exit_code = self._read_exit_observation(observation, task_id, attempt_id, process)
                if not is_valid:
                    return None
            return exit_code, bool(result.get("signal_attempts"))
        if observation.exists():
            is_valid, exit_code = self._read_exit_observation(observation, task_id, attempt_id, process)
            return (exit_code, False) if is_valid else None
        from .scheduler import _is_process_group_alive, _process_start_time_ticks

        group = process.get("process_group_id")
        if (
            isinstance(group, int)
            and group > 0
            and process.get("process_group_start_time_ticks") is not None
            and not self._wrapper_matches(process)
            and _process_start_time_ticks(group) is None
            and not _is_process_group_alive(group)
        ):
            self._record_diagnostic(process, "exit_observation_missing")
            return None
        self._renew_or_isolate(task_id, attempt_id, token, process, has_attempt_lock=has_attempt_lock)
        return None

    def _supervise_renamed_execution(
        self,
        process: dict[str, object],
        task_id: str,
        attempt_id: str,
        token: int,
    ) -> None:
        """Preserve and reconcile local execution launched under a replaced logical name."""
        observation = local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json"
        if not observation.exists():
            self._set_authority_state(process, "retained_execution")
            return
        is_valid, exit_code = self._read_exit_observation(observation, task_id, attempt_id, process)
        if not is_valid:
            return
        self._finalize(
            task_id,
            attempt_id,
            token,
            exit_code,
            was_terminated=False,
            is_renamed_execution=True,
        )

    def _reconcile_terminal_accounting(self, task_id: str, attempt_id: str) -> bool:
        """Retry reservation/accounting effects after terminal truth already committed."""
        try:
            task = load_task(self.cfg, task_id)
            if task.state.get("projection") not in {"succeeded", "failed", "cancelled"}:
                return False
            number = task.attempt_control.get("current_attempt_number")
            if not isinstance(number, int):
                return False
            attempt = AttemptRecord.from_dict(read_json(attempt_path(self.cfg.shared_root, task_id, number)))
            if attempt.attempt_id != attempt_id or attempt.phase not in {"succeeded", "failed", "cancelled"}:
                return False
            reason = attempt.result.get("reason") or "terminal_accounting_reconciliation"
            if has_reservation(self.reservation_runtime_root, attempt.reservation_id):
                release(self.reservation_runtime_root, attempt.reservation_id, reason)
            return True
        except (FileNotFoundError, OSError, RuntimeError, KeyError, TypeError, ValueError):
            self._record_diagnostic(
                {"attempt_id": attempt_id},
                "terminal_accounting_reconciliation_failed",
            )
            return False

    def _read_exit_observation(
        self, path: Path, task_id: str, attempt_id: str, process: dict[str, object]
    ) -> tuple[bool, int | None]:
        """Validate immutable exit identity before publishing terminal Task truth."""
        try:
            record = read_json(path)
            observation = record.get("exit_observation") if isinstance(record, dict) else None
        except (OSError, ValueError, TypeError):
            self._record_diagnostic(process, "exit_observation_unreadable")
            return False, None
        if not isinstance(observation, dict):
            self._record_diagnostic(process, "exit_observation_unreadable")
            return False, None
        if observation.get("protocol_version", 1) != 1:
            self._record_diagnostic(process, "exit_observation_protocol_unsupported")
            return False, None
        if observation.get("attempt_id") != attempt_id or observation.get("task_id") not in {
            None,
            task_id,
        }:
            self._record_diagnostic(process, "exit_observation_identity_mismatch")
            return False, None
        code = observation.get("observed_exit_code")
        if type(code) is not int:
            self._record_diagnostic(process, "exit_observation_code_invalid")
            return False, None
        return True, code

    def _reconcile_orphaned_process(self, process: dict[str, object], task: object) -> None:
        """Recover or finalize one process whose lease was archived while offline."""
        task_id = process.get("task_id")
        attempt_id = process.get("attempt_id")
        if not isinstance(task_id, str) or not isinstance(attempt_id, str):
            return
        if getattr(task, "state", {}).get("projection") == "running":
            self._repair_recovered_manifest(process, task_id, attempt_id)
            return
        if getattr(task, "state", {}).get("projection") != "blocked":
            self._record_diagnostic(process, "attempt_authority_superseded")
            return
        claim = getattr(task, "claim_control", {}).get("active_claim") or {}
        if claim:
            self._record_diagnostic(process, "attempt_authority_superseded")
            return
        number = getattr(task, "attempt_control", {}).get("current_attempt_number")
        if not isinstance(number, int):
            return
        try:
            attempt = AttemptRecord.from_dict(read_json(attempt_path(self.cfg.shared_root, task_id, number)))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            self._record_diagnostic(process, "orphan_attempt_unreadable")
            return
        if attempt.attempt_id != attempt_id or attempt.phase not in {
            "orphaned",
            "running",
            "succeeded",
            "failed",
            "cancelled",
        }:
            return

        paths = local_paths(self.cfg.runtime_root)
        observation_path_value = paths["observations"] / f"{attempt_id}.json"
        if observation_path_value.exists():
            try:
                is_valid, exit_code = self._read_exit_observation(observation_path_value, task_id, attempt_id, process)
                if not is_valid:
                    return
                from .scheduler import finalize_orphaned_attempt

                if finalize_orphaned_attempt(
                    self.cfg,
                    task_id,
                    attempt_id,
                    attempt.current_fencing_token,
                    exit_code=exit_code,
                    was_terminated=bool(getattr(task, "control", {}).get("terminate_running")),
                    reservation_runtime_root=self.reservation_runtime_root,
                ):
                    process.update(
                        {
                            "observed_state": "exited",
                            "observed_exit_code": exit_code,
                            "observed_exited_at": utc_now(),
                        }
                    )
                    atomic_replace(paths["processes"] / f"{attempt_id}.json", {"process": process})
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                self._record_diagnostic(process, "orphan_terminal_reconciliation_failed")
            return

        try:
            from .runtime.attempt_recovery import recover_running_attempt
            from .scheduler import _process_evidence_state

            if _process_evidence_state(attempt, process) != "alive":
                return
            if self._work is not None:
                self._work.recover(process)
                return
            recovered_token = recover_running_attempt(
                self.cfg,
                task_id,
                attempt_id,
                process["fencing_token"],
                manifest=process,
                reservation_runtime_root=self.reservation_runtime_root,
            )
            if recovered_token is not None:
                process["fencing_token"] = recovered_token
                atomic_replace(paths["processes"] / f"{attempt_id}.json", {"process": process})
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._record_diagnostic(process, "orphan_live_recovery_failed")

    def _repair_recovered_manifest(self, process: dict[str, object], task_id: str, attempt_id: str) -> None:
        """Finish local publication after shared recovery committed."""
        from .scheduler import _process_evidence_state

        with attempt_control_lock(self.cfg, attempt_id):
            task = load_task(self.cfg, task_id)
            with authority_locks(self.cfg, task):
                task = load_task(self.cfg, task_id)
                claim = task.claim_control.get("active_claim") or {}
                number = task.attempt_control.get("current_attempt_number")
                if (
                    task.state.get("projection") != "running"
                    or claim.get("attempt_id") != attempt_id
                    or claim.get("machine_name") != self.cfg.machine_name
                    or not isinstance(number, int)
                ):
                    return
                attempt = AttemptRecord.from_dict(read_json(attempt_path(self.cfg.shared_root, task_id, number)))
                old_token = process.get("fencing_token")
                if (
                    attempt.attempt_id != attempt_id
                    or attempt.phase != "running"
                    or attempt.authority_mode != "bounded_lease"
                    or attempt.current_fencing_token != claim.get("fencing_token")
                    or not isinstance(old_token, int)
                    or old_token >= attempt.current_fencing_token
                    or old_token not in attempt.token_history
                ):
                    return
                observation_path = local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json"
                has_valid_observation = False
                if observation_path.exists():
                    has_valid_observation, _ = self._read_exit_observation(
                        observation_path, task_id, attempt_id, process
                    )
                if not has_valid_observation and _process_evidence_state(attempt, process) != "alive":
                    return
                process.update(
                    fencing_token=attempt.current_fencing_token,
                    recovered_at=utc_now(),
                    observed_state="exited" if has_valid_observation else "running",
                    supervisor="agent",
                )
                path = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
                atomic_replace(path, {"process": process})

    def _has_terminal_attempt(self, task_id: str, attempt_id: str) -> bool:
        try:
            task = load_task(self.cfg, task_id)
            if task.state.get("projection") not in {"succeeded", "failed", "cancelled"}:
                return False
            if task.claim_control.get("active_claim"):
                return False
            number = task.attempt_control.get("current_attempt_number")
            if not isinstance(number, int):
                return False
            attempt = AttemptRecord.from_dict(read_json(attempt_path(self.cfg.shared_root, task_id, number)))
            if attempt.attempt_id != attempt_id or attempt.phase not in {"succeeded", "failed", "cancelled"}:
                return False
            manifest = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
            try:
                process = read_json(manifest).get("process")
            except FileNotFoundError:
                # Registration-only completion need not have created a manifest.
                pass
            except (OSError, ValueError) as error:
                self._record_diagnostic({"attempt_id": attempt_id}, "terminal_manifest_unreadable", error)
                return False
            else:
                if not (
                    isinstance(process, dict)
                    and process.get("task_id") == task_id
                    and process.get("attempt_id") == attempt_id
                    and process.get("fencing_token") == attempt.current_fencing_token
                ):
                    self._record_diagnostic({"attempt_id": attempt_id}, "terminal_manifest_unreadable")
                    return False
            archive = self.cfg.shared_root / "claims" / "archive" / task_id / f"{attempt.current_fencing_token}.json"
            if (
                not archive.exists()
                and not (
                    self.cfg.shared_root / "claims" / "pending" / task_id / f"{attempt.current_fencing_token}.json"
                ).exists()
            ):
                return False
            try:
                if has_reservation(self.reservation_runtime_root, attempt.reservation_id):
                    return False
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False
            if self._work is not None:
                return self._work.reconcile_archives(task_id)
            return reconcile_claim_archives(self.cfg, task_id)
        except (FileNotFoundError, OSError, RuntimeError, KeyError, TypeError, ValueError):
            return False

    def _remove_attempt_evidence(self, attempt_id: str) -> None:
        paths = local_paths(self.cfg.runtime_root)
        if self._work is not None:
            directory = paths["termination_decisions"] / attempt_id
            scan = EvidenceScan(directory)
            try:
                page = scan.take(8)
                for path in page.paths:
                    path.unlink(missing_ok=True)
                try:
                    directory.rmdir()
                except FileNotFoundError:
                    pass
                except OSError:
                    # Leave identifying evidence until bounded decision cleanup finishes.
                    return
            finally:
                scan.close()
        for name in (
            "processes",
            "registrations",
            "observations",
            "launch_intents",
            "wrappers",
            "authority_diagnostics",
        ):
            (paths[name] / f"{attempt_id}.json").unlink(missing_ok=True)
        if self._work is None:
            shutil.rmtree(paths["termination_decisions"] / attempt_id, ignore_errors=True)
        for cache in (self._last_renewal, self._failures, self._states, self._lease_expiries):
            cache.pop(attempt_id, None)
        for key in list(self._termination_deadlines):
            if key[0] == attempt_id:
                self._termination_deadlines.pop(key, None)

    def _remove_terminal_evidence(self) -> None:
        paths = local_paths(self.cfg.runtime_root)
        attempt_ids: set[str] = set()
        for name in (
            "processes",
            "registrations",
            "observations",
            "launch_intents",
            "wrappers",
            "authority_diagnostics",
        ):
            attempt_ids.update(path.stem for path in iter_json(paths[name]))
        if paths["termination_decisions"].is_dir():
            attempt_ids.update(path.name for path in paths["termination_decisions"].iterdir() if path.is_dir())
        for attempt_id in attempt_ids:
            self._remove_terminal_attempt_evidence(attempt_id)

    def _remove_terminal_attempt_evidence(self, attempt_id: str) -> None:
        paths = local_paths(self.cfg.runtime_root)
        task_id = None
        for name, record_key in (
            ("processes", "process"),
            ("registrations", "process_registration"),
            ("launch_intents", "launch_intent"),
            ("observations", "exit_observation"),
        ):
            path = paths[name] / f"{attempt_id}.json"
            if not path.exists():
                continue
            record = read_json(path).get(record_key, {})
            if not isinstance(record, dict):
                raise ValueError(f"local {record_key} must be an object")
            value = record.get("task_id")
            if isinstance(value, str):
                task_id = value
                break
        if task_id is None:
            # Attempt IDs are intentionally derived from the Task ID.  This
            # keeps legacy observations bounded without scanning all Tasks.
            candidate, separator, _number = attempt_id.rpartition("-attempt-")
            if separator and candidate and (self.cfg.shared_root / "tasks" / f"{candidate}.json").exists():
                task_id = candidate
        if task_id is not None and self._has_terminal_attempt(task_id, attempt_id):
            self._remove_attempt_evidence(attempt_id)

    def _renew_or_isolate(
        self, task_id: str, attempt_id: str, token: int, process: dict[str, object], *, has_attempt_lock: bool = False
    ) -> None:
        now = datetime.now(timezone.utc).timestamp()
        policy = self._refresh_policy()
        if policy is None:
            self._record_diagnostic(process, "lease_policy_unavailable")
            self._mark_shared_unavailable(process)
            return
        if self._policy_load_failed:
            self._record_diagnostic(process, "lease_policy_unavailable")
            self._mark_shared_unavailable(process)
            return
        if now - self._last_renewal.get(attempt_id, 0) < policy.renew_interval_seconds:
            return
        previous = self._last_renewal.get(attempt_id)
        if previous is not None and process.get("authority_mode") != "holder_bound":
            lateness = max(0.0, now - previous - policy.renew_interval_seconds)
            self.metrics["renewal_lateness.maximum_seconds"] = max(
                self.metrics.get("renewal_lateness.maximum_seconds", 0), lateness
            )
        self._last_renewal[attempt_id] = now
        renewal = renew_attempt_lease(self.cfg, task_id, attempt_id, token)
        outcome_metric = f"renewal.{renewal.outcome.value}"
        self.metrics[outcome_metric] = self.metrics.get(outcome_metric, 0) + 1
        if renewal.outcome is LeaseRenewalOutcome.NOT_REQUIRED:
            self._set_authority_state(process, "local_safe")
            return
        if renewal.outcome is LeaseRenewalOutcome.RENEWED:
            self._failures[attempt_id] = 0
            if renewal.lease_expires_at:
                self._lease_expiries[attempt_id] = renewal.lease_expires_at
            self._set_authority_state(process, "healthy")
            return
        if renewal.outcome is LeaseRenewalOutcome.RETRYABLE_ERROR:
            self._failures[attempt_id] = self._failures.get(attempt_id, 0) + 1
            expires = (
                renewal.lease_expires_at or self._lease_expiries.get(attempt_id) or process.get("lease_expires_at")
            )
            holder_bound = process.get("clock_error_bound_seconds")
            if (
                isinstance(expires, str)
                and isinstance(holder_bound, (int, float))
                and datetime.now(timezone.utc) >= holder_safe_deadline(expires, holder_bound)
            ):
                self._set_authority_state(process, "isolated")
            else:
                self._set_authority_state(process, "suspect")
            return
        resolution = resolve_execution_authority(
            self.cfg,
            task_id,
            attempt_id,
            token,
            uuid.uuid4().hex,
            reservation_runtime_root=self.reservation_runtime_root,
            defer_recovery=self._work is not None,
        )
        if resolution.outcome in {AuthorityResolutionOutcome.RENEWED, AuthorityResolutionOutcome.RECOVERED}:
            if resolution.effective_token is not None:
                process["fencing_token"] = resolution.effective_token
            self._set_authority_state(process, "healthy")
            return
        if resolution.outcome is AuthorityResolutionOutcome.AUTHORITY_UNAVAILABLE:
            self._set_authority_state(process, "isolated")
            return
        terminate = self._terminate_locked if has_attempt_lock else self._terminate
        terminate(
            task_id,
            attempt_id,
            token,
            process,
            resolution.outcome.value,
            resolution.reason or "authority_changed",
        )

    def _send_signals(self, attempt_id: str, decision_id: str) -> dict[str, object]:
        if self._work is None:
            return send_signals(self.cfg, attempt_id, decision_id)
        key = (attempt_id, decision_id)
        result, deadline = advance_signals(
            self.cfg, attempt_id, decision_id, sigterm_deadline=self._termination_deadlines.get(key)
        )
        if deadline is None:
            self._termination_deadlines.pop(key, None)
        else:
            self._termination_deadlines[key] = deadline
        return result

    def _terminate(
        self, task_id: str, attempt_id: str, token: int, process: dict[str, object], outcome: str, reason: str
    ) -> None:
        with attempt_control_lock(self.cfg, attempt_id):
            self._terminate_locked(task_id, attempt_id, token, process, outcome, reason)

    def _terminate_locked(
        self, task_id: str, attempt_id: str, token: int, process: dict[str, object], outcome: str, reason: str
    ) -> None:
        """Commit and advance termination while the caller owns the Attempt lock."""
        decision = create_decision(
            self.cfg,
            task_id=task_id,
            attempt_id=attempt_id,
            fencing_token=token,
            process=process,
            authority_outcome=outcome,
            reason=reason,
        )
        decision_id = decision["decision_id"]
        if commit_shared_termination(self.cfg, task_id, attempt_id, token, decision_id):
            update_decision(self.cfg, attempt_id, decision_id, shared_commitment="committed")
        else:
            commit_local_unavailable(self.cfg, attempt_id, decision_id)
        commit_signal(self.cfg, attempt_id, decision_id)
        self._send_signals(attempt_id, decision_id)

    def _finalize(
        self,
        task_id: str,
        attempt_id: str,
        token: int,
        exit_code: object,
        *,
        was_terminated: bool,
        is_renamed_execution: bool = False,
    ) -> None:
        task = load_task(self.cfg, task_id)
        result = None
        with authority_locks(self.cfg, task):
            task = load_task(self.cfg, task_id)
            claim = task.claim_control.get("active_claim") or {}
            if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != token:
                return
            number = task.attempt_control.get("current_attempt_number")
            if not isinstance(number, int):
                return
            code = exit_code if isinstance(exit_code, int) else None
            was_cancel_requested = bool(task.control.get("terminate_running"))
            phase = "cancelled" if was_terminated else ("succeeded" if code == 0 else "failed")
            reason = "terminated_by_agent" if was_terminated else ("completed" if code == 0 else "nonzero_exit")
            termination_result = (
                "terminated" if was_terminated else ("already_exited" if was_cancel_requested else None)
            )
            result = commit_terminal_transition_locked(
                self.cfg,
                task,
                TerminalTransition(
                    task_id,
                    attempt_id,
                    number,
                    token,
                    phase,
                    reason,
                    code,
                    frozenset({"running", "starting", "claimed"}),
                    frozenset({"claimed", "starting", "running"}),
                    "active",
                    termination_result,
                ),
            )
        if result.outcome != "committed":
            return
        committed_at = utc_now()
        try:
            observation = read_json(local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json")
            self._observe_latency("exit_to_terminal", observation.get("exit_observation", {}).get("observed_at"))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if result.reservation_id and (is_renamed_execution or result.reservation_machine_name == self.cfg.machine_name):
            release(self.reservation_runtime_root, result.reservation_id, reason)
        self._observe_latency("terminal_to_accounting", committed_at)
        manifest_path = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
        can_remove_evidence = False
        try:
            process = read_json(manifest_path).get("process")
            if (
                isinstance(process, dict)
                and process.get("task_id") == task_id
                and process.get("attempt_id") == attempt_id
                and process.get("fencing_token") == token
            ):
                process.update(
                    {"observed_state": "exited", "observed_exit_code": code, "observed_exited_at": utc_now()}
                )
                atomic_replace(manifest_path, {"process": process})
                can_remove_evidence = True
            else:
                self._record_diagnostic({"attempt_id": attempt_id}, "terminal_manifest_unreadable")
        except (OSError, ValueError) as error:
            self._record_diagnostic({"attempt_id": attempt_id}, "terminal_manifest_unreadable", error)
        if result.event:
            dispatch_task_lifecycle_hooks_noexcept(self.cfg, result.event)
        if can_remove_evidence:
            self._remove_attempt_evidence(attempt_id)
