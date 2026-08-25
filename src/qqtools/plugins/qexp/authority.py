"""Agent-owned runtime authority for locally registered qexp processes."""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .config_types import RootConfig
from .lifecycle import (TerminalTransition, commit_terminal_transition_locked,
                         dispatch_task_lifecycle_hooks_noexcept)
from .lease import (AuthorityResolutionOutcome, LeasePolicy, LeaseRenewalOutcome,
                    holder_safe_deadline, load_lease_policy)
from .runtime.claims import archive_claim
from .runtime.paths import attempt_path, local_paths
from .runtime.records import AttemptRecord, utc_now
from .runtime.reservations import release
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.tasks import load_task, save_task
from .runtime.termination import (attempt_control_lock, commit_local_unavailable, commit_signal,
                                  create_decision, send_signals, update_decision)
from .scheduler import (authority_locks, commit_shared_termination, renew_attempt_lease,
                        resolve_execution_authority)


class AuthoritySupervisor:
    """The sole local writer of live-process authority, termination, and terminal truth."""

    def __init__(self, cfg: RootConfig) -> None:
        self.cfg = cfg
        self._last_renewal: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._states: dict[str, str] = {}
        self._lease_expiries: dict[str, str] = {}
        self._policy: LeasePolicy | None = None
        self._policy_load_failed = False
        self._refresh_policy()

    def recover_startup(self) -> None:
        self._materialize_registrations()
        for directory in local_paths(self.cfg.runtime_root)["termination_decisions"].glob("*"):
            if directory.is_dir():
                for decision in iter_json(directory):
                    value = read_json(decision).get("termination_decision", {})
                    if value.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent"}:
                        with attempt_control_lock(self.cfg, value["attempt_id"]):
                            send_signals(self.cfg, value["attempt_id"], value["decision_id"])

    def tick(self) -> None:
        self._materialize_registrations()
        for path in iter_json(local_paths(self.cfg.runtime_root)["processes"]):
            process = read_json(path).get("process", {})
            if process.get("protocol_version") != 1:
                continue
            try:
                self._supervise(process)
            except OSError as exc:
                self._record_diagnostic(process, "shared_storage_unavailable", exc)
                self._mark_shared_unavailable(process)

    def _refresh_policy(self) -> LeasePolicy | None:
        try:
            self._policy = load_lease_policy(self.cfg)
            self._policy_load_failed = False
            atomic_replace(local_paths(self.cfg.runtime_root)["lease_policy_cache"], {
                "lease_policy": asdict(self._policy),
            })
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
            atomic_replace(local_paths(self.cfg.runtime_root)["authority_diagnostics"] / f"{attempt_id}.json",
                           {"authority_diagnostic": value})
        except OSError:
            pass

    def _set_authority_state(self, process: dict[str, object], state: str) -> None:
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        self._states[attempt_id] = state
        process["authority_state"] = state
        try:
            atomic_replace(local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json",
                           {"process": process})
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
        if (policy and isinstance(expires, str) and isinstance(holder_bound, (int, float))
                and datetime.now(timezone.utc) >= holder_safe_deadline(expires, holder_bound)):
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
                    attempt = AttemptRecord.from_dict(read_json(attempt_path(self.cfg.shared_root, task_id, number)))
                except (FileNotFoundError, KeyError, ValueError):
                    return
                if (claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != token
                        or claim.get("machine_name") != self.cfg.machine_name
                        or attempt.attempt_id != attempt_id or attempt.current_fencing_token != token
                        or attempt.machine_name != self.cfg.machine_name):
                    return
                if claim.get("launch_state") not in {"starting", "running"} or attempt.phase not in {"starting", "running"}:
                    return
                for key in ("wrapper_pid", "wrapper_start_time_ticks", "process_group_id", "process_group_start_time_ticks"):
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
                attempt.phase = "running"
                atomic_replace(attempt_path(self.cfg.shared_root, task_id, number), attempt.to_dict())
                if claim.get("launch_state") != "running":
                    claim["launch_state"] = "running"
                    task.meta["revision"] += 1
                    task.meta["updated_at"] = utc_now()
                    save_task(self.cfg, task)

    def _materialize_registrations(self) -> None:
        self._materialize_unverified_intents()
        for path in iter_json(local_paths(self.cfg.runtime_root)["registrations"]):
            registration = read_json(path).get("process_registration", {})
            if registration.get("protocol_version") != 1:
                continue
            attempt_id = registration.get("attempt_id")
            if not isinstance(attempt_id, str):
                continue
            manifest = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
            if not manifest.exists():
                value = dict(registration)
                value.update({"observed_state": "running", "supervisor": "agent",
                              "authority_state": "healthy", "created_by": "agent"})
                atomic_replace(manifest, {"process": value})
            self._publish_running(registration, manifest)

    def _materialize_unverified_intents(self) -> None:
        for path in iter_json(local_paths(self.cfg.runtime_root)["launch_intents"]):
            intent = read_json(path).get("launch_intent", {})
            if intent.get("protocol_version") != 1:
                continue
            attempt_id = intent.get("attempt_id")
            if not isinstance(attempt_id, str):
                continue
            registration = local_paths(self.cfg.runtime_root)["registrations"] / f"{attempt_id}.json"
            manifest = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
            if registration.exists() or manifest.exists():
                continue
            if self._wrapper_matches(intent):
                continue
            value = dict(intent)
            value.update({"observed_state": "launch_unverifiable", "supervisor": "agent",
                          "authority_state": "isolated", "created_by": "agent"})
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
        if process.get("observed_state") == "launch_unverifiable":
            self._record_diagnostic(process, "launch_unverifiable")
            return
        for decision_file in iter_json(local_paths(self.cfg.runtime_root)["termination_decisions"] / attempt_id):
            decision = read_json(decision_file).get("termination_decision", {})
            if decision.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent"}:
                with attempt_control_lock(self.cfg, attempt_id):
                    result = send_signals(self.cfg, attempt_id, decision["decision_id"])
                if result.get("state") == "confirmed":
                    observation = local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json"
                    exit_code = None
                    if observation.exists():
                        exit_code = read_json(observation).get("exit_observation", {}).get("observed_exit_code")
                    self._finalize(
                        task_id,
                        attempt_id,
                        token,
                        exit_code,
                        was_terminated=bool(result.get("signal_attempts")),
                    )
                return
        observation = local_paths(self.cfg.runtime_root)["observations"] / f"{attempt_id}.json"
        if observation.exists():
            exit_code = read_json(observation).get("exit_observation", {}).get("observed_exit_code")
            self._finalize(task_id, attempt_id, token, exit_code, was_terminated=False)
            return
        self._renew_or_isolate(task_id, attempt_id, token, process)

    def _renew_or_isolate(self, task_id: str, attempt_id: str, token: int, process: dict[str, object]) -> None:
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
        self._last_renewal[attempt_id] = now
        renewal = renew_attempt_lease(self.cfg, task_id, attempt_id, token)
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
            expires = renewal.lease_expires_at or self._lease_expiries.get(attempt_id) or process.get("lease_expires_at")
            holder_bound = process.get("clock_error_bound_seconds")
            if (isinstance(expires, str) and isinstance(holder_bound, (int, float))
                    and datetime.now(timezone.utc) >= holder_safe_deadline(expires, holder_bound)):
                self._set_authority_state(process, "isolated")
            else:
                self._set_authority_state(process, "suspect")
            return
        resolution = resolve_execution_authority(self.cfg, task_id, attempt_id, token, uuid.uuid4().hex)
        if resolution.outcome in {AuthorityResolutionOutcome.RENEWED, AuthorityResolutionOutcome.RECOVERED}:
            self._set_authority_state(process, "healthy")
            return
        if resolution.outcome is AuthorityResolutionOutcome.AUTHORITY_UNAVAILABLE:
            self._set_authority_state(process, "isolated")
            return
        self._terminate(
            task_id, attempt_id, token, process, resolution.outcome.value,
            resolution.reason or "authority_changed",
        )

    def _terminate(self, task_id: str, attempt_id: str, token: int, process: dict[str, object], outcome: str,
                   reason: str) -> None:
        with attempt_control_lock(self.cfg, attempt_id):
            decision = create_decision(self.cfg, task_id=task_id, attempt_id=attempt_id, fencing_token=token,
                                       process=process, authority_outcome=outcome, reason=reason)
            decision_id = decision["decision_id"]
            if commit_shared_termination(self.cfg, task_id, attempt_id, token, decision_id):
                update_decision(self.cfg, attempt_id, decision_id, shared_commitment="committed")
            else:
                commit_local_unavailable(self.cfg, attempt_id, decision_id)
            commit_signal(self.cfg, attempt_id, decision_id)
            send_signals(self.cfg, attempt_id, decision_id)

    def _finalize(self, task_id: str, attempt_id: str, token: int, exit_code: object,
                  *, was_terminated: bool) -> None:
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
            reason = ("terminated_by_agent" if was_terminated else
                      ("completed" if code == 0 else "nonzero_exit"))
            termination_result = ("terminated" if was_terminated else
                                  ("already_exited" if was_cancel_requested else None))
            result = commit_terminal_transition_locked(
                self.cfg, task, TerminalTransition(task_id, attempt_id, number, token, phase, reason,
                    code, frozenset({"running", "starting", "claimed"}),
                    frozenset({"claimed", "starting", "running"}), "active",
                    termination_result))
        if result.outcome != "committed":
            return
        if result.reservation_id and result.reservation_machine_name == self.cfg.machine_name:
            release(self.cfg.runtime_root, result.reservation_id, reason)
        manifest_path = local_paths(self.cfg.runtime_root)["processes"] / f"{attempt_id}.json"
        process = read_json(manifest_path).get("process", {})
        if (process.get("task_id") == task_id and process.get("attempt_id") == attempt_id
                and process.get("fencing_token") == token):
            process.update({"observed_state": "exited", "observed_exit_code": code,
                            "observed_exited_at": utc_now()})
            atomic_replace(manifest_path, {"process": process})
        if result.event:
            dispatch_task_lifecycle_hooks_noexcept(self.cfg, result.event)
