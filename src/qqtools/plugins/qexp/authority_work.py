"""Cooperative evidence lanes for one project's authority supervisor."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator

from .runtime.authority_scan import EvidenceScan
from .runtime.claim_archive_scan import ClaimArchiveDiscovery, is_pending_archive_clear
from .runtime.paths import local_paths
from .runtime.records import validate_identifier
from .runtime.responsibility import ResponsibilityReader
from .runtime.responsibility_cleanup import CleanupRequest
from .runtime.store import read_json
from .runtime.termination import attempt_control_lock
from .runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

if TYPE_CHECKING:
    from .authority import AuthoritySupervisor


@dataclass
class _Lane:
    scan: EvidenceScan
    operation: Callable[[Path], None]
    visited: int = 0
    processed: int = 0
    failures: int = 0
    sweeps: int = 0
    seconds: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    has_sweep_error: bool = False
    has_completed: bool = False
    pending_since: float | None = None


class AuthorityWork:
    """Retain bounded discovery state; records and existing fences remain authoritative."""

    def __init__(self, supervisor: AuthoritySupervisor) -> None:
        self.supervisor = supervisor
        self.diagnostics = RuntimeDiagnostics()
        self.paths = local_paths(supervisor.cfg.runtime_root)
        self._responsibilities = ResponsibilityReader(supervisor.cfg.runtime_root, owner=supervisor.recovery_owner)
        self._responsibility_turn = 0
        self._active: OrderedDict[str, None] = OrderedDict()
        self._pending_control: Generator[None, None, int | None] | None = None
        self._pending_attempt: str | None = None
        self._pending_started: float | None = None
        self._pending_kind: str | None = None
        self._active_limit = 256
        self._active_admission_deferred = 0
        self._archives = ClaimArchiveDiscovery(supervisor.cfg)
        self._slice_token = object()
        self._turn = 0
        self._startup_turn = 0
        self._cleanup_turn = 0
        self._decision_scan: EvidenceScan | None = None
        self._decision_root: Path | None = None
        self._decision_queue: deque[Path] = deque()
        self._queued_decisions: set[Path] = set()
        self._initial_decisions: set[Path] = set()
        self._cleanup_scans = [
            EvidenceScan(self.paths[name])
            for name in (
                "processes",
                "registrations",
                "observations",
                "launch_intents",
                "wrappers",
                "authority_diagnostics",
            )
        ]
        self._lanes = {
            "registrations": _Lane(EvidenceScan(self.paths["registrations"]), self._registration),
            "supervision": _Lane(EvidenceScan(self.paths["processes"]), self._process),
            "observations": _Lane(EvidenceScan(self.paths["observations"]), self._observation),
            "intents": _Lane(EvidenceScan(self.paths["launch_intents"]), self._intent),
            "termination": _Lane(
                EvidenceScan(self.paths["termination_decisions"], directories=True), self._termination
            ),
        }
        # Reserve half the service opportunities for already discovered Attempts.
        # Every discovery/repair lane retains one turn in each fourteen-step round.
        self._order = tuple(
            name
            for lane in (
                "registrations",
                "supervision",
                "observations",
                "intents",
                "termination",
                "cleanup",
                "responsibilities",
            )
            for name in ("active", lane)
        )
        self._cleanup_visited = 0
        self._cleanup_failures = 0
        self._active_processed = 0
        self._active_failures = 0
        self._idle_steps_reassigned = 0
        self._last_served: dict[str, float] = {}
        self._maximum_service_gap = 0.0

    @property
    def is_startup_complete(self) -> bool:
        mode = self._responsibilities.discovery_mode
        if mode in {"checking", "unavailable"}:
            return False
        if mode == "primary":
            return (
                self._pending_control is None
                and self._responsibilities.is_initial_sweep_complete
                and not self._initial_decisions
            )
        return self._pending_control is None and all(lane.has_completed for lane in self._lanes.values())

    def close(self) -> None:
        self._responsibilities.close()
        self.cancel_pending_control()
        self._archives.close()
        for lane in self._lanes.values():
            lane.scan.close()
        for scan in self._cleanup_scans:
            scan.close()
        if self._decision_scan is not None:
            self._decision_scan.close()

    def is_control_pending(self, attempt_id: str) -> bool:
        return attempt_id == self._pending_attempt

    def cancel_pending_control(self, *, is_complete: bool = False) -> None:
        if self._pending_control is not None:
            if not is_complete and self._responsibilities.discovery_mode == "primary":
                self._responsibilities.invalidate_initial_sweep()
                self._last_served.pop(self._pending_attempt, None)
            self._pending_control.close()
        self._pending_control = None
        self._pending_attempt = None
        self._pending_started = None
        self._pending_kind = None

    def recover(self, process: dict[str, object]) -> None:
        """Queue at most one locked recovery inventory; other Attempts retry later."""
        if self._pending_control is not None:
            if self._responsibilities.discovery_mode == "primary":
                raise RuntimeError("recovery candidate deferred while another control is pending")
            return
        from .runtime.attempt_recovery import recovery_steps

        self._pending_attempt = process["attempt_id"]
        self._pending_started = time.monotonic()
        self._pending_kind = "recovery"
        self._pending_control = recovery_steps(
            self.supervisor.cfg,
            process["task_id"],
            process["attempt_id"],
            process["fencing_token"],
            reservation_runtime_root=self.supervisor.reservation_runtime_root,
        )

    def supervise(self, process: dict[str, object]) -> None:
        """Finish a short inventory now, or retain one long inventory per project."""
        steps = self.supervisor.supervision_steps(process)
        retained = False
        try:
            next(steps)
            if self._pending_control is None:
                self._pending_control = steps
                self._pending_attempt = process["attempt_id"]
                self._pending_started = time.monotonic()
                self._pending_kind = "supervision"
                retained = True
            elif self._responsibilities.discovery_mode == "primary":
                raise RuntimeError("supervision candidate deferred while another control is pending")
        except StopIteration:
            pass
        finally:
            if not retained:
                steps.close()

    def _control_step(self) -> None:
        if self._pending_control is None:
            return
        attempt_id = self._pending_attempt
        try:
            if self._pending_kind == "recovery" and (self.paths["observations"] / f"{attempt_id}.json").exists():
                self.cancel_pending_control()
                return
            next(self._pending_control)
        except StopIteration:
            self.cancel_pending_control(is_complete=True)
            if attempt_id in self._active:
                self._last_served[attempt_id] = time.monotonic()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            kind = self._pending_kind
            self.cancel_pending_control()
            self.supervisor._record_diagnostic({"attempt_id": attempt_id}, f"{kind}_failed", exc)

    def _intent(self, path: Path) -> None:
        if path.stem != self._pending_attempt and not self._responsibilities.is_cleanup_pending(path.stem):
            self.supervisor._materialize_unverified_intent(path)

    def reconcile_archives(self, task_id: str) -> bool:
        if is_pending_archive_clear(self.supervisor.cfg, task_id):
            return True
        try:
            self._archives.step(8)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.supervisor._record_diagnostic({"attempt_id": "control-plane"}, "claim_archive_replay_unavailable", exc)
            raise
        return is_pending_archive_clear(self.supervisor.cfg, task_id)

    def _remember(self, attempt_id: str) -> None:
        if attempt_id not in self._active:
            # Discovery must not evict the next due member of the active rotation.
            # Overflow remains discoverable and is serviced directly by _process.
            if len(self._active) >= self._active_limit:
                self._active_admission_deferred += 1
                return
            self._active[attempt_id] = None

    def _registration(self, path: Path) -> None:
        if path.stem == self._pending_attempt or self._responsibilities.is_cleanup_pending(path.stem):
            return
        self.supervisor._materialize_registrations(registration_paths=(path,), should_materialize_intents=False)
        if (self.paths["processes"] / path.name).exists():
            self._remember(path.stem)

    def _process(self, path: Path) -> None:
        if path.stem == self._pending_attempt or self._responsibilities.is_cleanup_pending(path.stem):
            return
        now = time.monotonic()
        previous = self._last_served.get(path.stem)
        if previous is not None and now - previous < min(1.0, self.supervisor.renewal_interval_seconds):
            return
        process = read_json(path).get("process", {})
        if not isinstance(process, dict):
            raise ValueError("local process manifest must be an object")
        if process.get("protocol_version") != 1:
            return
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str) or attempt_id != path.stem:
            return
        self._remember(attempt_id)
        if previous is not None:
            self._maximum_service_gap = max(self._maximum_service_gap, now - previous)
        self.supervisor.service_process(process)
        if attempt_id in self._active:
            self._last_served[attempt_id] = now

    def _observation(self, path: Path) -> None:
        if self._responsibilities.is_cleanup_pending(path.stem):
            return
        if path.stem == self._pending_attempt:
            if self._pending_kind != "recovery":
                return
            self.cancel_pending_control()
        manifest = self.supervisor.prepare_observed_exit(path)
        if manifest is not None:
            self._last_served.pop(path.stem, None)
            self._process(manifest)

    def _termination(self, path: Path) -> None:
        # The outer lane hands off just one directory. Its nested records are
        # consumed on subsequent turns rather than eagerly expanding a tree.
        if self._responsibilities.discovery_mode == "primary":
            if path == self._decision_root or path in self._queued_decisions:
                if not self._responsibilities.is_initial_sweep_complete:
                    self._initial_decisions.add(path)
                return
            if self._decision_scan is not None:
                if len(self._decision_queue) >= 256:
                    raise RuntimeError("indexed termination discovery backlog is full")
                self._decision_queue.append(path)
                self._queued_decisions.add(path)
                if not self._responsibilities.is_initial_sweep_complete:
                    self._initial_decisions.add(path)
                return
            if not self._responsibilities.is_initial_sweep_complete:
                self._initial_decisions.add(path)
        self._decision_root = path
        self._decision_scan = EvidenceScan(path)

    def _termination_step(self) -> bool:
        if self._decision_scan is None:
            return False
        page = self._decision_scan.take(1, slice_token=self._slice_token)
        for path in page.paths:
            value = read_json(path).get("termination_decision", {})
            if not isinstance(value, dict):
                raise ValueError("local termination decision must be an object")
            if self._pending_attempt is not None and value.get("attempt_id") == self._pending_attempt:
                continue
            if self._responsibilities.is_cleanup_pending(value.get("attempt_id")):
                continue
            if value.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent"}:
                with attempt_control_lock(self.supervisor.cfg, value["attempt_id"]):
                    self.supervisor._send_signals(value["attempt_id"], value["decision_id"])
        if page.is_complete:
            self._decision_scan.close()
            self._initial_decisions.discard(self._decision_root)
            self._decision_scan = None
            self._decision_root = None
            if self._decision_queue:
                following = self._decision_queue.popleft()
                self._queued_decisions.remove(following)
                self._termination(following)
        return True

    def _active_step(self) -> None:
        if not self._active:
            return
        attempt_id, _ = self._active.popitem(last=False)
        path = self.paths["processes"] / f"{attempt_id}.json"
        if not path.exists():
            self._last_served.pop(attempt_id, None)
            return
        self._active[attempt_id] = None
        self._process(path)
        self._active_processed += 1

    def _cleanup_step(self) -> None:
        scan = self._cleanup_scans[self._cleanup_turn]
        self._cleanup_turn = (self._cleanup_turn + 1) % len(self._cleanup_scans)
        page = scan.take(1, slice_token=self._slice_token)
        self._cleanup_visited += page.entries_visited
        for path in page.paths:
            if path.stem != self._pending_attempt:
                self.supervisor._remove_terminal_attempt_evidence(path.stem)

    def cleanup_responsibility(self, request: CleanupRequest) -> bool:
        return self._responsibilities.cleanup(request)

    def _responsibility_step(self) -> bool:
        if self._pending_control is not None and self._responsibilities.discovery_mode == "primary":
            # Preserve prefetched candidates until their semantic work can own
            # the slot. Restarting traversal here can repeatedly rediscover the
            # same long-running first member and starve every later member.
            return False
        entry = self._responsibilities.take()
        if entry is None:
            return False
        try:
            result = self._service_responsibility(entry)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._responsibilities.acknowledge(is_success=False, stage=entry["stage"])
            raise
        self._responsibilities.acknowledge()
        return result

    def _service_responsibility(self, entry: dict) -> bool:
        attempt_id = validate_identifier(entry["identity"], "attempt_id")
        if attempt_id == self._pending_attempt:
            return True
        cleanup = CleanupRequest.from_entry(entry)
        if cleanup is None:
            cleanup = self.supervisor._cleanup_request_for_membership(entry)
        if cleanup is not None:
            if self._responsibilities.cleanup(cleanup):
                self.supervisor._forget_attempt(attempt_id)
            return True
        for name, operation in (
            ("observations", self._observation),
            ("registrations", self._registration),
            ("processes", self._process),
            ("launch_intents", self._intent),
        ):
            path = self.paths[name] / f"{attempt_id}.json"
            if path.exists():
                operation(path)
                if name in {"registrations", "launch_intents"} and self._responsibilities.discovery_mode == "primary":
                    manifest = self.paths["processes"] / path.name
                    if manifest.exists():
                        self._process(manifest)
                break
        if self._responsibilities.discovery_mode == "primary":
            decisions = self.paths["termination_decisions"] / attempt_id
            if decisions.exists():
                self._termination(decisions)
        self.supervisor._remove_terminal_attempt_evidence(attempt_id)
        return True

    def tick(self, limit: int = 64) -> dict[str, object]:
        with activate_diagnostics(self.diagnostics):
            return self._tick(limit)

    def _tick(self, limit: int) -> dict[str, object]:
        """Execute at most limit semantic steps, with fixed service for every lane."""
        if type(limit) is not int or limit < 1 or limit > 256:
            raise ValueError("authority work limit must be an integer from 1 to 256")
        self._slice_token = object()
        if self._responsibilities.owner is not None:
            self._responsibilities.poll()
            if self._responsibilities.discovery_mode == "checking":
                return self.snapshot()
        is_primary = self._responsibilities.discovery_mode == "primary"
        order = ("active", "responsibilities", "active", "termination") if is_primary else self._order
        completed_scans: set[EvidenceScan] = set()
        if self._pending_control is not None:
            self._control_step()
            limit -= 1
        self._responsibility_turn = min(4, self._responsibility_turn + 1)
        for _ in range(limit):
            self._turn %= len(order)
            name = order[self._turn]
            self._turn = (self._turn + 1) % len(order)
            if name == "active" and not self._active:
                # Borrow only empty active turns; cached Attempts keep their share.
                name = "responsibilities" if is_primary else order[self._turn]
                unfinished = [] if is_primary else [key for key, lane in self._lanes.items() if not lane.has_completed]
                if unfinished:
                    name = unfinished[self._startup_turn % len(unfinished)]
                    self._startup_turn += 1
                self._idle_steps_reassigned += 1
            started = time.monotonic()
            lane = self._lanes.get(name)
            try:
                if name == "active":
                    self._active_step()
                elif name == "cleanup":
                    self._cleanup_step()
                elif name == "responsibilities":
                    if is_primary or self._responsibility_turn == 4:
                        self._responsibility_turn = 0
                        self._responsibility_step()
                elif name == "termination" and is_primary:
                    self._termination_step()
                elif name == "termination" and self._termination_step():
                    lane.visited += 1
                else:
                    if lane.scan in completed_scans:
                        continue
                    page = lane.scan.take(1, slice_token=self._slice_token)
                    lane.visited += page.entries_visited
                    for path in page.paths:
                        lane.operation(path)
                        lane.processed += 1
                    if page.is_complete:
                        completed_scans.add(lane.scan)
                        lane.sweeps += 1
                        if not lane.has_sweep_error:
                            lane.has_completed = True
                            lane.pending_since = None
                        lane.has_sweep_error = False
                        lane.started_at = time.monotonic()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                if is_primary and name == "termination" and self._decision_scan is not None:
                    # A failed initial decision must be revisited before startup
                    # can consume this directory's successful EOF.
                    self._decision_scan.close()
                if lane is not None:
                    lane.failures += 1
                    lane.has_sweep_error = True
                    if lane.pending_since is None:
                        lane.pending_since = time.monotonic()
                elif name == "active":
                    self._active_failures += 1
                elif name == "responsibilities":
                    self._responsibilities.failures += 1
                else:
                    self._cleanup_failures += 1
                self.supervisor._record_diagnostic({"attempt_id": "control-plane"}, f"{name}_unavailable", exc)
            finally:
                if lane is not None:
                    lane.seconds += max(0.0, time.monotonic() - started)
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        return {
            "startup_complete": self.is_startup_complete,
            "discovery_mode": self._responsibilities.discovery_mode,
            "discovery_error": self._responsibilities.discovery_error,
            "termination_backlog": len(self._decision_queue),
            "responsibility_discovery_failures": self._responsibilities.failures,
            "pending_control_attempt": self._pending_attempt,
            "pending_control_kind": self._pending_kind,
            "control_pending_age_seconds": (
                None if self._pending_started is None else max(0.0, now - self._pending_started)
            ),
            "metrics": dict(self.supervisor.metrics),
            "operations": self.diagnostics.snapshot(),
            "oldest_active_due_age_seconds": max(
                (
                    max(0.0, now - served - min(1.0, self.supervisor.renewal_interval_seconds))
                    for served in self._last_served.values()
                ),
                default=0.0,
            ),
            "archive_cursor_count": self._archives.cursor_count,
            "archive_cursor_limit": 2,
            "active_cache_size": len(self._active),
            "active_cache_limit": self._active_limit,
            "active_admission_deferred": self._active_admission_deferred,
            "active_processed": self._active_processed,
            "active_failures": self._active_failures,
            "idle_steps_reassigned": self._idle_steps_reassigned,
            "maximum_service_gap_seconds": self._maximum_service_gap,
            "cleanup_entries": self._cleanup_visited,
            "cleanup_failures": self._cleanup_failures,
            "lanes": {
                name: {
                    "entries": lane.visited,
                    "processed": lane.processed,
                    "failures": lane.failures,
                    "sweeps": lane.sweeps,
                    "work_seconds": lane.seconds,
                    "sweep_age_seconds": max(0.0, now - lane.started_at),
                    "oldest_failed_work_age_seconds": (
                        None if lane.pending_since is None else max(0.0, now - lane.pending_since)
                    ),
                }
                for name, lane in self._lanes.items()
            },
        }
