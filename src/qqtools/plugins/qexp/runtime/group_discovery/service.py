"""Machine-local scheduling and cleanup for Group discovery."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...agent.context import MachineRuntime, ProjectBinding
from ...agent.working_set import BindingTurn
from ..group_namespace import (
    GroupNotPublished,
    GroupPublicationUnavailable,
    group_directory,
    is_group_authority_isolated,
    read_group,
)
from ..paths import group_path
from .control import GroupControlCursor, advance_group_control_rechecks
from .discovery import (
    _SOURCE_IO_BYTES,
    _SOURCE_OPERATIONS,
    _SOURCE_SLICE_SECONDS,
    GroupDiscoveryService,
    publish_submission_debt,
)
from .discovery import publish_group_locator_for_transition as publish_group_locator_for_transition
from .slice_io import SliceIO
from .source_sweep import SubmissionSourceSweep

# These bounds cap cooperative work requested per slice; they are not hard
# latency guarantees for filesystem calls or synchronization.
_WORKER_WAIT_SECONDS = 0.001
_WORKER_IDLE_SECONDS = 1.0
_WORKER_COOLDOWN_SECONDS = 1.0
_LOCATOR_LANE_CYCLE = ("control", "control", "membership", "maintenance")
_MAX_RESIDENT_GROUPS = 64
_MAX_PARSER_OWNERS = 16
_MAX_CLOSE_WORK = 16
_MAX_LOCATOR_CANDIDATES = 64
_MAX_SERVICE_DESCRIPTORS = 256


@dataclass(slots=True)
class _GroupServiceEntry:
    """Retained discovery and maintenance owners for one Group key."""

    key: tuple[str, str, str, str]
    binding_key: tuple[str, str, str]
    binding: ProjectBinding
    group: str
    service: Any | None
    maintenance: Any | None
    locator_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    locator_observed_at: dict[str, float] = field(default_factory=dict)
    locator_mode: bool = False
    pending_lane: str | None = None
    control_cursor: GroupControlCursor = field(default_factory=GroupControlCursor)
    retry_delay: float = 1.0
    locator_retry_delay: dict[str, float] = field(default_factory=dict)
    locator_cooldown_until: dict[str, float] = field(default_factory=dict)
    service_turn: bool = True
    cooldown_until: float = 0.0
    restart_after_close: bool = False


@dataclass(slots=True)
class _CloseWork:
    """One cooperatively closed owner, advanced at most once per turn."""

    kind: str
    owner: Any
    binding_key: tuple[str, str, str] | None = None
    phase: str = "service"
    requested: bool = False


class MachineGroupDiscoveryWorker:
    """Run bounded Group discovery independently of machine scheduling."""

    def __init__(self, runtime: MachineRuntime) -> None:
        if not isinstance(runtime, MachineRuntime):
            raise TypeError("runtime must be a MachineRuntime")
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="qexp-group-discovery",
            daemon=True,
        )
        self._entries: dict[tuple[str, str, str, str], _GroupServiceEntry] = {}
        self._entry_queue: deque[_GroupServiceEntry] = deque()
        self._restart_due: dict[tuple[str, str, str, str], _GroupServiceEntry] = {}
        self._close_queue: deque[_CloseWork] = deque()
        self._close_work_index: dict[tuple[str, int], _CloseWork] = {}
        self._closing_group_keys: set[tuple[str, str, str, str]] = set()
        self._closing_binding_keys: set[tuple[str, str, str]] = set()
        self._close_work_counts: dict[tuple[str, str, str], int] = {}
        self._bindings_by_key: dict[tuple[str, str, str], ProjectBinding] = {}
        self._locator_traversals: dict[tuple[tuple[str, str, str], str], Any] = {}
        self._lane_turns: dict[tuple[str, str, str], int] = {}
        self._group_turns: dict[tuple[str, str, str], tuple[BindingTurn, dict[str, int]]] = {}
        self._legacy_sweep_passes: dict[tuple[str, str, str], int] = {}
        self._pending_locator: tuple[str, dict[str, Any]] | None = None
        self._active_project_slice = False
        self._lane_metrics: dict[str, dict[str, int | float]] = {
            lane: {
                "pending_locators_encountered": 0,
                "resident_entries": 0,
                "completions": 0,
                "retries": 0,
                "evictions": 0,
                "bytes": 0,
                "reads": 0,
                "writes": 0,
                "descriptor_high_water": 0,
                "cap_refusals": 0,
                "oldest_pending_age_seconds": 0.0,
            }
            for lane in ("control", "membership", "maintenance")
        }

    @property
    def metrics(self) -> dict[str, dict[str, int | float]]:
        """Return the maintained counters without scanning shared history."""

        now = time.monotonic()
        for lane in self._lane_metrics:
            observed = [
                entry.locator_observed_at[lane] for entry in self._entries.values() if lane in entry.locator_observed_at
            ]
            self._lane_metrics[lane]["oldest_pending_age_seconds"] = max(
                (now - started for started in observed), default=0.0
            )
        return {lane: dict(values) for lane, values in self._lane_metrics.items()}

    @property
    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running."""

        return self._thread.is_alive()

    def start(self) -> None:
        """Start the independent daemon worker."""

        self._thread.start()

    def stop(self) -> None:
        """Request stop and wait briefly without touching worker-owned state."""

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        registry_revision: int | None = None
        bindings: list[ProjectBinding] = []
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep] = {}
        sweep_complete_at: dict[tuple[str, str, str], float] = {}
        binding_cursor = 0
        try:
            while not self._stop.is_set():
                try:
                    revision, registered = self._runtime.load_registry_snapshot()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    self._advance_close_one()
                    self._stop.wait(_WORKER_WAIT_SECONDS)
                    continue
                self._runtime.working_set.reconcile(registered, revision=revision)
                resident = self._runtime.working_set.resident_bindings()
                disabled = [binding for binding in resident if not binding.enabled]
                service_bindings = [binding for binding in resident if binding.enabled]
                if registry_revision is None or registry_revision != revision or bindings != service_bindings:
                    self._reconcile_bindings(service_bindings, sweeps)
                    sweep_complete_at = {
                        key: due for key, due in sweep_complete_at.items() if key in self._bindings_by_key
                    }
                    bindings = service_bindings
                    registry_revision = revision
                    binding_cursor %= max(1, len(bindings))

                # Cleanup receives one cooperative step even while healthy
                # Groups continue to receive their own fair work quantum.
                activity_keys: set[tuple[str, str, str]] = set()
                closed_binding = self._advance_close_one()
                if closed_binding is not None:
                    activity_keys.add(closed_binding)
                self._ack_disabled_bindings(disabled, sweeps)
                turn_binding = bindings[binding_cursor % len(bindings)] if bindings else None
                if turn_binding is not None:
                    self._ensure_group_turn(turn_binding, sweeps, sweep_complete_at)
                    retry_count_before = sum(values["retries"] for values in self._lane_metrics.values())
                selected = self._next_group(
                    bindings,
                    sweeps,
                    sweep_complete_at,
                    binding_cursor,
                )
                if bindings:
                    if selected is None:
                        binding_cursor = (binding_cursor + 1) % len(bindings)
                    else:
                        binding_cursor = (selected[0] + 1) % len(bindings)
                if selected is not None:
                    _, binding, group = selected
                    activity_keys.add(self._binding_key(binding))
                    self._admit_group(binding, group)
                    if self._pending_locator is not None:
                        lane, record = self._pending_locator
                        self._advance_active_locator(binding, group, lane, record)
                    elif not self._active_project_slice:
                        advanced_binding = self._advance_one_entry()
                        if advanced_binding is not None:
                            activity_keys.add(self._binding_key(advanced_binding))
                else:
                    advanced_binding = self._advance_one_entry()
                    if advanced_binding is not None:
                        activity_keys.add(self._binding_key(advanced_binding))
                restarted_binding = self._restart_one_due()
                if restarted_binding is not None:
                    activity_keys.add(self._binding_key(restarted_binding))
                if (
                    turn_binding is not None
                    and sum(values["retries"] for values in self._lane_metrics.values()) > retry_count_before
                ):
                    activity_keys.add(self._binding_key(turn_binding))
                for binding_key in activity_keys:
                    self._refresh_group_turn_for_key(binding_key, sweeps, sweep_complete_at)
                self._ack_group_turns([] if turn_binding is None else [turn_binding], sweeps, sweep_complete_at)
                has_work = bool(bindings or self._entries or self._restart_due or self._close_queue or sweeps)
                if self._stop.wait(_WORKER_WAIT_SECONDS if has_work else _WORKER_IDLE_SECONDS):
                    break
        finally:
            self._shutdown(sweeps)

    @staticmethod
    def _binding_key(binding: ProjectBinding) -> tuple[str, str, str]:
        return (
            binding.project_id,
            binding.registration_generation or "",
            str(binding.shared_root),
        )

    @classmethod
    def _group_key(cls, binding: ProjectBinding, group: str) -> tuple[str, str, str, str]:
        return (*cls._binding_key(binding), group)

    def _reconcile_bindings(
        self,
        registered: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        """Retire only binding objects that changed across a registry refresh."""

        current = {self._binding_key(binding): binding for binding in registered}
        for binding_key, previous in tuple(self._bindings_by_key.items()):
            replacement = current.get(binding_key)
            if replacement == previous:
                continue
            self._closing_binding_keys.add(binding_key)
            for key, entry in tuple(self._entries.items()):
                if entry.binding_key == binding_key:
                    self._queue_entry_close(entry, block_binding=True)
            for key, entry in tuple(self._restart_due.items()):
                if entry.binding_key == binding_key:
                    self._restart_due.pop(key, None)
            sweep = sweeps.get(binding_key)
            if sweep is not None and self._queue_sweep_close(sweep, binding_key, block_binding=True):
                sweeps.pop(binding_key, None)
            if self._close_work_counts.get(binding_key, 0) == 0:
                self._closing_binding_keys.discard(binding_key)
        for binding_key, sweep in tuple(sweeps.items()):
            if binding_key not in current and self._queue_sweep_close(sweep, binding_key, block_binding=True):
                sweeps.pop(binding_key, None)
        self._bindings_by_key = current
        self._lane_turns = {key: turn for key, turn in self._lane_turns.items() if key in current}
        self._group_turns = {key: turn for key, turn in self._group_turns.items() if key in current}
        self._legacy_sweep_passes = {key: count for key, count in self._legacy_sweep_passes.items() if key in current}
        self._locator_traversals = {
            key: traversal for key, traversal in self._locator_traversals.items() if key[0] in current
        }

    def _ensure_group_turn(
        self,
        binding: ProjectBinding,
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        key = self._binding_key(binding)
        if key not in self._group_turns:
            self._refresh_group_turn_for_key(key, sweeps, complete_at)

    def _refresh_group_turn_for_key(
        self,
        binding_key: tuple[str, str, str],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        binding = self._bindings_by_key.get(binding_key)
        if binding is None:
            return
        previous = self._group_turns.get(binding_key)
        if previous is not None:
            self._runtime.working_set.acknowledge(previous[0], quiescent=False)
        from .locator import LANES as locator_lanes

        starts = {
            lane: getattr(self._locator_traversals.get((binding_key, lane)), "completed_passes", 0)
            for lane in locator_lanes
        }
        starts["legacy"] = self._legacy_sweep_passes.get(binding_key, 0)
        self._group_turns[binding_key] = (self._runtime.working_set.begin_turn(binding, "group"), starts)

    def _ack_group_turns(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        from .locator import LANES as locator_lanes

        resource_binding_keys = self._resource_binding_keys(sweeps)
        for binding in bindings:
            binding_key = self._binding_key(binding)
            turn_state = self._group_turns.get(binding_key)
            if turn_state is None:
                continue
            turn, starting_passes = turn_state
            if binding_key in resource_binding_keys:
                self._refresh_group_turn_for_key(binding_key, sweeps, complete_at)
                continue
            activation_state = self._activation_state(binding.shared_root)
            if activation_state == "active":
                complete = all(
                    getattr(self._locator_traversals.get((binding_key, lane)), "completed_passes", 0)
                    >= starting_passes.get(lane, 0) + 1
                    for lane in locator_lanes
                )
            elif activation_state in {None, "preparing", "fenced", "building"}:
                complete = self._legacy_sweep_passes.get(binding_key, 0) > starting_passes.get("legacy", 0)
            else:
                complete = False
            if not complete:
                continue
            if self._runtime.working_set.is_lane_quiescent(binding, "group"):
                continue
            if not self._runtime.working_set.is_current_turn(turn):
                self._refresh_group_turn_for_key(binding_key, sweeps, complete_at)
                continue
            self._runtime.working_set.acknowledge(turn, quiescent=True)

    def _ack_disabled_bindings(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        """Acknowledge disabled bindings only after their owned resources close."""

        resource_binding_keys = self._resource_binding_keys(sweeps)
        for binding in bindings:
            binding_key = self._binding_key(binding)
            if binding_key in resource_binding_keys:
                continue
            if self._runtime.working_set.is_lane_quiescent(binding, "group"):
                continue
            turn = self._runtime.working_set.begin_turn(binding, "group")
            self._runtime.working_set.acknowledge(turn, quiescent=True)

    def _resource_binding_keys(
        self,
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> set[tuple[str, str, str]]:
        """Index binding-owned resources once for linear acknowledgement passes."""

        keys = {entry.binding_key for entry in self._entries.values()}
        keys.update(entry.binding_key for entry in self._restart_due.values())
        keys.update(key for key, count in self._close_work_counts.items() if count)
        keys.update(sweeps)
        return keys

    def _eligible_binding(self, binding: ProjectBinding) -> bool:
        """Check one binding's live eligibility without scanning the registry."""

        if not binding.enabled or binding.project_id in self._runtime.upgrade_admission_blocked_projects:
            return False
        try:
            if self._runtime.binding_state(binding) != "enabled":
                return False
            return is_group_authority_isolated(binding.shared_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False

    def _eligible_bindings(self, bindings: list[ProjectBinding]) -> list[ProjectBinding]:
        result: list[ProjectBinding] = []
        for binding in bindings:
            if self._eligible_binding(binding):
                result.append(binding)
        return result

    def _resident_entry_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(entry): entry for entry in self._restart_due.values()})
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return len(entries)

    def _parser_owner_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return sum(1 for entry in entries.values() if entry.service is not None and entry.service.owns_source_parser)

    @staticmethod
    def _activation_state(root: Path) -> str | None:
        from .activation import group_service_routing_state

        return group_service_routing_state(root)

    def _close_legacy_binding_owners(
        self,
        binding_key: tuple[str, str, str],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        sweep = sweeps.get(binding_key)
        if sweep is not None and self._queue_sweep_close(sweep, binding_key):
            sweeps.pop(binding_key, None)
        for entry in tuple(self._entries.values()):
            if entry.binding_key == binding_key and not entry.locator_mode:
                if len(self._close_work_index) >= _MAX_CLOSE_WORK:
                    return
                self._queue_entry_close(entry)

    def _advance_active_traversal(
        self,
        index: int,
        binding: ProjectBinding,
        key: tuple[str, str, str],
    ) -> tuple[int, ProjectBinding, str] | None:
        turn = self._lane_turns.get(key, 0)
        lane = _LOCATOR_LANE_CYCLE[turn]
        self._lane_turns[key] = (turn + 1) % len(_LOCATOR_LANE_CYCLE)
        traversal_key = (key, lane)
        try:
            from .locator import GroupLocatorTraversal

            traversal = self._locator_traversals.get(traversal_key)
            if traversal is None:
                traversal = GroupLocatorTraversal(binding.shared_root, lane)
                self._locator_traversals[traversal_key] = traversal
            record = traversal.advance()
            self._lane_metrics[lane]["reads"] += 1
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._lane_metrics[lane]["retries"] += 1
            return None
        if record is None:
            return None
        self._lane_metrics[lane]["pending_locators_encountered"] += 1
        self._lane_metrics[lane]["bytes"] += len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        group = record["identity"]["group"]
        try:
            read_group(binding.shared_root, group)
        except (FileNotFoundError, GroupNotPublished, GroupPublicationUnavailable):
            # Preserve the locator and let this lane retry after other digests.
            self._lane_metrics[lane]["retries"] += 1
            return None
        self._pending_locator = (lane, record)
        return index, binding, group

    def _next_group(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
        start: int,
    ) -> tuple[int, ProjectBinding, str] | None:
        self._pending_locator = None
        self._active_project_slice = False
        if not bindings:
            return None
        index = start % len(bindings)
        binding = bindings[index]
        key = self._binding_key(binding)
        if key in self._closing_binding_keys:
            return None
        known_activation_state = self._activation_state(binding.shared_root)
        if (
            known_activation_state not in {"active", "degraded"}
            and key not in sweeps
            and complete_at.get(key, 0.0) > time.monotonic()
        ):
            return None
        if not self._eligible_binding(binding):
            return None
        activation_state = known_activation_state
        if activation_state == "active":
            self._active_project_slice = True
            self._close_legacy_binding_owners(key, sweeps)
            return self._advance_active_traversal(index, binding, key)
        if activation_state == "degraded":
            self._active_project_slice = True
            self._close_legacy_binding_owners(key, sweeps)
            return None
        if key not in sweeps:
            try:
                sweeps[key] = SubmissionSourceSweep(group_directory(binding.shared_root), page_size=1)
            except (OSError, RuntimeError, ValueError, TypeError):
                self._lane_metrics["control"]["retries"] += 1
                complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                return None
        sweep = sweeps[key]
        io = SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS)
        try:
            step = sweep.advance(
                io,
                max_entries=1,
                soft_deadline=time.monotonic() + _SOURCE_SLICE_SECONDS,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._lane_metrics["control"]["retries"] += 1
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            if self._queue_sweep_close(sweep, key):
                sweeps.pop(key, None)
            return None
        if step.candidates:
            group = step.candidates[0].stem
            try:
                read_group(binding.shared_root, group)
            except (FileNotFoundError, GroupNotPublished, GroupPublicationUnavailable):
                self._lane_metrics["membership"]["retries"] += 1
                return None
            return index, binding, group
        if step.state == "complete":
            sweeps.pop(key, None)
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            self._legacy_sweep_passes[key] = self._legacy_sweep_passes.get(key, 0) + 1
        elif step.state == "failed":
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            if self._queue_sweep_close(sweep, key):
                sweeps.pop(key, None)
        return None

    def _new_entry(
        self,
        binding: ProjectBinding,
        group: str,
        *,
        locator_lane: str | None = None,
        locator_record: dict[str, Any] | None = None,
    ) -> _GroupServiceEntry:
        locator_mode = locator_record is not None
        service: GroupDiscoveryService | None = None
        maintenance: Any | None = None
        if not locator_mode:
            service = GroupDiscoveryService(binding.shared_root, group)
        elif locator_lane == "membership":
            service = GroupDiscoveryService(
                binding.shared_root,
                group,
                mode="locator",
                locator_generation=locator_record["generation"],
            )
        if not locator_mode or locator_lane == "maintenance":
            maintenance = self._new_maintenance(binding.shared_root, group)
        return _GroupServiceEntry(
            key=self._group_key(binding, group),
            binding_key=self._binding_key(binding),
            binding=binding,
            group=group,
            service=service,
            maintenance=maintenance,
            locator_records={locator_lane: locator_record} if locator_record is not None and locator_lane else {},
            locator_observed_at={locator_lane: time.monotonic()} if locator_record is not None and locator_lane else {},
            locator_mode=locator_mode,
            pending_lane=locator_lane,
        )

    @staticmethod
    def _new_maintenance(root: Path, group: str) -> Any:
        # Each retained Group owns a separately budgeted maintenance session.
        from .maintenance import GroupMaintenance

        return GroupMaintenance(root, group)

    def _admit_group(self, binding: ProjectBinding, group: str) -> None:
        key = self._group_key(binding, group)
        if self._pending_locator is not None:
            lane, record = self._pending_locator
            existing = self._entries.get(key)
            candidate_count = sum(len(entry.locator_records) for entry in self._entries.values())
            if existing is None and (
                key in self._closing_group_keys
                or self._resident_entry_count() >= _MAX_RESIDENT_GROUPS
                or candidate_count >= _MAX_LOCATOR_CANDIDATES
            ):
                self._lane_metrics[lane]["cap_refusals"] += 1
                if key not in self._closing_group_keys:
                    self._evict_one_locator_entry(exclude=key)
                return
            if (
                existing is not None
                and lane not in existing.locator_records
                and candidate_count >= _MAX_LOCATOR_CANDIDATES
            ):
                self._lane_metrics[lane]["cap_refusals"] += 1
                self._evict_one_locator_entry(exclude=key)
                return
            if existing is None:
                try:
                    existing = self._new_entry(
                        binding,
                        group,
                        locator_lane=lane,
                        locator_record=record,
                    )
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    self._lane_metrics[lane]["retries"] += 1
                    return
                self._entries[key] = existing
                self._entry_queue.append(existing)
            else:
                existing.locator_mode = True
                existing.locator_records[lane] = record
                existing.locator_observed_at.setdefault(lane, time.monotonic())
                existing.pending_lane = lane
                if lane == "membership":
                    if existing.service is None:
                        existing.service = GroupDiscoveryService(
                            binding.shared_root,
                            group,
                            mode="locator",
                            locator_generation=record["generation"],
                        )
                    else:
                        existing.service.update_locator_generation(record["generation"])
                elif lane == "maintenance" and existing.maintenance is None:
                    existing.maintenance = self._new_maintenance(binding.shared_root, group)
            self._refresh_lane_residency()
            return
        if key in self._entries or key in self._restart_due or key in self._closing_group_keys:
            return
        if self._resident_entry_count() >= _MAX_RESIDENT_GROUPS:
            self._lane_metrics["membership"]["cap_refusals"] += 1
            return
        try:
            entry = self._new_entry(binding, group)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return
        self._entries[key] = entry
        self._entry_queue.append(entry)

    def _evict_one_locator_entry(self, *, exclude: tuple[str, str, str, str]) -> bool:
        """Release one durable-locator owner so a later Group can enter the cap."""

        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return False
        for entry in tuple(self._entry_queue):
            if entry.key == exclude or not entry.locator_mode or self._entries.get(entry.key) is not entry:
                continue
            self._queue_entry_close(entry)
            return True
        return False

    def _refresh_lane_residency(self) -> None:
        descriptors = self._service_descriptor_count()
        for lane in self._lane_metrics:
            self._lane_metrics[lane]["resident_entries"] = sum(
                lane in entry.locator_records for entry in self._entries.values()
            )
            self._lane_metrics[lane]["descriptor_high_water"] = max(
                self._lane_metrics[lane]["descriptor_high_water"], descriptors
            )

    def _restart_one_due(self) -> ProjectBinding | None:
        if not self._restart_due:
            return None
        key = next(iter(self._restart_due))
        old = self._restart_due.pop(key)
        binding = self._bindings_by_key.get(old.binding_key)
        if binding != old.binding:
            return None
        now = time.monotonic()
        if old.cooldown_until > now or not self._eligible_binding(binding):
            self._restart_due[key] = old
            return None
        try:
            read_group(binding.shared_root, old.group)
            if not group_path(binding.shared_root, old.group).exists():
                return binding
            entry = self._new_entry(binding, old.group)
        except FileNotFoundError:
            return binding
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            old.cooldown_until = now + _WORKER_COOLDOWN_SECONDS
            self._restart_due[key] = old
            return binding
        self._entries[key] = entry
        self._entry_queue.append(entry)
        return binding

    def _advance_one_entry(self) -> ProjectBinding | None:
        for _ in range(len(self._entry_queue)):
            entry = self._entry_queue.popleft()
            if self._entries.get(entry.key) is not entry:
                continue
            self._entry_queue.append(entry)
            if not self._eligible_binding(entry.binding):
                continue
            if entry.binding_key in self._closing_binding_keys:
                self._queue_entry_close(entry, block_binding=True)
                return entry.binding
            if entry.locator_mode:
                if not entry.locator_records:
                    self._queue_entry_close(entry)
                    return entry.binding
                lane = entry.pending_lane or next(iter(entry.locator_records), None)
                record = entry.locator_records.get(lane) if lane is not None else None
                if lane is not None and record is not None:
                    self._advance_active_locator(entry.binding, entry.group, lane, record)
                    return entry.binding
                continue
            try:
                read_group(entry.binding.shared_root, entry.group)
                if not group_path(entry.binding.shared_root, entry.group).exists():
                    self._queue_entry_close(entry)
                    continue
            except FileNotFoundError:
                self._queue_entry_close(entry)
                continue
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
            if entry.service_turn:
                entry.service_turn = False
                if entry.cooldown_until > time.monotonic():
                    continue
                result = self._advance_current(entry.service)
                state = result.get("state")
                if state == "complete" or state == "ambiguous":
                    entry.cooldown_until = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                elif state in {"error", "closed"}:
                    entry.cooldown_until = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                    self._queue_entry_close(entry, restart=True)
                return entry.binding
            entry.service_turn = True
            try:
                entry.maintenance.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                # Maintenance owns its own retry state. A transient maintenance
                # failure never retires a healthy discovery service.
                return entry.binding
            return entry.binding
        return None

    def _advance_active_locator(
        self,
        binding: ProjectBinding,
        group: str,
        lane: str,
        observed: dict[str, Any],
    ) -> None:
        key = self._group_key(binding, group)
        entry = self._entries.get(key)
        if entry is None or not entry.locator_mode:
            return
        now = time.monotonic()
        if entry.locator_cooldown_until.get(lane, 0.0) > now:
            return
        from . import locator

        try:
            current = locator.read_group_locator(binding.shared_root, group, lane)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._schedule_entry_retry(entry, lane)
            return
        if current is None:
            entry.locator_records.pop(lane, None)
            entry.locator_observed_at.pop(lane, None)
            entry.locator_retry_delay.pop(lane, None)
            entry.locator_cooldown_until.pop(lane, None)
            entry.pending_lane = next(iter(entry.locator_records), None)
            self._refresh_lane_residency()
            if lane == "membership" and entry.service is not None:
                self._queue_entry_close(entry)
            elif not entry.locator_records:
                self._queue_entry_close(entry)
            return
        observed = current
        entry.locator_records[lane] = current
        entry.pending_lane = lane
        self._refresh_lane_residency()
        if lane == "membership":
            service = entry.service
            if service is None:
                service = GroupDiscoveryService(
                    binding.shared_root,
                    group,
                    mode="locator",
                    locator_generation=observed["generation"],
                )
                entry.service = service
            else:
                service.update_locator_generation(observed["generation"])
            parser_owners = self._parser_owner_count()
            if not service.owns_source_parser and parser_owners >= _MAX_PARSER_OWNERS:
                self._lane_metrics[lane]["cap_refusals"] += 1
                return
            if self._service_descriptor_count() >= _MAX_SERVICE_DESCRIPTORS:
                self._lane_metrics[lane]["cap_refusals"] += 1
                return
            result = self._advance_current(service)
            state = result.get("state")
            if state == "complete":
                try:
                    acknowledged = service.acknowledge_if_quiescent()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    acknowledged = False
                if acknowledged:
                    entry.locator_records.pop(lane, None)
                    entry.locator_observed_at.pop(lane, None)
                    entry.locator_retry_delay.pop(lane, None)
                    entry.locator_cooldown_until.pop(lane, None)
                    entry.pending_lane = next(iter(entry.locator_records), None)
                    entry.service = None
                    self._lane_metrics[lane]["completions"] += 1
                    self._lane_metrics[lane]["writes"] += 1
                    self._refresh_lane_residency()
                    if not entry.locator_records:
                        self._queue_entry_close(entry)
                return
            if state in {"error", "ambiguous"}:
                self._schedule_entry_retry(entry, lane)
            else:
                entry.locator_retry_delay[lane] = 1.0
                entry.locator_cooldown_until.pop(lane, None)
            return
        if lane == "control":
            if self._advance_control_locator(entry, observed):
                entry.locator_records.pop(lane, None)
                entry.locator_observed_at.pop(lane, None)
                entry.locator_retry_delay.pop(lane, None)
                entry.locator_cooldown_until.pop(lane, None)
                entry.pending_lane = next(iter(entry.locator_records), None)
                self._lane_metrics[lane]["completions"] += 1
                self._lane_metrics[lane]["writes"] += 1
                self._refresh_lane_residency()
                if not entry.locator_records:
                    self._queue_entry_close(entry)
            return
        if lane == "maintenance":
            maintenance = entry.maintenance
            if maintenance is None:
                maintenance = self._new_maintenance(binding.shared_root, group)
                entry.maintenance = maintenance
            try:
                result = maintenance.advance(locator_generation=observed["generation"])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                self._schedule_entry_retry(entry, lane)
                return
            if result.get("state") in {"error", "ambiguous", "waiting"}:
                self._schedule_entry_retry(entry, lane)
                return
            entry.locator_retry_delay[lane] = 1.0
            entry.locator_cooldown_until.pop(lane, None)
            try:
                acknowledged = maintenance.acknowledge_if_quiescent(observed["generation"])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                acknowledged = False
            if acknowledged:
                entry.locator_records.pop(lane, None)
                entry.locator_observed_at.pop(lane, None)
                entry.locator_retry_delay.pop(lane, None)
                entry.locator_cooldown_until.pop(lane, None)
                entry.pending_lane = next(iter(entry.locator_records), None)
                entry.maintenance = None
                self._lane_metrics[lane]["completions"] += 1
                self._lane_metrics[lane]["writes"] += 1
                self._refresh_lane_residency()
                if not entry.locator_records:
                    self._queue_entry_close(entry)
            return
        raise RuntimeError(f"unknown Group service lane {lane!r}")

    def _service_descriptor_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return sum(
            (1 if entry.service is not None and entry.service.owns_source_parser else 0)
            + (entry.maintenance.open_descriptor_count if entry.maintenance is not None else 0)
            for entry in entries.values()
        )

    def _schedule_entry_retry(self, entry: _GroupServiceEntry, lane: str) -> None:
        delay = entry.locator_retry_delay.get(lane, 1.0)
        entry.locator_cooldown_until[lane] = time.monotonic() + delay
        entry.locator_retry_delay[lane] = min(delay * 2, 60.0)
        self._lane_metrics[lane]["retries"] += 1

    def _advance_control_locator(self, entry: _GroupServiceEntry, observed: dict[str, Any]) -> bool:
        binding = entry.binding
        group = entry.group
        cfg = binding.root_config()
        try:
            from ...commands.group import reconcile_group_cancel_operations

            reconcile_group_cancel_operations(
                cfg,
                group,
                include_legacy=False,
                reservation_runtime_root=cfg.runtime_root,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._schedule_entry_retry(entry, "control")
            return False

        try:
            state = advance_group_control_rechecks(cfg, group, observed["generation"], entry.control_cursor)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._schedule_entry_retry(entry, "control")
            return False
        if state == "blocked":
            self._schedule_entry_retry(entry, "control")
        else:
            entry.locator_retry_delay["control"] = 1.0
            entry.locator_cooldown_until.pop("control", None)
        return state == "quiescent"

    def _queue_entry_close(
        self,
        entry: _GroupServiceEntry,
        *,
        restart: bool = False,
        block_binding: bool = False,
    ) -> None:
        close_id = ("entry", id(entry))
        if close_id in self._close_work_index:
            return
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return
        if self._entries.get(entry.key) is entry:
            self._entries.pop(entry.key, None)
        for lane in entry.locator_records:
            self._lane_metrics[lane]["evictions"] += 1
        if block_binding:
            self._closing_binding_keys.add(entry.binding_key)
            entry.restart_after_close = False
            self._restart_due.pop(entry.key, None)
        elif restart:
            entry.restart_after_close = True
        self._close_work_index[close_id] = _CloseWork(
            "entry",
            entry,
            binding_key=entry.binding_key,
        )
        self._close_queue.append(self._close_work_index[close_id])
        self._closing_group_keys.add(entry.key)
        self._close_work_counts[entry.binding_key] = self._close_work_counts.get(entry.binding_key, 0) + 1

    def _queue_sweep_close(
        self,
        sweep: SubmissionSourceSweep,
        binding_key: tuple[str, str, str],
        *,
        block_binding: bool = False,
    ) -> bool:
        close_id = ("sweep", id(sweep))
        if close_id in self._close_work_index:
            return True
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return False
        if block_binding:
            self._closing_binding_keys.add(binding_key)
        work = _CloseWork("sweep", sweep, binding_key=binding_key)
        self._close_work_index[close_id] = work
        self._close_queue.append(work)
        self._close_work_counts[binding_key] = self._close_work_counts.get(binding_key, 0) + 1
        return True

    def _queue_simple_close(self, kind: str, owner: Any, binding_key: tuple[str, str, str]) -> None:
        close_id = (kind, id(owner))
        if close_id in self._close_work_index:
            return
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return
        work = _CloseWork(kind, owner, binding_key=binding_key)
        self._close_work_index[close_id] = work
        self._close_queue.append(work)
        self._close_work_counts[binding_key] = self._close_work_counts.get(binding_key, 0) + 1

    def _advance_close_one(self) -> tuple[str, str, str] | None:
        if not self._close_queue:
            return None
        work = self._close_queue.popleft()
        try:
            done = self._advance_close_work(work)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            done = False
        if done:
            self._finish_close_work(work)
        else:
            self._close_queue.append(work)
        return work.binding_key

    def _advance_close_work(self, work: _CloseWork) -> bool:
        if work.kind == "entry":
            entry = work.owner
            if work.phase == "service":
                if entry.service is None:
                    work.phase = "maintenance"
                    work.requested = False
                    return False
                if not work.requested:
                    entry.service.request_close()
                    work.requested = True
                if entry.service.is_closed:
                    work.phase = "maintenance"
                    work.requested = False
                    return False
                entry.service.advance()
                if entry.service.is_closed:
                    work.phase = "maintenance"
                    work.requested = False
                return False
            if entry.maintenance is None:
                return True
            if not work.requested:
                entry.maintenance.request_close()
                work.requested = True
            if entry.maintenance.is_closed:
                return True
            entry.maintenance.advance()
            return entry.maintenance.is_closed
        owner = work.owner
        if not work.requested:
            owner.request_close()
            work.requested = True
        if owner.is_closed:
            return True
        if work.kind == "sweep":
            owner.advance(
                SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS),
                max_entries=1,
            )
        elif work.kind == "service":
            owner.advance()
        elif work.kind == "maintenance":
            owner.advance()
        else:
            raise RuntimeError(f"unknown Group discovery close work kind: {work.kind}")
        return owner.is_closed

    def _finish_close_work(self, work: _CloseWork) -> None:
        self._close_work_index.pop((work.kind, id(work.owner)), None)
        binding_key = work.binding_key
        if binding_key is not None:
            remaining = self._close_work_counts.get(binding_key, 1) - 1
            if remaining <= 0:
                self._close_work_counts.pop(binding_key, None)
                self._closing_binding_keys.discard(binding_key)
            else:
                self._close_work_counts[binding_key] = remaining
        if work.kind == "entry":
            entry = work.owner
            self._closing_group_keys.discard(entry.key)
            if (
                entry.restart_after_close
                and self._bindings_by_key.get(entry.binding_key) == entry.binding
                and entry.key not in self._entries
            ):
                self._restart_due[entry.key] = entry

    def _shutdown(self, sweeps: dict[tuple[str, str, str], SubmissionSourceSweep]) -> None:
        """Drain all remaining owners after the daemon loop stops."""

        entries: dict[int, _GroupServiceEntry] = {}
        for entry in self._entries.values():
            entries[id(entry)] = entry
        for entry in self._restart_due.values():
            entries[id(entry)] = entry
        for entry in self._entry_queue:
            entries[id(entry)] = entry
        for work in self._close_queue:
            if work.kind == "entry":
                entries[id(work.owner)] = work.owner
        for entry in entries.values():
            self._close_service(entry.service)
            self._close_maintenance(entry.maintenance)
        for work in self._close_queue:
            if work.kind == "sweep":
                self._close_sweep(work.owner)
            elif work.kind == "service":
                self._close_service(work.owner)
            elif work.kind == "maintenance":
                self._close_maintenance(work.owner)
        for sweep in sweeps.values():
            self._close_sweep(sweep)

    @staticmethod
    def _advance_current(service: GroupDiscoveryService) -> dict[str, object]:
        try:
            return service.advance()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {
                "state": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "prefix": 0,
                "tail": 0,
            }

    @staticmethod
    def _should_rotate(state: object, reason: object) -> bool:
        return state in {"error", "closed"}

    @staticmethod
    def _close_service(service: GroupDiscoveryService | None) -> None:
        if service is None:
            return
        service.request_close()
        while not service.is_closed:
            try:
                service.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not service.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)
        # Ownership remains on this daemon thread after stop's bounded join.

    @staticmethod
    def _close_sweep(sweep: SubmissionSourceSweep) -> None:
        sweep.request_close()
        while not sweep.is_closed:
            try:
                sweep.advance(
                    SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS),
                    max_entries=1,
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not sweep.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)

    @staticmethod
    def _close_maintenance(maintenance: Any) -> None:
        if maintenance is None:
            return
        maintenance.request_close()
        while not maintenance.is_closed:
            try:
                maintenance.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not maintenance.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)


__all__ = ["GroupDiscoveryService", "MachineGroupDiscoveryWorker", "publish_submission_debt"]
