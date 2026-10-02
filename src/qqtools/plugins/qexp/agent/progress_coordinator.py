"""Bounded local progress sampling coordinated with typed shared transactions."""

from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from qqtools.qexp._progress_protocol import read_advisory_snapshot, replace_advisory_snapshot

from ..config_types import RootConfig
from ..runtime.directory_capture import read_directory_entry
from ..runtime.progress import ProgressProjector, _mailbox_signature
from ..runtime.progress_v2 import ProgressV2Projector
from .bindings import ProjectBinding
from .progress_transport import progress_context

_MAX_PROJECTORS = 64
_MAX_PROJECTS_PER_SLICE = 16
_MAX_LOCAL_ENTRIES_PER_SLICE = 8
_CADENCE_DEADLINES = ("cache_next_due", "shared_next_due")
_CADENCE_FLAGS = (
    "cache_initial_available",
    "shared_initial_available",
    "cache_final_available",
    "shared_final_available",
)


@dataclass(slots=True)
class _ProjectProgress:
    binding: ProjectBinding
    projectors: dict[int, Any] = field(default_factory=dict)
    offsets: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    next_version: int = 1
    parameters: dict[str, Any] | None = None
    observed_binding: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    retry_at: float = 0.0
    can_evict: bool = True
    scan_checkpoint: tuple[int, int, int, float] = (0, 0, 1, 0.0)
    parking_retry_at: float = 0.0

    def close(self) -> None:
        for projector in self.projectors.values():
            projector.close()


class ProgressObservationCoordinator:
    """Keep cadence local; never inspect or publish shared Project state inline."""

    def __init__(self, runtime: Any, *, clock=time.monotonic) -> None:
        self.runtime = runtime
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str | None, str], _ProjectProgress] = OrderedDict()
        self._cursor = 0
        self._clock_scope = uuid.uuid4().hex

    @staticmethod
    def _key(binding: ProjectBinding) -> tuple[str, str | None, str]:
        return binding.project_id, binding.registration_generation, str(binding.shared_root)

    def close(self) -> None:
        for entry in self._entries.values():
            entry.close()
        self._entries.clear()

    def _checkpoint_root(self, entry: _ProjectProgress) -> Path:
        return self.runtime.project_paths(entry.binding.project_id)["root"] / "progress-coordinator"

    def _restore_scan(self, entry: _ProjectProgress) -> None:
        try:
            saved = read_advisory_snapshot(self._checkpoint_root(entry) / "scan.json", max_bytes=4096)
            offsets = saved["offsets"]
            if (
                saved["clock_scope"] != self._clock_scope
                or saved["generation"] != entry.binding.registration_generation
                or saved["shared_root"] != str(entry.binding.shared_root)
                or type(saved["next_version"]) is not int
                or saved["next_version"] not in (1, 2)
                or not isinstance(offsets, dict)
                or set(offsets) != {"1", "2"}
                or any(type(value) is not int or not 0 <= value < (1 << 63) for value in offsets.values())
                or type(saved["retry_at"]) not in (int, float)
                or not math.isfinite(saved["retry_at"])
                or saved["retry_at"] < 0
            ):
                return
            entry.offsets = {version: offsets[str(version)] for version in (1, 2)}
            entry.next_version = saved["next_version"]
            entry.retry_at = saved["retry_at"]
            entry.scan_checkpoint = self._scan_signature(entry)
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            pass

    @staticmethod
    def _scan_signature(entry: _ProjectProgress) -> tuple[int, int, int, float]:
        return entry.offsets[1], entry.offsets[2], entry.next_version, entry.retry_at

    def _park(self, entry: _ProjectProgress) -> bool:
        if not entry.can_evict or entry.parking_retry_at > self._clock():
            return False
        signature = self._scan_signature(entry)
        if signature == entry.scan_checkpoint:
            entry.close()
            return True
        try:
            self._checkpoint_root(entry).mkdir(exist_ok=True)
            replace_advisory_snapshot(
                self._checkpoint_root(entry) / "scan.json",
                {
                    "clock_scope": self._clock_scope,
                    "generation": entry.binding.registration_generation,
                    "shared_root": str(entry.binding.shared_root),
                    "offsets": {str(version): offset for version, offset in entry.offsets.items()},
                    "next_version": entry.next_version,
                    "retry_at": entry.retry_at,
                },
            )
        except (OSError, ValueError, TypeError):
            entry.parking_retry_at = self._clock() + 1.0
            return False
        entry.scan_checkpoint = signature
        entry.close()
        return True

    def _cadence_path(self, entry: _ProjectProgress, context: Mapping[str, Any]) -> Path:
        return self._checkpoint_root(entry) / f"v{context['protocol_version']}" / f"{context['attempt_id']}.json"

    def _restore_cadence(self, entry: _ProjectProgress, projector: Any, context: dict[str, Any]) -> None:
        attempt_id = context["attempt_id"]
        if attempt_id in projector._entries:
            return
        try:
            saved = read_advisory_snapshot(self._cadence_path(entry, context), max_bytes=65_536)
            cadence = saved["cadence"]
            if (
                saved["clock_scope"] != self._clock_scope
                or saved["generation"] != entry.binding.registration_generation
                or saved["context"] != context
                or not isinstance(cadence, dict)
                or set(cadence) != set(_CADENCE_DEADLINES + _CADENCE_FLAGS)
                or any(type(cadence[field]) is not bool for field in _CADENCE_FLAGS)
                or any(
                    cadence[field] is not None
                    and (
                        type(cadence[field]) not in (int, float)
                        or not math.isfinite(cadence[field])
                        or cadence[field] < 0
                    )
                    for field in _CADENCE_DEADLINES
                )
            ):
                return
            # Payloads and publication truth are freshly restored from existing
            # snapshots. Only scheduling cadence survives a cache eviction.
            state = projector._restore(attempt_id, entry.observed_binding, interval=context["interval_seconds"])
            state.update(cadence)
            for field in _CADENCE_DEADLINES:
                if state[field] is None:
                    state[field] = float("-inf")
            projector._entries[attempt_id] = state
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            pass

    def _save_cadence(self, entry: _ProjectProgress, context: dict[str, Any]) -> None:
        projector = entry.projectors.get(context["protocol_version"])
        state = None if projector is None else projector._entries.get(context["attempt_id"])
        path = self._cadence_path(entry, context)
        try:
            if state is None:
                path.unlink(missing_ok=True)
            elif all(field in state for field in _CADENCE_DEADLINES + _CADENCE_FLAGS):
                path.parent.mkdir(parents=True, exist_ok=True)
                replace_advisory_snapshot(
                    path,
                    {
                        "clock_scope": self._clock_scope,
                        "generation": entry.binding.registration_generation,
                        "context": context,
                        "cadence": {
                            **{
                                field: state[field] if math.isfinite(state[field]) else None
                                for field in _CADENCE_DEADLINES
                            },
                            **{field: state[field] for field in _CADENCE_FLAGS},
                        },
                    },
                )
            entry.can_evict = True
        except (OSError, ValueError, TypeError):
            # Retain the owner rather than reset a successfully advanced clock.
            entry.can_evict = False

    def _capture(self, entry: _ProjectProgress) -> None:
        root = self.runtime.project_paths(entry.binding.project_id)["root"]
        # Count every local name and alternate protocol directories. Streams
        # close on each entry, so neither idle projects nor eviction retain FDs.
        for _ in range(_MAX_LOCAL_ENTRIES_PER_SLICE):
            version = entry.next_version
            entry.next_version = 3 - version
            directory = Path(root) / ("progress-contexts" if version == 1 else "progress-v2-contexts")
            try:
                name, offset = read_directory_entry(directory, entry.offsets[version])
            except FileNotFoundError:
                entry.offsets[version] = 0
                continue
            entry.offsets[version] = offset if name is not None else 0
            if name is None or not name.endswith(".json"):
                continue
            try:
                context = progress_context(read_advisory_snapshot(directory / name))
                if context["attempt_id"] != name[:-5] or context["protocol_version"] != version:
                    continue
                if context["machine_name"] != entry.binding.machine_name:
                    continue
                mailbox = (
                    Path(root)
                    / "progress"
                    / context["attempt_id"]
                    / ("latest.json" if version == 1 else "latest-v2.json")
                )
                if _mailbox_signature(mailbox) is None:
                    continue
            except (OSError, ValueError, TypeError, RecursionError):
                continue
            entry.parameters = {"context": context, "projection": None}
            self.runtime.working_set.activate(entry.binding, "local_progress")
            return

    def _projector(self, entry: _ProjectProgress, version: int) -> Any:
        projector = entry.projectors.get(version)
        if projector is not None:
            return projector

        def resolve(_cfg: Any, context: dict[str, Any]) -> dict[str, Any] | None:
            if entry.parameters is None or context != entry.parameters["context"]:
                raise ValueError("progress observation does not match the captured local context")
            return entry.observed_binding

        def publish(context: dict[str, Any], binding: dict[str, Any], latest: dict[str, Any]) -> str:
            if entry.parameters is None or context != entry.parameters["context"] or binding != entry.observed_binding:
                raise ValueError("progress projection does not match its observation")
            entry.parameters = {"context": dict(context), "projection": dict(latest)}
            return "deferred"

        cls = ProgressProjector if version == 1 else ProgressV2Projector
        cfg = RootConfig.from_canonical_paths(
            entry.binding.shared_root,
            entry.binding.shared_root.parent,
            entry.binding.machine_name,
            self.runtime.project_paths(entry.binding.project_id)["root"],
        )
        projector = cls(
            cfg,
            registration_generation=entry.binding.registration_generation,
            resolver=resolve,
            shared_snapshot=lambda _binding: entry.snapshot,
            shared_publish=publish,
            clock=self._clock,
        )
        entry.projectors[version] = projector
        return projector

    def _apply(self, entry: _ProjectProgress, evidence: Mapping[str, Any]) -> None:
        parameters = entry.parameters
        if parameters is None:
            return
        context = parameters["context"]
        version = context["protocol_version"]
        root = self.runtime.project_paths(entry.binding.project_id)["root"]
        directory = "progress-contexts" if version == 1 else "progress-v2-contexts"
        try:
            if read_advisory_snapshot(Path(root) / directory / f"{context['attempt_id']}.json") != context:
                entry.parameters = None
                return
        except (OSError, ValueError, TypeError, RecursionError):
            entry.parameters = None
            return
        state = evidence.get("state")
        if state is None or state == "blocked":
            entry.retry_at = self._clock() + 1.0
            return
        projector = self._projector(entry, version)
        attempt_id = context["attempt_id"]
        if state == "retired":
            projector._retire(attempt_id)
            entry.parameters = None
        elif state == "stale":
            projector._entries.pop(attempt_id, None)
            entry.parameters = None
        elif state == "published":
            projector.acknowledge_publication(
                attempt_id, dict(evidence["snapshot"]), terminal=evidence["binding"]["terminal"]
            )
            entry.parameters = None
        elif state == "observed":
            entry.observed_binding = dict(evidence["binding"])
            entry.snapshot = None if evidence["snapshot"] is None else dict(evidence["snapshot"])
            self._restore_cadence(entry, projector, context)
            if version == 1:
                projector.observe(attempt_id)
            else:
                projector._observe(attempt_id)
            if entry.parameters is not None and entry.parameters["projection"] is None:
                entry.parameters = None
        self._save_cadence(entry, context)

    def advance(self, controller: Any, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Capture bounded local work and apply only exact consumed shared results."""
        current = {self._key(binding): binding for binding in bindings}
        for key in tuple(self._entries):
            if key not in current:
                self._entries.pop(key).close()
        protected = {
            request.project_id
            for request in controller.executor.unresolved_requests()
            if request.operation_kind == "progress_projection"
        }
        count = len(bindings)
        for offset in range(min(_MAX_PROJECTS_PER_SLICE, count)):
            binding = bindings[(self._cursor + offset) % count]
            key = self._key(binding)
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= _MAX_PROJECTORS:
                    victim = next(
                        (
                            candidate
                            for candidate, owner in self._entries.items()
                            if owner.can_evict
                            and owner.parameters is None
                            and owner.binding.project_id not in protected
                            and self._park(owner)
                        ),
                        None,
                    )
                    if victim is None:
                        continue
                    self._entries.pop(victim)
                entry = _ProjectProgress(binding)
                self._restore_scan(entry)
                self._entries[key] = entry
            else:
                entry.binding = binding
            if entry.parameters is None:
                try:
                    self._capture(entry)
                except (OSError, RuntimeError, ValueError, TypeError):
                    entry.offsets = {1: 0, 2: 0}
            self._entries.move_to_end(key)
        if count:
            self._cursor = (self._cursor + min(_MAX_PROJECTS_PER_SLICE, count)) % count
        parameters = {
            entry.binding.project_id: entry.parameters
            for entry in self._entries.values()
            if entry.parameters is not None and entry.retry_at <= self._clock()
        }
        results = controller.advance_progress_projection(bindings, registry_revision, parameters)
        by_project = {entry.binding.project_id: entry for entry in self._entries.values()}
        for project_id, evidence in results.items():
            entry = by_project.get(project_id)
            if entry is not None:
                try:
                    self._apply(entry, evidence)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    entry.parameters = None
                    entry.retry_at = self._clock() + 1.0
