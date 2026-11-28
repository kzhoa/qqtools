"""Retained demand and completion state for bounded ready-index repairs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, KeysView, Mapping
from typing import TYPE_CHECKING, Any

from ..identity import ValidationIdentity
from .rotation import BindingRotation

if TYPE_CHECKING:
    from ...bindings import ProjectBinding
    from ..executor import ProjectIOExecutor
    from ..protocol import ProjectIORequest


class ReadyIndexBuildPolicy:
    kind = "scheduler_ready_index_build"
    service = "ready_index"

    def __init__(self, limit: int) -> None:
        self._rotation = BindingRotation(limit)
        self.limit = limit
        self._needed: dict[ValidationIdentity, None] = {}
        self._settled: dict[ValidationIdentity, None] = {}

    @property
    def needed(self) -> KeysView[ValidationIdentity]:
        return self._needed.keys()

    @property
    def settled(self) -> KeysView[ValidationIdentity]:
        return self._settled.keys()

    def clear_need(self, identity: ValidationIdentity) -> None:
        self._needed.pop(identity, None)

    def mark_needed(self, identity: ValidationIdentity, has_backoff: Callable[[ValidationIdentity], bool]) -> None:
        self._settled.pop(identity, None)
        if identity in self._needed:
            return
        if len(self._needed) >= self.limit:
            victim = next((item for item in self._needed if has_backoff(item)), next(iter(self._needed)))
            del self._needed[victim]
        self._needed[identity] = None

    def refresh_bindings(
        self, eligible: dict[ValidationIdentity, ProjectBinding], observed_only: bool
    ) -> dict[ValidationIdentity, ProjectBinding]:
        self._needed = {item: None for item in self._needed if item in eligible}
        self._settled = {item: None for item in self._settled if item in eligible}
        return {item: binding for item, binding in eligible.items() if not observed_only or item in self._needed}

    def candidates(self, current: Collection[ValidationIdentity], *, can_start: bool) -> Iterator[ValidationIdentity]:
        for identity in self._rotation.candidates(current, can_start=can_start):
            if identity not in self._settled:
                yield identity

    @staticmethod
    def prepare(executor: ProjectIOExecutor, binding: ProjectBinding, revision: int) -> ProjectIORequest:
        return executor.prepare_scheduler_ready_index_build(binding, revision)

    def on_completion(self, identity: ValidationIdentity, evidence: Mapping[str, Any]) -> None:
        if evidence.get("state") in {"active", "degraded"}:
            self._needed.pop(identity, None)
            self._settled.pop(identity, None)
            self._settled[identity] = None
            if len(self._settled) > self.limit:
                del self._settled[next(iter(self._settled))]
