"""Scheduling policy for bounded deadline scans."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ..identity import ValidationIdentity
from .rotation import BindingRotation

if TYPE_CHECKING:
    from ...bindings import ProjectBinding
    from ..executor import ProjectIOExecutor
    from ..protocol import ProjectIORequest


class DueOfferPolicy:
    kind = "scheduler_due_offer"
    service = "deadline"

    def __init__(self, limit: int) -> None:
        self._rotation = BindingRotation(limit)

    def candidates(
        self, current: Collection[ValidationIdentity], *, can_start: bool, excluded_projects: Collection[str]
    ) -> Iterator[ValidationIdentity]:
        for identity in self._rotation.candidates(current, can_start=can_start):
            if identity.project_id not in excluded_projects:
                yield identity

    @staticmethod
    def prepare(executor: ProjectIOExecutor, binding: ProjectBinding, revision: int) -> ProjectIORequest:
        return executor.prepare_scheduler_due_offer(binding, revision)

    def on_completion(self, identity: ValidationIdentity, evidence: Mapping[str, Any]) -> float | None:
        """Give other work a turn after an empty deadline scan."""
        return 1.0 if evidence.get("outcome") == "noop" and evidence.get("reason") == "no_due_deadline" else None
