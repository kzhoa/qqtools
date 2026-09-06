"""Shared value records for the ready projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReadyScope = Literal["home", "shared"]


@dataclass(frozen=True, slots=True)
class ReadyMarkerRef:
    task_id: str
    generation: int
    queue_scope: ReadyScope
    home_machine: str
    partition: str
    catalog_page: int
    marker_name: str

    @property
    def identity(self) -> str:
        return f"{self.task_id}.{self.generation}"
