"""Immutable inputs shared by scheduling decisions.

This module contains no qexp persistence, CLI, or agent imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Resources available to one scheduling cycle."""
    cpu: int = 0
    gpus: int = 0
    memory_bytes: int = 0

@dataclass(frozen=True, slots=True)
class TaskDemand:
    """Normalized resource demand for a task."""
    cpu: int = 0
    gpus: int = 0
    memory_bytes: int = 0

@dataclass(frozen=True, slots=True)
class AdmissionInput:
    """Pure admission input; policy evaluation must not perform I/O."""
    demand: TaskDemand
    available: ResourceSnapshot
    labels: Mapping[str, str] = ()
