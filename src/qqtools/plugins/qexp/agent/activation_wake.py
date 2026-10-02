"""Machine-local activation generation fencing asynchronous idle proofs."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from ..runtime.store import atomic_replace, read_json_limited

if TYPE_CHECKING:
    from .context import MachineRuntime, ProjectBinding


class ActivationWakeFence:
    """Invalidate service proofs after an activation serialized with idle exit."""

    def __init__(self, runtime: MachineRuntime) -> None:
        self.runtime = runtime
        self._has_captured = False
        self._generation: str | None = None
        self._registry_revision: int | None = None
        self._bindings: tuple[ProjectBinding, ...] = ()

    def _read(self) -> str | None:
        try:
            value = read_json_limited(
                self.runtime.paths["agent"] / "activation-wake.json",
                max_bytes=1024,
                record_type="machine_activation_wake",
            )
        except FileNotFoundError:
            return None
        record = value.get("machine_activation_wake") if isinstance(value, dict) else None
        if (
            set(value) != {"machine_activation_wake"}
            or not isinstance(record, dict)
            or set(record) != {"version", "runtime_id", "generation"}
            or type(record["version"]) is not int
            or record["version"] != 1
            or record["runtime_id"] != self.runtime.instance_id
            or not isinstance(record["generation"], str)
            or re.fullmatch(r"[0-9a-f]{32}", record["generation"]) is None
        ):
            raise ValueError("machine activation wake record is malformed or belongs to another runtime.")
        return record["generation"]

    def publish_locked(self) -> None:
        """Publish one wake while the caller owns the agent lifecycle guard."""
        atomic_replace(
            self.runtime.paths["agent"] / "activation-wake.json",
            {
                "machine_activation_wake": {
                    "version": 1,
                    "runtime_id": self.runtime.instance_id,
                    "generation": uuid.uuid4().hex,
                }
            },
        )

    def capture(self, registry_revision: int, bindings: list[ProjectBinding]) -> None:
        """Capture before service work and invalidate all older binding turns."""
        was_captured = self._has_captured
        self._has_captured = False
        generation = self._read()
        if generation != self._generation or (not was_captured and self._registry_revision is not None):
            for binding in bindings:
                self.runtime.working_set.activate(binding, "local_activation")
        self._generation = generation
        self._registry_revision = registry_revision
        self._bindings = tuple(bindings)
        self._has_captured = True

    def is_current(self) -> bool:
        """Verify local capture identity; unreadable evidence never means idle."""
        if not self._has_captured:
            return False
        try:
            revision, bindings = self.runtime.load_registry_snapshot()
            return (
                revision == self._registry_revision
                and tuple(bindings) == self._bindings
                and self._read() == self._generation
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False
