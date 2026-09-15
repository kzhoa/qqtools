"""Best-effort progress I/O, separate from heartbeat and authority threads."""

from __future__ import annotations

import threading
from typing import Any

from ..runtime.progress import ProgressProjector, resolve_progress_binding
from . import helpers


class ProgressObservationLoop:
    """A slow application mailbox must never delay the machine control plane."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qexp-machine-progress", daemon=True)
        self._projectors: dict[tuple[str, str | None], ProgressProjector] = {}
        self._cursor = 0

    def start(self) -> None:
        try:
            self._thread.start()
        except Exception:
            # Optional observation must not prevent the actual agent starting.
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _resolver(self, binding: Any):
        def resolve(cfg: Any, context: dict[str, Any]):
            if self._stop.is_set() or not self._runtime.binding_write_eligible(binding):
                raise ValueError("progress registration is not current")
            return resolve_progress_binding(cfg, context)
        return resolve

    def cycle(self) -> None:
        _, bindings = self._runtime.load_registry()
        current = {(binding.project_id, binding.registration_generation) for binding in bindings}
        for key in set(self._projectors) - current:
            self._projectors.pop(key).close()
        if not bindings:
            return
        # Bound work and rotate projects rather than favoring the first project.
        for offset in range(min(16, len(bindings))):
            if self._stop.is_set():
                return
            binding = bindings[(self._cursor + offset) % len(bindings)]
            try:
                if not (binding.enabled or self._runtime.binding_state(binding) == "draining"):
                    continue
                if not self._runtime.binding_write_eligible(binding):
                    continue
                key = binding.project_id, binding.registration_generation
                projector = self._projectors.get(key)
                if projector is None:
                    cfg = helpers._binding_config(self._runtime, binding)
                    projector = ProgressProjector(cfg, resolver=self._resolver(binding))
                    self._projectors[key] = projector
                projector.tick()
            except Exception:
                continue
        self._cursor = (self._cursor + min(16, len(bindings))) % len(bindings)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self.cycle()
                except Exception:
                    pass
                self._stop.wait(1.0)
        finally:
            for projector in self._projectors.values():
                projector.close()
