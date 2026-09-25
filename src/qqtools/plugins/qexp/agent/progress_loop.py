"""Best-effort progress I/O, separate from heartbeat and authority threads."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..runtime.progress import ProgressProjector, has_local_progress_mailbox, resolve_progress_binding
from ..runtime.progress_v2 import ProgressV2Projector, has_local_progress_v2_mailbox
from . import helpers


class ProgressObservationLoop:
    """A slow application mailbox must never delay the machine control plane."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qexp-machine-progress", daemon=True)
        self._projectors: dict[tuple[str, str | None, str], ProgressProjector] = {}
        self._v2_projectors: dict[tuple[str, str | None, str], ProgressV2Projector] = {}
        self._cursor = 0
        self._registry_revision: int | None = None

    def start(self) -> None:
        try:
            self._thread.start()
        except Exception:
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
        revision, bindings = self._runtime.load_registry_snapshot()
        self._runtime.working_set.reconcile(bindings, revision=revision)
        if revision != self._registry_revision:
            current = {
                (binding.project_id, binding.registration_generation, str(binding.shared_root)) for binding in bindings
            }
            for key in set(self._projectors) - current:
                self._projectors.pop(key).close()
            for key in set(self._v2_projectors) - current:
                self._v2_projectors.pop(key).close()
            self._registry_revision = revision
        if not bindings:
            return
        for offset in range(min(16, len(bindings))):
            if self._stop.is_set():
                return
            binding = bindings[(self._cursor + offset) % len(bindings)]
            key = binding.project_id, binding.registration_generation, str(binding.shared_root)
            try:
                # The cheap gate is entirely machine-local. Projects that never
                # publish progress cause no additional shared registration polls.
                runtime_root = self._runtime.project_paths(binding.project_id)["root"]
                has_v1_mailbox = has_local_progress_mailbox(runtime_root)
                has_v2_contexts = (Path(runtime_root) / "progress-v2-contexts").is_dir()
                has_v2_mailbox = has_local_progress_v2_mailbox(runtime_root) if has_v2_contexts else False
                if not has_v1_mailbox and not has_v2_mailbox:
                    projector = self._projectors.pop(key, None)
                    if projector is not None:
                        projector.close()
                    projector_v2 = self._v2_projectors.pop(key, None)
                    if projector_v2 is not None:
                        projector_v2.close()
                    continue
                self._runtime.working_set.activate(binding, "local_progress")
                if not (binding.enabled or self._runtime.binding_state(binding) == "draining"):
                    continue
                if not self._runtime.binding_write_eligible(binding):
                    continue
                cfg = helpers._binding_config(self._runtime, binding)
                if has_v1_mailbox:
                    projector = self._projectors.get(key)
                    if projector is None:
                        projector = ProgressProjector(
                            cfg,
                            resolver=self._resolver(binding),
                            registration_generation=binding.registration_generation,
                        )
                        self._projectors[key] = projector
                    projector.tick()
                if has_v2_mailbox:
                    projector_v2 = self._v2_projectors.get(key)
                    if projector_v2 is None:
                        projector_v2 = ProgressV2Projector(
                            cfg,
                            resolver=self._resolver(binding),
                            registration_generation=binding.registration_generation,
                        )
                        self._v2_projectors[key] = projector_v2
                    projector_v2.tick()
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
            for projector in self._v2_projectors.values():
                projector.close()
