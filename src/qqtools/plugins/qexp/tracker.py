from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable


def _parse_visible_gpu_ids_from_env() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None

    raw = raw.strip()
    if not raw:
        return []

    visible_gpu_ids: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        visible_gpu_ids.append(int(token))
    return visible_gpu_ids


def _discover_visible_gpu_ids_with_pynvml() -> list[int]:
    pynvml = importlib.import_module("pynvml")
    pynvml.nvmlInit()
    try:
        visible_gpu_ids = _parse_visible_gpu_ids_from_env()
        if visible_gpu_ids is not None:
            return visible_gpu_ids

        count = int(pynvml.nvmlDeviceGetCount())
        return list(range(count))
    finally:
        pynvml.nvmlShutdown()


def _discover_visible_gpu_ids_with_nvidia_smi() -> list[int]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is not available on PATH.")

    visible_gpu_ids = _parse_visible_gpu_ids_from_env()
    if visible_gpu_ids is not None:
        return visible_gpu_ids

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_ids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        gpu_ids.append(int(line))
    return gpu_ids


def probe_gpu_backend() -> tuple[str | None, list[int]]:
    try:
        return "pynvml", _discover_visible_gpu_ids_with_pynvml()
    except Exception:
        pass

    try:
        return "nvidia-smi", _discover_visible_gpu_ids_with_nvidia_smi()
    except Exception:
        return None, []


@dataclass(slots=True)
class Tracker:
    visible_gpu_ids: list[int] = field(default_factory=list)
    reserved_gpu_ids: set[int] = field(default_factory=set)
    task_id_to_gpu_ids: dict[str, list[int]] = field(default_factory=dict)
    backend_name: str | None = None
    gpu_probe: Callable[[], tuple[str | None, list[int]]] = probe_gpu_backend

    def refresh_visibility(self) -> None:
        backend_name, visible_gpu_ids = self.gpu_probe()
        self.backend_name = backend_name
        self.visible_gpu_ids = list(visible_gpu_ids)

    def get_allocatable_gpu_ids(self) -> list[int]:
        return [gpu_id for gpu_id in self.visible_gpu_ids if gpu_id not in self.reserved_gpu_ids]

    def allocate(self, task_id: str, num_gpus: int) -> list[int] | None:
        allocatable_gpu_ids = self.get_allocatable_gpu_ids()
        if len(allocatable_gpu_ids) < num_gpus:
            return None

        assigned_gpu_ids = allocatable_gpu_ids[:num_gpus]
        self.task_id_to_gpu_ids[task_id] = assigned_gpu_ids
        self.reserved_gpu_ids.update(assigned_gpu_ids)
        return assigned_gpu_ids

    def release(self, task_id: str) -> None:
        assigned_gpu_ids = self.task_id_to_gpu_ids.pop(task_id, [])
        for gpu_id in assigned_gpu_ids:
            self.reserved_gpu_ids.discard(gpu_id)
