"""Measure qexp dormant working-set cost at fixed production budgets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
import tracemalloc
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.working_set import SERVICE_LANES, BindingWorkingSet
from qqtools.plugins.qexp.runtime.project_activation import publish_project_activation
from qqtools.plugins.qexp.runtime.store import atomic_replace


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS is unavailable")


def _binding(root: Path, runtime: MachineRuntime, index: int) -> ProjectBinding:
    project_id = f"project-{index:04d}"
    shared_root = root / project_id / ".qexp"
    atomic_replace(shared_root / "project" / "identity.json", {"project": {"project_id": project_id}})
    (shared_root / "locks").mkdir(parents=True, exist_ok=True)
    return ProjectBinding(
        project_id,
        shared_root,
        "qualification-machine",
        registration_generation=uuid.uuid4().hex,
        runtime_instance_id=runtime.instance_id,
        runtime_root=str(root / project_id / "runtime"),
    )


def _retire(working_set: BindingWorkingSet, binding: ProjectBinding) -> None:
    for lane in SERVICE_LANES:
        turn = working_set.begin_turn(binding, lane)
        if not working_set.acknowledge(turn, quiescent=True):
            raise RuntimeError(f"binding did not retire: {binding.project_id}/{lane}")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def _profile(root: Path, count: int, cycles: int, soak_cycles: int) -> dict[str, Any]:
    runtime = MachineRuntime(root / "machine")
    runtime.ensure_layout()
    bindings = tuple(_binding(root, runtime, index) for index in range(count))
    with runtime.registry_guard() as acquired:
        if not acquired:
            raise RuntimeError("registry lock is unavailable")
        runtime.registration.save_registry_locked(1, list(bindings))

    working_set = BindingWorkingSet(runtime, process_fence=f"qualification-{count}")
    started = time.perf_counter()
    working_set.reconcile(bindings, revision=1)
    startup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for binding in bindings:
        _retire(working_set, binding)
    retirement_seconds = time.perf_counter() - started
    if working_set.resident_bindings():
        raise RuntimeError("setup retained resident bindings")

    renewals = 0

    def renew(
        _binding: ProjectBinding,
        *,
        renew: bool = False,
        renewal_horizon_seconds: float = 0.0,
    ) -> bool:
        nonlocal renewals
        if not renew:
            raise RuntimeError("dormant eligibility check did not request renewal")
        if renewal_horizon_seconds <= 0:
            raise RuntimeError("dormant eligibility check did not schedule the next renewal horizon")
        renewals += 1
        return True

    runtime.binding_write_eligible = renew
    signature_reads = 0
    original_read_signature = working_set._read_signature

    def read_signature(binding: ProjectBinding):
        nonlocal signature_reads
        signature_reads += 1
        return original_read_signature(binding)

    working_set._read_signature = read_signature
    tracemalloc.start()
    memory_before, _ = tracemalloc.get_traced_memory()
    rss_before = _rss_bytes()
    descriptors_before = len(tuple(Path("/proc/self/fd").iterdir()))
    latencies = []
    for _ in range(cycles):
        started = time.perf_counter()
        working_set.reconcile(bindings, revision=1)
        working_set.poll_dormant(limit=4)
        working_set.renew_dormant_registrations(limit=64)
        latencies.append(time.perf_counter() - started)
    memory_after, memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    descriptors_after = len(tuple(Path("/proc/self/fd").iterdir()))
    rss_after = _rss_bytes()

    expected_reads = cycles * min(4, count)
    expected_renewals = cycles * min(64, count)
    if signature_reads != expected_reads or renewals != expected_renewals:
        raise RuntimeError("steady-state work exceeded or missed its fixed budget")
    steady_p95 = _p95(latencies)
    if steady_p95 > 0.100:
        raise RuntimeError(f"steady p95 exceeded the declared 100 ms threshold: {steady_p95:.6f}s")

    target = bindings[-1]
    publish_project_activation(SimpleNamespace(shared_root=target.shared_root), "qualification_wake")
    wake_started = time.perf_counter()
    wake_rounds = 0
    while target not in working_set.resident_bindings():
        wake_rounds += 1
        working_set.poll_dormant(limit=4)
        if wake_rounds > (count + 3) // 4:
            raise RuntimeError("wake discovery exceeded ceil(N / 4) rounds")
    wake_seconds = time.perf_counter() - wake_started

    soak_started = time.perf_counter()
    selected_soak_cycles = soak_cycles if count == 1000 else 0
    for _ in range(selected_soak_cycles):
        working_set.reconcile(bindings, revision=1)
        working_set.poll_dormant(limit=4)
        working_set.renew_dormant_registrations(limit=64)
    soak_seconds = time.perf_counter() - soak_started
    return {
        "registered_projects": count,
        "startup_seconds": startup_seconds,
        "retirement_seconds": retirement_seconds,
        "steady_cycles": cycles,
        "steady_mean_ms": statistics.fmean(latencies) * 1000,
        "steady_p95_ms": steady_p95 * 1000,
        "project_truth_reads_before_wake": expected_reads,
        "lease_renewals_before_wake": expected_renewals,
        "wake_rounds": wake_rounds,
        "wake_seconds": wake_seconds,
        "resident_after_wake": len(working_set.resident_bindings()),
        "compact_binding_states": len(working_set._states),
        "descriptor_delta": descriptors_after - descriptors_before,
        "traced_memory_delta_bytes": memory_after - memory_before,
        "traced_memory_peak_bytes": memory_peak,
        "rss_delta_bytes": rss_after - rss_before,
        "soak_cycles": selected_soak_cycles,
        "soak_seconds": soak_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--soak-cycles", type=int, default=10_000)
    parser.add_argument("--projects", type=int, nargs="+", default=[1, 100, 1000])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.cycles < 1 or args.soak_cycles < 0 or any(count < 1 for count in args.projects):
        raise SystemExit("projects and cycles must be positive; soak cycles cannot be negative")
    args.output.mkdir(parents=True)
    cases = []
    try:
        for count in args.projects:
            case_root = args.output / f"projects-{count}"
            case_root.mkdir()
            cases.append(_profile(case_root, count, args.cycles, args.soak_cycles))
    finally:
        for child in args.output.glob("projects-*"):
            shutil.rmtree(child, ignore_errors=True)
    result = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "thresholds": {
            "steady_p95_ms": 100,
            "dormant_checkpoint_reads_per_cycle": 4,
            "dormant_lease_renewals_per_heartbeat": 64,
            "wake_rounds": "ceil(registered_projects / 4)",
        },
        "cases": cases,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
