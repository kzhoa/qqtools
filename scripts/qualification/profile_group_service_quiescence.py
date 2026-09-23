"""Produce isolated Group service scale, churn, and real-process soak evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from qqtools.plugins.qexp.runtime.group_discovery import activation, locator, maintenance, service
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from qqtools.version import __version__
from tests.helpers.qexp.group_service_qualification import (
    measure_locator_traversal,
    populate_settled_group_history,
    process_resources,
)
from tests.helpers.qexp_discovery import isolated_group


def _activate(cfg) -> None:
    for _ in range(2_000_000):
        if activation.advance_group_service_activation(cfg)["state"] == "active":
            return
    raise RuntimeError("Group service activation did not converge")


def _lifecycle(cfg, cycle: int) -> None:
    lane, reason = (
        ("control", "task_change"),
        ("control", "group_operation"),
        ("membership", "submission_finalize"),
        ("maintenance", "metadata_cleanup"),
    )[cycle % 4]
    with group_writer_lock(cfg, "experiment"):
        record = locator.publish_group_locator_locked(cfg, "experiment", lane, reason)
        if not locator.acknowledge_group_locator_locked(
            cfg,
            "experiment",
            lane,
            record["generation"],
            retirement_ready=lambda: True,
        ):
            raise RuntimeError("Group locator lifecycle did not acknowledge its exact generation")


def run(output: Path, histories: list[int], cycles: int, soak_seconds: int) -> dict:
    """Run qualification in a new directory and return its retained evidence."""
    output.mkdir()
    original_version = activation.__version__
    activation.__version__ = activation.WRITER_FLOOR
    started = time.time()
    try:
        history_rows = []
        for history in histories:
            cfg = isolated_group(output / f"history-{history}", tail=0)
            _activate(cfg)
            populate_settled_group_history(cfg, history)
            with group_writer_lock(cfg, "experiment"):
                locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")
            measurement = measure_locator_traversal(cfg.shared_root, "control", "experiment")
            history_rows.append({"retained_groups": history, **measurement})
            print(f"completed retained_groups={history}", flush=True)

        churn_cfg = isolated_group(output / "churn", tail=0)
        _activate(churn_cfg)
        churn_started = time.monotonic()
        churn_samples = []
        for cycle in range(cycles):
            _lifecycle(churn_cfg, cycle)
            if cycle in {0, cycles // 2, cycles - 1}:
                elapsed = time.monotonic() - churn_started
                churn_samples.append(
                    {
                        "cycle": cycle + 1,
                        "elapsed_seconds": elapsed,
                        "completion_rate_per_second": (cycle + 1) / max(elapsed, 1e-9),
                        **process_resources(),
                    }
                )
        print(f"completed accelerated_lifecycles={cycles}", flush=True)

        soak_cfg = isolated_group(output / "soak", tail=0)
        _activate(soak_cfg)
        soak_started = time.monotonic()
        soak_samples = [{"elapsed_seconds": 0.0, **process_resources()}]
        soak_cycles = 0
        next_sample = soak_started + 60.0
        while time.monotonic() - soak_started < soak_seconds:
            _lifecycle(soak_cfg, soak_cycles)
            soak_cycles += 1
            now = time.monotonic()
            if now >= next_sample:
                soak_samples.append({"elapsed_seconds": now - soak_started, **process_resources()})
                next_sample = now + 60.0
            time.sleep(min(1.0, max(0.0, soak_started + soak_seconds - time.monotonic())))
        soak_samples.append({"elapsed_seconds": time.monotonic() - soak_started, **process_resources()})

        sources = [
            Path(__file__),
            Path(locator.__file__),
            Path(activation.__file__),
            Path(service.__file__),
            Path(maintenance.__file__),
        ]
        result = {
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "source_version": __version__,
            "qualification_writer_floor": activation.WRITER_FLOOR,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "root": str(output.resolve()),
            "started_at_unix": started,
            "duration_seconds": time.time() - started,
            "filesystem_cache": "uncontrolled warm cache after isolated fixture creation",
            "durability": "real locator file and directory fsync; no physical power-loss emulation",
            "histories": history_rows,
            "accelerated_lifecycles": cycles,
            "churn_samples": churn_samples,
            "resource_envelope": {
                "maximum_group_service_descriptors": 256,
                "maximum_group_service_threads": 1,
                "maximum_resident_entries": 64,
                "maximum_rss_sample_spread_kib": 32 * 1024,
            },
            "soak_seconds": soak_seconds,
            "soak_lifecycles": soak_cycles,
            "soak_samples": soak_samples,
        }
        all_samples = [*churn_samples, *soak_samples]
        if all_samples:
            descriptor_spread = max(item["descriptors"] for item in all_samples) - min(
                item["descriptors"] for item in all_samples
            )
            thread_spread = max(item["threads"] for item in all_samples) - min(item["threads"] for item in all_samples)
            rss_spread = max(item["rss_kib"] for item in all_samples) - min(item["rss_kib"] for item in all_samples)
            result["observed_resource_spread"] = {
                "descriptors": descriptor_spread,
                "threads": thread_spread,
                "rss_kib": rss_spread,
            }
            result["resource_envelope_passed"] = (
                descriptor_spread <= 2 and thread_spread == 0 and rss_spread <= 32 * 1024
            )
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result.get("resource_envelope_passed") is False:
            raise RuntimeError("Group service qualification exceeded its process resource envelope")
        return result
    finally:
        activation.__version__ = original_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--histories", type=int, nargs="+", default=[0, 1_000, 10_000, 100_000])
    parser.add_argument("--cycles", type=int, default=100_000)
    parser.add_argument("--soak-seconds", type=int, default=86_400)
    args = parser.parse_args()
    if args.cycles < 1 or args.cycles > 100_000:
        parser.error("cycles must be 1..100000")
    if args.soak_seconds < 0 or args.soak_seconds > 86_400:
        parser.error("soak-seconds must be 0..86400")
    if not args.histories or any(value not in {0, 1_000, 10_000, 100_000} for value in args.histories):
        parser.error("histories must contain values from 0, 1000, 10000, 100000")
    result = run(args.output, args.histories, args.cycles, args.soak_seconds)
    print(
        json.dumps(
            {key: result[key] for key in ("duration_seconds", "accelerated_lifecycles", "soak_seconds")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
