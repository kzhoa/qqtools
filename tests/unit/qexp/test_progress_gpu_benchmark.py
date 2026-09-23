"""Qualification rules for the bounded GPU progress benchmark."""

from __future__ import annotations

import runpy
import time
from pathlib import Path

import pytest

_BENCHMARK = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/benchmarks/qexp_progress_gpu.py"))


def _group(regressions: list[float]) -> list[dict[str, object]]:
    group: list[dict[str, object]] = []
    for repetition, regression in enumerate(regressions):
        group.extend(
            [
                {"profile": "baseline_v1", "repetition": repetition, "steps_per_second": 100.0},
                {"profile": "candidate_v2", "repetition": repetition, "steps_per_second": 100.0 - regression},
            ]
        )
    return group


def test_three_percent_target_preserves_inconclusive_noise() -> None:
    compare = _BENCHMARK["_compare"]
    assert compare(_group([2.0, 2.1, 2.2]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "within_3_percent_target"
    )
    assert compare(_group([2.9, 3.0, 3.1]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "inconclusive_noise_crosses_3_percent_target"
    )
    assert compare(_group([2.0, 2.0, 4.9]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "inconclusive_noise_crosses_3_percent_target"
    )
    assert compare(_group([3.9, 4.0, 4.1]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "above_3_percent_target"
    )
    assert compare(_group([-2.0, 2.0, 6.0]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "inconclusive_noise_or_insufficient_repeats"
    )
    assert compare(_group([-0.2, -0.3, 8.6]), "baseline_v1", "candidate_v2", 3)["classification"] == (
        "inconclusive_noise_or_insufficient_repeats"
    )


def test_expired_case_deadline_prevents_child_launch(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="exceeded ten minutes"):
        _BENCHMARK["_run_profile"](
            profile="candidate_v2",
            measurement_kind="throughput",
            sync_every=16,
            import_root=tmp_path,
            expected_package=tmp_path,
            workers=1,
            clients=0,
            repetition=0,
            case_index=0,
            case_deadline=time.monotonic() - 1,
            args=None,
            work_root=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_latency_is_only_scheduled_for_one_representative_cell(tmp_path: Path) -> None:
    namespace = _BENCHMARK["_run_cli"].__globals__
    original = namespace["_run_profile"]
    calls: list[tuple[int, int, str, int]] = []

    def fake_run_profile(**kwargs):
        calls.append((kwargs["workers"], kwargs["clients"], kwargs["measurement_kind"], kwargs["repetition"]))
        return {
            "workers": kwargs["workers"],
            "clients_requested": kwargs["clients"],
            "measurement_kind": kwargs["measurement_kind"],
            "profile": kwargs["profile"],
            "repetition": kwargs["repetition"],
            "steps_per_second": 100.0,
        }

    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    (baseline / "qqtools").mkdir(parents=True)
    (candidate / "qqtools").mkdir(parents=True)
    args = _BENCHMARK["_cli_parser"]().parse_args(
        [
            "--baseline-package-path",
            str(baseline),
            "--candidate-source-path",
            str(candidate),
            "--gpu-index",
            "0",
            "--workers",
            "1",
            "4",
            "--clients",
            "0",
            "2",
            "--repeats",
            "3",
            "--latency-case",
            "4",
            "2",
        ]
    )
    try:
        namespace["_run_profile"] = fake_run_profile
        report = _BENCHMARK["_run_cli"](args)
    finally:
        namespace["_run_profile"] = original
    assert len(report["throughput_runs"]) == 2 * 2 * 3 * 3
    assert len(report["full_step_latency_runs"]) == 3
    assert {
        (workers, clients, kind, repetition)
        for workers, clients, kind, repetition in calls
        if kind == "full_step_latency"
    } == {(4, 2, "full_step_latency", 0)}
    args.latency_case = None
    calls.clear()
    try:
        namespace["_run_profile"] = fake_run_profile
        default_report = _BENCHMARK["_run_cli"](args)
    finally:
        namespace["_run_profile"] = original
    assert default_report["configuration"]["latency_case"] == [1, 0]
    assert len(default_report["full_step_latency_runs"]) == 3
