"""Opt-in checks for the benchmark's algorithm coverage and reported fields."""

import pytest

from compare_fast_swap import timing_comparison


@pytest.mark.parametrize("should_two_pointer", [False, True])
def test_timing_reports_all_tiers_and_latency_range(capsys, should_two_pointer):
    timing_comparison(12, "uniform", 3, 2, should_two_pointer)
    rows = [line.split(",") for line in capsys.readouterr().out.splitlines()
            if line.startswith("12,")]
    expected = {"layered", "lpt_fast", "fast_swap", "lpt", "lpt_best"}
    if should_two_pointer:
        expected.add("two_pointer")
    assert {row[4] for row in rows} == expected
    assert len(rows) == len(expected)
    for row in rows:
        assert len(row) == 11
        median_ms, min_ms, max_ms, peak, p99, step_sum = map(float, row[5:])
        assert 0 <= min_ms <= median_ms <= max_ms
        assert 0 <= p99 <= peak <= step_sum
