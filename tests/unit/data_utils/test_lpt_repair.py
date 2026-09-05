from fractions import Fraction
from itertools import combinations
from math import factorial

import numpy as np
import pytest

import qqtools.data.qbalance as balance


@pytest.mark.parametrize("group_count", [2, 3])
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_small_assignments_cover_labelled_fixed_capacity_partitions(group_count, slots):
    assignments = balance._small_batch_assignments(group_count, slots)
    total = group_count * slots
    expected = factorial(total) // factorial(slots) ** group_count
    assert assignments.shape == (expected, group_count, slots)
    assert not assignments.flags.writeable
    assert len(np.unique(assignments.reshape(expected, total), axis=0)) == expected
    np.testing.assert_array_equal(np.sort(assignments.reshape(expected, total), axis=1),
                                  np.broadcast_to(np.arange(total), (expected, total)))
    assert assignments is balance._small_batch_assignments(group_count, slots)


def test_repair_escapes_certified_two_batch_local_optimum():
    costs = np.array([56, 55, 34, 85, 41, 76, 21, 34, 1, 60, 2, 20], dtype=float)
    batches = np.array([[3, 2, 8], [5, 7, 11], [9, 0, 10], [4, 6, 1]])
    original = batches.copy()
    repaired = balance._repair_rank_batches(costs, batches, 2, 7, 0)
    loads = costs[repaired].sum(1)
    assert (loads.max(), np.percentile(loads, 99), np.sort(loads)[1::2].sum()) == (
        126, 125.91, 244,
    )
    np.testing.assert_array_equal(batches, original)
    np.testing.assert_array_equal(np.sort(repaired.ravel()), np.arange(12))
    plan = balance._plan_rank_batches(costs, batch_size=3, world_size=2, seed=7,
                                     strategy="lpt_best", should_shuffle=False)
    assert costs[plan].sum(2).max() == 126


@pytest.mark.parametrize("batch_size", [2, 3, 4, 16, 128])
def test_candidates_only_reassign_three_slots_per_batch(batch_size):
    costs = np.random.default_rng(42).uniform(1, 100, 3 * batch_size)
    rows = np.arange(len(costs)).reshape(3, batch_size)
    rows.setflags(write=False)
    costs.setflags(write=False)
    for window in (0, 1, 15):
        candidates = balance._rank_batch_repair_candidates(costs, rows, window)
        assert 0 < len(candidates) <= balance._LPT_REPAIR_CANDIDATES
        for candidate in candidates:
            assert candidate.shape == rows.shape
            assert np.all((candidate != rows).sum(1) <= 3)
            np.testing.assert_array_equal(np.sort(candidate.ravel()), np.arange(len(costs)))


@pytest.mark.parametrize("seed", range(5))
def test_candidate_shortlist_matches_independent_local_enumeration(seed):
    costs = np.random.default_rng(seed).integers(0, 100, 9).astype(float)
    rows = np.arange(9).reshape(3, 3)
    signatures = set()
    for first in combinations(range(9), 3):
        rest = tuple(i for i in range(9) if i not in first)
        for second in combinations(rest, 3):
            third = tuple(i for i in rest if i not in second)
            loads = [costs[list(g)].sum() for g in (first, second, third)]
            signatures.add(tuple(sorted(loads, reverse=True)))
    expected = sorted(signatures)[:balance._LPT_REPAIR_CANDIDATES]
    candidates = balance._rank_batch_repair_candidates(costs, rows, 0)
    actual = [tuple(sorted(costs[c].sum(1), reverse=True)) for c in candidates]
    assert actual == expected


@pytest.mark.parametrize("should_certify_peak", [False, True])
def test_repair_has_fixed_window_and_global_score_budgets(monkeypatch, should_certify_peak):
    costs = np.arange(1, 241, dtype=float)
    batches = np.arange(len(costs)).reshape(-1, 4)
    peak = costs[batches].sum(1).max()
    windows = []
    scores = []
    original_quality = balance._rank_batch_quality

    def candidates(costs, rows, window):
        windows.append(window)
        return [rows.copy(), rows.copy()]

    def quality(loads, world_size):
        scores.append(1)
        return original_quality(loads, world_size)

    monkeypatch.setattr(balance, "_rank_batch_repair_candidates", candidates)
    monkeypatch.setattr(balance, "_rank_batch_quality", quality)
    repaired = balance._repair_rank_batches(
        costs, batches, 2, 7, peak if should_certify_peak else 0,
    )
    budget = (balance._LPT_SECONDARY_REPAIR_WINDOWS if should_certify_peak
              else balance._LPT_REPAIR_WINDOWS)
    assert windows == list(range(budget))
    assert len(scores) == 1 + budget * balance._LPT_REPAIR_CANDIDATES
    np.testing.assert_array_equal(repaired, batches)


def test_repair_rejects_bad_global_candidate(monkeypatch):
    costs = np.arange(1, 13, dtype=float)
    batches = np.array([[0, 5, 8, 11], [1, 4, 7, 10], [2, 3, 6, 9]])
    monkeypatch.setattr(balance, "_rank_batch_repair_candidates",
                        lambda costs, rows, window: [np.sort(rows.ravel()).reshape(3, 4)])
    repaired = balance._repair_rank_batches(costs, batches, 1, 7, 0)
    np.testing.assert_array_equal(repaired, batches)


@pytest.mark.parametrize("rows", [np.arange(6).reshape(6, 1), np.arange(6).reshape(1, 6),
                                  np.empty((0, 3), dtype=int)])
def test_repair_skips_fixed_or_empty_plans(monkeypatch, rows):
    def unexpected(*args):
        raise AssertionError("No repair should be planned")

    monkeypatch.setattr(balance, "_rank_batch_repair_candidates", unexpected)
    assert balance._repair_rank_batches(np.arange(6, dtype=float), rows, 1, 7, 0) is rows


@pytest.mark.parametrize("cost_values", [
    [1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 100], [1e308, 1e308, 1, 1, 1, 1],
    [1e-320, 5e-324, 1e-323, 0, 0, 0], [0.1, 0.2, 0.3, 1e100, 1e-100, 0],
])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 6])
def test_lower_bound_is_conservative_for_integer_float_and_extreme_inputs(cost_values, batch_size):
    costs = np.array(cost_values, dtype=float)
    order = np.argsort(-costs, kind="stable")
    exact = sorted(Fraction(float(c)) for c in costs)
    bound = max(sum(exact) / (len(costs) // batch_size), exact[-1] + sum(exact[:batch_size - 1]))
    actual = balance._rank_batch_peak_lower_bound(costs, order, batch_size)
    assert np.isfinite(actual)
    assert Fraction(actual) <= bound
    assert actual >= costs.max()


def test_lower_bound_uses_selected_occurrences_including_padding():
    costs = np.array([10000, 6, 2, 1], dtype=float)
    # The outlier is dropped; repeated occurrences contribute to the mean and companions.
    selected = np.array([1, 2, 2, 3, 3, 3])
    bound = balance._rank_batch_peak_lower_bound(costs, selected, 3)
    assert 7.99 < bound <= 8


@pytest.mark.parametrize("batch_size,world_size", [(2, 1), (4, 4), (16, 2), (64, 1)])
@pytest.mark.parametrize("seed", [0, 7])
def test_repair_preserves_incumbent_occurrences_and_determinism(batch_size, world_size, seed):
    costs = np.random.default_rng(42).lognormal(3, 1.5, batch_size * 8 - 1)
    # Include a repeated occurrence, as happens with sampler padding.
    batches = np.resize(np.arange(len(costs)), batch_size * 8).reshape(-1, batch_size)
    costs.setflags(write=False)
    batches.setflags(write=False)
    original = balance._rank_batch_quality(costs[batches].sum(1), world_size)
    repaired = balance._repair_rank_batches(costs, batches, world_size, seed, 0)
    assert balance._rank_batch_quality(costs[repaired].sum(1), world_size) <= original
    np.testing.assert_array_equal(np.sort(repaired.ravel()), np.sort(batches.ravel()))
    np.testing.assert_array_equal(
        repaired, balance._repair_rank_batches(costs, batches, world_size, seed, 0),
    )


def test_repair_never_materializes_overflowing_candidates():
    costs = np.array([1.7e308, 4e307, 3e307, 2e307, 1, 1, 1, 1])
    batches = np.array([[0, 4, 5, 6], [1, 2, 3, 7]])
    with np.errstate(over="raise", invalid="raise"):
        repaired = balance._repair_rank_batches(costs, batches, 1, 7, 1.7e308)
    assert np.isfinite(costs[repaired].sum(1)).all()
    np.testing.assert_array_equal(np.sort(repaired.ravel()), np.arange(8))


def test_skips_full_lower_bound_when_largest_sample_already_certifies_peak(monkeypatch):
    def unexpected(*args):
        raise AssertionError("The largest sample alone already certifies the peak")

    monkeypatch.setattr(balance, "_rank_batch_peak_lower_bound", unexpected)
    costs = np.r_[1000.0, np.zeros(63)]
    plan = balance._plan_rank_batches(costs, batch_size=4, world_size=2, strategy="lpt_best")
    assert costs[plan].sum(2).max() == 1000
    np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(64))
