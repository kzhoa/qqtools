"""Derived step optimization must preserve base artifacts and every occurrence."""

import numpy as np
import pytest

import qqtools.data.qbalance as balance


def _metrics(costs, batches, world_size):
    loads = np.sort(costs[batches].sum(1))
    if not len(loads):
        return np.zeros(3)
    return np.array([loads[-1], np.percentile(loads, 99), loads[world_size - 1::world_size].sum()])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("total", [0, 1, 48, 53])
@pytest.mark.parametrize("should_use_best", [False, True])
def test_step_refinement_never_worsens_metrics_or_mutates_input(
    seed, world_size, total, should_use_best,
):
    costs = np.random.default_rng(seed).integers(0, 100, total).astype(float)
    base = balance._partition_rank_batches(costs, batch_size=4, strategy="lpt_best", seed=seed)
    batches = balance._complete_rank_batch_tail(costs, base, world_size, seed, False, "lpt_best")
    before = batches.copy()
    batches.setflags(write=False)
    actual = balance._optimize_step_batches(
        costs, batches, world_size, seed, should_use_best=should_use_best,
    )
    assert np.all(_metrics(costs, actual, world_size) <= _metrics(costs, before, world_size))
    np.testing.assert_array_equal(np.sort(actual.ravel()), np.sort(before.ravel()))
    np.testing.assert_array_equal(batches, before)
    again = balance._optimize_step_batches(
        costs, batches, world_size, seed, should_use_best=should_use_best,
    )
    np.testing.assert_array_equal(actual, again)


def test_restores_step_score_without_changing_peak_or_p99():
    costs = np.array([4, 65, 24, 67, 34, 3, 55, 18, 90, 58, 44, 56.])
    base = balance._partition_rank_batches(costs, batch_size=2, strategy="lpt_best", seed=7)
    before = base.full_batches.copy()
    actual = balance._optimize_step_batches(costs, base.full_batches, 2, 7)
    np.testing.assert_array_equal(_metrics(costs, before, 2)[:2], _metrics(costs, actual, 2)[:2])
    assert _metrics(costs, actual, 2)[2] < _metrics(costs, before, 2)[2]
    np.testing.assert_array_equal(base.full_batches, before)
    plan = balance._plan_rank_batches(
        costs, batch_size=2, world_size=2, strategy="lpt_best", seed=7, should_shuffle=False,
    )
    np.testing.assert_array_equal(_metrics(costs, plan.reshape(-1, 2), 2), _metrics(costs, actual, 2))


@pytest.mark.parametrize("batches,world_size", [(np.empty((0, 2), dtype=int), 2),
                         (np.arange(8).reshape(4, 2), 1), (np.arange(8).reshape(8, 1), 2)])
def test_skips_unnecessary_search(monkeypatch, batches, world_size):
    def unexpected(*args):
        raise AssertionError("No meaningful step refinement is possible")

    monkeypatch.setattr(balance, "_refine_rank_batches", unexpected)
    monkeypatch.setattr(balance, "_repair_rank_batches", unexpected)
    assert balance._optimize_step_batches(np.arange(8.), batches, world_size, 7) is batches


def test_rejects_regressing_candidate_from_either_search(monkeypatch):
    costs = np.array([9., 7, 2, 1])
    batches = np.array([[0, 3], [1, 2]])
    batches.setflags(write=False)
    bad = np.array([[0, 1], [2, 3]])
    monkeypatch.setattr(balance, "_refine_rank_batches", lambda *args, **kwargs: bad.copy())
    monkeypatch.setattr(balance, "_repair_rank_batches", lambda *args: bad.copy())
    actual = balance._optimize_step_batches(costs, batches, 2, 7)
    np.testing.assert_array_equal(actual, batches)


def test_middle_runs_one_pass_and_no_three_batch_repair(monkeypatch):
    calls = []
    original = balance._refine_rank_batches

    def refine(*args, **kwargs):
        calls.append(kwargs["passes"])
        return original(*args, **kwargs)

    def unexpected(*args):
        raise AssertionError("Middle must not run three-batch repair")

    monkeypatch.setattr(balance, "_refine_rank_batches", refine)
    monkeypatch.setattr(balance, "_repair_rank_batches", unexpected)
    costs = np.random.default_rng(7).integers(1, 100, 48).astype(float)
    balance._plan_rank_batches(costs, batch_size=4, world_size=2, strategy="lpt")
    assert calls == [1]


@pytest.mark.parametrize("seed", range(8))
def test_best_retains_middle_final_quality_on_identical_occurrences(seed):
    costs = np.random.default_rng(seed).integers(1, 100, 48).astype(float)
    plans = [balance._plan_rank_batches(
        costs, batch_size=4, world_size=2, strategy=strategy, seed=7, should_shuffle=False,
    ) for strategy in ("lpt", "lpt_best")]
    scores = [balance._rank_batch_quality(costs[plan].sum(2).ravel(), 2) for plan in plans]
    assert scores[1] <= scores[0]
