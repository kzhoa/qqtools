import numpy as np
import pytest

import qqtools.data.qbalance as balance


STRATEGIES = ("lpt_fast", "lpt", "lpt_best")


def _quality(costs, plan):
    loads = costs[plan].sum(axis=2)
    if not loads.size:
        return 0.0, 0.0, 0.0
    ordered = np.sort(loads.ravel())
    peak = float(ordered[-1])
    return peak, float(np.percentile(ordered, 99)), float(np.square(ordered / peak).sum())


@pytest.mark.parametrize("total", [0, 1, 16, 65, 256])
@pytest.mark.parametrize("batch_size,world_size", [(1, 1), (4, 1), (4, 4), (16, 4)])
@pytest.mark.parametrize("should_drop", [True, False])
@pytest.mark.parametrize("seed", [0, 7])
def test_tiers_preserve_tail_counts_and_base_quality_order(
    total, batch_size, world_size, should_drop, seed
):
    costs = np.random.default_rng(42).lognormal(3, 1.5, total)
    costs.setflags(write=False)
    group_size = batch_size * world_size
    steps = total // group_size if should_drop else (total + group_size - 1) // group_size
    target = steps * group_size
    scores = []
    for strategy in STRATEGIES:
        kwargs = dict(batch_size=batch_size, world_size=world_size, seed=seed,
                      strategy=strategy, should_drop_last=should_drop, should_shuffle=False)
        plan = balance._plan_rank_batches(costs, **kwargs)
        assert plan.shape == (steps, world_size, batch_size)
        assert plan.dtype == np.int64 and plan.flags.c_contiguous
        assert np.all((plan >= 0) & (plan < total))
        if should_drop:
            assert len(np.unique(plan)) == target
        else:
            np.testing.assert_array_equal(np.unique(plan), np.arange(total))
        np.testing.assert_array_equal(plan, balance._plan_rank_batches(costs, **kwargs))
        partition = balance._partition_rank_batches(
            costs, batch_size=batch_size, seed=seed, strategy=strategy,
        )
        scores.append(_quality(costs, partition.full_batches[:, None, :]))
    assert scores[2] <= scores[1] <= scores[0]


@pytest.mark.parametrize("distribution", ["uniform", "lognormal", "bimodal", "outlier"])
def test_tier_quality_on_representative_distributions(distribution):
    rng = np.random.default_rng(42)
    if distribution == "uniform":
        costs = rng.integers(1, 101, 4096)
    elif distribution == "lognormal":
        costs = np.rint(rng.lognormal(3, 1.5, 4096))
    elif distribution == "bimodal":
        costs = np.where(rng.random(4096) < 0.1, 1000, 1)
    else:
        costs = np.ones(4096)
        costs[0] = 100_000
    scores = [_quality(costs, balance._plan_rank_batches(
        costs, batch_size=16, world_size=4, strategy=strategy, should_shuffle=False
    )) for strategy in STRATEGIES]
    assert scores[2][:2] <= scores[1][:2] <= scores[0][:2]


def test_best_can_strictly_improve_peak():
    costs = np.random.default_rng(0).integers(1, 100, 48)
    medium = balance._plan_rank_batches(costs, batch_size=4, world_size=2,
                                       strategy="lpt", seed=7, should_shuffle=False)
    best = balance._plan_rank_batches(costs, batch_size=4, world_size=2,
                                     strategy="lpt_best", seed=7, should_shuffle=False)
    assert _quality(costs, best)[0] == 185 < _quality(costs, medium)[0] == 187


def test_medium_retains_fast_candidate_if_capacity_is_worse(monkeypatch):
    costs = np.array([9, 7, 2, 1], dtype=float)
    monkeypatch.setattr(balance, "_partition_capacity_batches",
                        lambda *args: np.array([[0, 1], [2, 3]]))
    plan = balance._plan_rank_batches(costs, batch_size=2, strategy="lpt", should_shuffle=False)
    assert _quality(costs, plan)[0] == 10


@pytest.mark.parametrize("batch_size", [1, 16])
def test_skips_more_work_when_loads_cannot_improve(monkeypatch, batch_size):
    def unexpected_work(*args):
        raise AssertionError("A fixed or perfectly equal load vector needs no further search")

    monkeypatch.setattr(balance, "_partition_capacity_batches", unexpected_work)
    monkeypatch.setattr(balance, "_refine_rank_batches", unexpected_work)
    for strategy in STRATEGIES:
        costs = np.arange(16) if batch_size == 1 else np.ones(64)
        balance._plan_rank_batches(costs, batch_size=batch_size, strategy=strategy)


def test_best_rejects_regressing_refinement(monkeypatch):
    batches = np.array([[0, 3], [1, 2]])
    costs = np.array([8, 7, 2, 1], dtype=float)
    # Start with unequal but well-paired loads, so refinement is attempted.
    costs[0] = 9

    def worsen_pair(costs, trial, loads, first, second):
        trial[:] = [[0, 1], [2, 3]]

    monkeypatch.setattr(balance, "_swap_rank_batch_pair", worsen_pair)
    actual = balance._refine_rank_batches(costs, batches, 1, 7)
    np.testing.assert_array_equal(actual, batches)


@pytest.mark.parametrize("seed", range(20))
def test_binary_swap_search_matches_exhaustive_pair_search(seed):
    costs = np.random.default_rng(seed).integers(0, 100, 16).astype(float)
    batches = np.arange(16).reshape(2, 8)
    loads = costs[batches].sum(1)
    gap = abs(loads[0] - loads[1])
    expected = min([gap] + [abs(loads[0] - loads[1] - 2 * (costs[x] - costs[y]))
                           for x in batches[0] for y in batches[1]])
    balance._swap_rank_batch_pair(costs, batches, loads, 0, 1)
    actual = costs[batches].sum(1)
    assert abs(actual[0] - actual[1]) == expected
    assert actual.max() <= loads.max()
    np.testing.assert_array_equal(np.sort(batches.ravel()), np.arange(16))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tiers_handle_zero_costs_and_reject_overflow(strategy):
    plan = balance._plan_rank_batches(np.zeros(64), batch_size=4, strategy=strategy)
    np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(64))
    with pytest.raises(ValueError, match="remain finite"):
        balance._plan_rank_batches([1e308, 1e308], batch_size=2, strategy=strategy)


def test_finite_quality_when_only_epoch_sum_overflows():
    quality = balance._rank_batch_quality(np.full(16, 1e308), 4)
    assert quality == (1e308, 1e308, 4.0)


def test_overflowing_fast_candidate_does_not_block_a_finite_higher_tier():
    costs = np.array([1.7e308, 4e307, 3e307, 2e307, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="remain finite"):
        balance._plan_rank_batches(costs, batch_size=4, strategy="lpt_fast")
    for strategy in ("lpt", "lpt_best"):
        plan = balance._plan_rank_batches(costs, batch_size=4, strategy=strategy)
        assert np.isfinite(costs[plan].sum(2)).all()


def test_rejects_unknown_rank_batch_strategy():
    with pytest.raises(ValueError, match="Unsupported rank-batch strategy"):
        balance._plan_rank_batches([], batch_size=1, strategy="lpt2")
