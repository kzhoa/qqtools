"""Stage boundaries and exact tail accounting, independent of timing benchmarks."""

import numpy as np
import pytest

import qqtools.data.qbalance as balance


@pytest.mark.parametrize("strategy", balance._LPT_STRATEGIES)
@pytest.mark.parametrize("total", [0, 1, 3, 4, 5, 31, 128, 131])
def test_base_partition_covers_all_samples_without_padding(strategy, total):
    costs = np.random.default_rng(42).lognormal(size=total)
    before = costs.copy()
    partition = balance._partition_rank_batches(costs, batch_size=4, seed=7, strategy=strategy)
    batches, remainder = partition.full_batches, partition.remainder
    assert batches.shape == (total // 4, 4)
    assert remainder.shape == (total % 4,)
    assert batches.dtype == remainder.dtype == np.int64
    assert batches.flags.c_contiguous and remainder.flags.c_contiguous
    assert not batches.flags.writeable and not remainder.flags.writeable
    np.testing.assert_array_equal(np.sort(np.r_[batches.ravel(), remainder]), np.arange(total))
    # Selection occurs before sorting and is independent of algorithm tier.
    np.testing.assert_array_equal(remainder, np.random.default_rng(7).permutation(total)[total // 4 * 4:])
    np.testing.assert_array_equal(costs, before)


@pytest.mark.parametrize("strategy", balance._LPT_STRATEGIES)
@pytest.mark.parametrize("seed", [0, 7, 29])
def test_all_tiers_keep_identical_base_batches_across_world_sizes(strategy, seed, monkeypatch):
    costs = np.random.default_rng(13).integers(0, 100, 240)
    kwargs = dict(batch_size=4, seed=seed, strategy=strategy, should_shuffle=False)
    base = balance._partition_rank_batches(costs, batch_size=4, seed=seed, strategy=strategy)
    partitioner = balance._partition_rank_batches
    captured = []

    def capture(*args, **kwargs):
        partition = partitioner(*args, **kwargs)
        if kwargs["strategy"] == strategy:
            captured.append(partition.full_batches.copy())
        return partition

    monkeypatch.setattr(balance, "_partition_rank_batches", capture)
    for world_size in (1, 2, 3, 4, 5, 6):
        plan = balance._plan_rank_batches(costs, world_size=world_size, **kwargs)
        actual = captured[-1]
        np.testing.assert_array_equal(actual, base.full_batches)
        np.testing.assert_array_equal(np.sort(plan.ravel()), np.arange(len(costs)))


@pytest.mark.parametrize("strategy", balance._LPT_STRATEGIES)
@pytest.mark.parametrize("total", [0, 1, 7, 8, 127, 128, 129, 139])
@pytest.mark.parametrize("world_size", [1, 4, 8])
@pytest.mark.parametrize("should_drop", [False, True])
def test_tail_repair_is_bounded_reproducible_and_does_not_mutate_base(
    strategy, total, world_size, should_drop,
):
    costs = np.random.default_rng(42).integers(0, 100, total)
    base = balance._partition_rank_batches(costs, batch_size=4, seed=7, strategy=strategy)
    before, remainder = base.full_batches.copy(), base.remainder.copy()
    actual = balance._complete_rank_batch_tail(costs, base, world_size, 7, should_drop, strategy)
    count = len(before)
    global_size = 4 * world_size
    target = (total // global_size if should_drop else (total + global_size - 1) // global_size)
    assert actual.shape == (target * world_size, 4)
    np.testing.assert_array_equal(base.full_batches, before)
    np.testing.assert_array_equal(base.remainder, remainder)
    if not len(remainder) and count % world_size == 0:
        np.testing.assert_array_equal(actual, before)
        return
    # Independently reconstruct which occurrences the stage is allowed to change.
    rng = np.random.default_rng(7)
    pool_ids = rng.choice(count, min(count, count % world_size + world_size), replace=False)
    kept_ids = sorted(set(range(count)) - set(pool_ids))
    kept = before[kept_ids]
    assert count - len(kept) <= 2 * world_size - 1
    np.testing.assert_array_equal(actual[:len(kept)], kept)
    pool = np.r_[before[pool_ids].ravel(), remainder]
    rng.shuffle(pool)
    needed = actual.size - kept.size
    expected = pool[:needed] if should_drop else np.resize(pool, needed)
    np.testing.assert_array_equal(np.sort(actual[len(kept):].ravel()), np.sort(expected))
    again = balance._complete_rank_batch_tail(costs, base, world_size, 7, should_drop, strategy)
    np.testing.assert_array_equal(actual, again)


def test_no_tail_skips_repair_work(monkeypatch):
    costs = np.arange(32, dtype=float)
    base = balance._partition_rank_batches(costs, batch_size=4)

    def unexpected(*args):
        raise AssertionError("Divisible base plans must not be repartitioned")

    monkeypatch.setattr(balance, "_partition_selected_batches", unexpected)
    assert balance._complete_rank_batch_tail(costs, base, 4, 0, False, "lpt") is base.full_batches


def test_world_independent_score_handles_large_loads():
    assert balance._rank_batch_quality(np.full(16, 1e308)) == (1e308, 1e308, 16.0)


def test_dropping_entire_dataset_skips_grouping(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("An empty final plan needs no base partition")

    monkeypatch.setattr(balance, "_partition_rank_batches", unexpected)
    plan = balance._plan_rank_batches(
        [1e308, 1e308], batch_size=2, world_size=2, should_drop_last=True,
    )
    assert plan.shape == (0, 2, 2)
