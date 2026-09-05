import importlib
import pickle
import warnings

import pytest

from qqtools import qLmdbDataset
from qqtools.torch.ddp import BalancedBatchSampler

lmdb = pytest.importorskip("lmdb")


class _CostDataset(qLmdbDataset):
    @property
    def lmdb_files(self):
        return ["raw/data.lmdb"]

    def get_sample_cost(self, idx):
        return self.get(idx)["cost"]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "raw" / "data.lmdb"
    path.parent.mkdir()
    with lmdb.open(str(path), subdir=False, map_size=1 << 20) as environment:
        with environment.begin(write=True) as transaction:
            for idx, cost in enumerate([9, 1, 8, 2, 7, 3]):
                transaction.put(str(idx).encode(), pickle.dumps({"id": idx, "cost": cost}))
            transaction.put(b"length", pickle.dumps(6))
    instance = _CostDataset(root=tmp_path, balance_mode="runtime", balance_strategy="v3")
    yield instance
    instance.close()


@pytest.mark.integration
@pytest.mark.parametrize("should_shuffle", [False, True])
@pytest.mark.parametrize("should_subset", [False, True])
def test_loader_delegates_strategy_and_order(dataset, monkeypatch, should_shuffle, should_subset):
    module = importlib.import_module("qqtools.torch.qlmdbdataset")
    calls = []

    def build_sampler(*args, **kwargs):
        assert "strategy" not in kwargs
        assert "sample_order" not in kwargs
        sampler = BalancedBatchSampler(*args, **kwargs)
        calls.append(sampler)
        return sampler

    monkeypatch.setattr(module, "BalancedBatchSampler", build_sampler)
    selected = dataset[[4, 1, 3, 0]] if should_subset else dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = selected.to_dataloader(batch_size=2, shuffle=should_shuffle)
        observed = sorted(int(value) for batch in loader for value in batch["id"])
    assert len(calls) == 1
    assert observed == ([0, 1, 3, 4] if should_subset else list(range(6)))
    assert not any(issubclass(item.category, FutureWarning) for item in caught)
    assert list(loader.batch_sampler) == list(BalancedBatchSampler(
        selected.sample_costs, batch_size=2, shuffle=should_shuffle, seed=selected.balance_seed,
    ))


@pytest.mark.integration
def test_loader_inherits_default_validation_tail_policy(dataset):
    with pytest.raises(ValueError, match="Non-shuffled LPT"):
        dataset.to_dataloader(batch_size=4, shuffle=False)
    loader = dataset.to_dataloader(batch_size=4, shuffle=False, drop_last=True)
    observed = [int(value) for batch in loader for value in batch["id"]]
    assert len(observed) == len(set(observed)) == 4
    loader = dataset.to_dataloader(batch_size=4, shuffle=True)
    observed = [int(value) for batch in loader for value in batch["id"]]
    assert len(observed) == 8
    assert set(observed) == set(range(6))
