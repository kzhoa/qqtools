import copy
import json
import multiprocessing
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from qqtools import qLmdbDataset
from qqtools.data.qbalance import compute_global_even_sort_order
from qqtools.torch import qLmdbDataset as TorchQLmdbDataset
from qqtools.torch import qLmdbDatasetBase
from qqtools.torch.ddp import BalancedBatchSampler

lmdb = pytest.importorskip("lmdb")


def _write_lmdb(
    path: Path,
    samples: list[object],
    *,
    include_length: bool = True,
    serializer=pickle.dumps,
    is_subdir: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(str(path), subdir=is_subdir, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            for idx, sample in enumerate(samples):
                transaction.put(str(idx).encode("ascii"), serializer(sample))
            if include_length:
                transaction.put(b"length", pickle.dumps(len(samples)))
        environment.sync()
    finally:
        environment.close()


class _PlainDataset(qLmdbDataset):
    def __init__(self, root: Path, files: list[str], **kwargs) -> None:
        self._files = files
        super().__init__(root=root, **kwargs)

    @property
    def lmdb_files(self):
        return self._files


class _BalancedDataset(_PlainDataset):
    def get_sample_cost(self, idx: int):
        return self.get(idx)["cost"]


class _RawCostDataset(_PlainDataset):
    @staticmethod
    def parse_value(blob: bytes):
        raise AssertionError("balance metadata must not parse training samples")

    def get_sample_cost(self, idx: int):
        return pickle.loads(self.get_raw_blob(idx))["cost"]


class _FailingRawCostDataset(_RawCostDataset):
    def __init__(self, root: Path, files: list[str], **kwargs) -> None:
        self.close_calls = 0
        super().__init__(root, files, **kwargs)

    def get_sample_cost(self, idx: int):
        self.get_raw_blob(idx)
        raise RuntimeError("cost extraction failed")

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class _JsonDataset(_PlainDataset):
    @staticmethod
    def parse_value(blob: bytes):
        return json.loads(blob.decode("utf-8"))


class _PrefetchedCostDataset(_PlainDataset):
    def __init__(self, root: Path, files: list[str], **kwargs) -> None:
        self.cost_calls: list[int] = []
        self.storage_open_calls = 0
        super().__init__(root, files, **kwargs)

    def get_sample_cost(self, idx: int):
        self.cost_calls.append(idx)
        return self.get(idx)["cost"]

    def _ensure_storage_open(self) -> None:
        self.storage_open_calls += 1
        super()._ensure_storage_open()


def _samples(costs: list[float]) -> list[dict]:
    return [
        {
            "id": idx,
            "cost": cost,
            "value": torch.tensor([idx], dtype=torch.int64),
        }
        for idx, cost in enumerate(costs)
    ]


def _collate_ids(batch: list[dict]) -> list[int]:
    return [int(sample["id"]) for sample in batch]


def test_public_exports_resolve_to_new_dataset_class():
    assert qLmdbDataset is TorchQLmdbDataset
    assert issubclass(TorchQLmdbDataset, qLmdbDatasetBase)


def test_get_raw_blob_returns_exact_bytes_without_parsing(tmp_path: Path):
    root = tmp_path / "dataset"
    blobs = [b'{"id": 3}', b'{"id": 7}']
    _write_lmdb(
        root / "raw" / "data.lmdb",
        blobs,
        serializer=lambda blob: blob,
    )
    dataset = _JsonDataset(root, ["raw/data.lmdb"])

    assert dataset.get_raw_blob(0) == blobs[0]
    assert dataset.get_raw_blob(-1) == blobs[-1]
    assert dataset[1] == {"id": 7}


def test_get_raw_blob_uses_global_multishard_indices_and_reports_errors(tmp_path: Path):
    root = tmp_path / "dataset"
    first_samples = [{"id": 0}, {"id": 1}]
    second_samples = [{"id": 2}, {"id": 3}]
    _write_lmdb(root / "raw" / "0.lmdb", first_samples)
    second_path = root / "raw" / "1.lmdb"
    _write_lmdb(second_path, second_samples)
    dataset = _PlainDataset(root, ["raw/0.lmdb", "raw/1.lmdb"])

    assert pickle.loads(dataset.get_raw_blob(1)) == first_samples[1]
    assert pickle.loads(dataset.get_raw_blob(2)) == second_samples[0]
    assert pickle.loads(dataset.get_raw_blob(-1)) == second_samples[1]

    subset = dataset[[3, 0]]
    assert pickle.loads(subset.get_raw_blob(0)) == first_samples[0]
    with pytest.raises(IndexError, match="out of range"):
        dataset.get_raw_blob(4)
    with pytest.raises(IndexError, match="out of range"):
        dataset.get_raw_blob(-5)

    dataset.close()
    environment = lmdb.open(str(second_path), subdir=False, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            transaction.delete(b"1")
    finally:
        environment.close()

    with pytest.raises(IndexError, match=r"Missing LMDB sample key.*local index 1.*1\.lmdb"):
        dataset.get_raw_blob(3)


def test_plain_multishard_reading_is_sorted_and_runtime_storage_is_lazy(tmp_path: Path):
    root = tmp_path / "dataset"
    later_samples = [
        {"id": 2, "cost": 8.0, "value": torch.tensor([2])},
        {"id": 3, "cost": 2.0, "value": torch.tensor([3])},
    ]
    _write_lmdb(root / "raw" / "b.lmdb", later_samples)
    _write_lmdb(root / "raw" / "a.lmdb", _samples([9.0, 1.0]))

    dataset = _PlainDataset(root, ["raw/b.lmdb", "raw/a.lmdb"])

    assert dataset._environments is None
    assert len(dataset) == 4
    assert dataset._environments is None
    assert dataset[0]["id"] == 0
    assert dataset[-1]["id"] == 3
    assert dataset._environments is not None

    subset = dataset[[3, 0]]
    assert subset._environments is None
    assert [subset[idx]["id"] for idx in range(len(subset))] == [3, 0]

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._environments is None
    assert restored[2]["id"] == 2
    dataset.close()
    restored.close()


def test_multishard_global_index_mapping_and_length_fallback(tmp_path: Path):
    root = tmp_path / "dataset"
    first = [{"id": 0}, {"id": 1}]
    second = [{"id": 2}, {"id": 3}, {"id": 4}]
    _write_lmdb(root / "raw" / "0.lmdb", first, include_length=False)
    _write_lmdb(root / "raw" / "1.lmdb", second)

    dataset = _PlainDataset(root, ["raw/0.lmdb", "raw/1.lmdb"])

    assert len(dataset) == 5
    assert [dataset[idx]["id"] for idx in range(5)] == [0, 1, 2, 3, 4]
    with pytest.raises(IndexError, match="out of range"):
        dataset[5]


def test_non_integer_length_metadata_is_rejected(tmp_path: Path):
    root = tmp_path / "dataset"
    path = root / "raw" / "data.lmdb"
    _write_lmdb(path, [{"id": 0}])
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(b"length", pickle.dumps(1.5))
    finally:
        environment.close()

    dataset = _PlainDataset(root, ["raw/data.lmdb"])
    with pytest.raises(RuntimeError, match="must be an integer"):
        len(dataset)


def test_directory_lmdb_and_custom_parser_are_supported(tmp_path: Path):
    root = tmp_path / "dataset"
    samples = [{"id": 3}, {"id": 7}]
    _write_lmdb(
        root / "raw" / "json.lmdb",
        samples,
        serializer=lambda sample: json.dumps(sample).encode("utf-8"),
        is_subdir=True,
    )

    dataset = _JsonDataset(root, ["raw/json.lmdb"])

    assert len(dataset) == 2
    assert dataset[1] == {"id": 7}


def test_plain_mode_creates_no_balance_artifacts(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0, 2.0]))

    dataset = _PlainDataset(root, ["raw/data.lmdb"])

    assert dataset.processed_file_names == []
    assert not (dataset.processed_dir / "balance_meta.npz").exists()
    assert not (dataset.processed_dir / "balance_order.npy").exists()
    assert not dataset.rewrite_path.exists()
    with pytest.raises(RuntimeError, match="disabled"):
        _ = dataset.sample_costs


def test_missing_inputs_and_balance_hook_fail_early(tmp_path: Path):
    root = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError, match="Missing LMDB input"):
        _PlainDataset(root, ["raw/missing.lmdb"])

    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    with pytest.raises(RuntimeError, match="get_sample_cost"):
        _PlainDataset(root, ["raw/data.lmdb"], balance_mode="runtime")


def test_balance_assets_are_deterministic_and_used_by_loader(tmp_path: Path):
    root = tmp_path / "dataset"
    costs = np.asarray([9.0, 1.0, 8.0, 2.0, 7.0], dtype=np.float64)
    _write_lmdb(root / "raw" / "data.lmdb", _samples(costs.tolist()))

    dataset = _BalancedDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="runtime",
        balance_seed=7,
        balance_strategy="v3",
    )

    expected_order = compute_global_even_sort_order(costs, seed=7, strategy="v3")
    assert np.array_equal(dataset.sample_costs, costs)
    assert np.array_equal(dataset.sample_order, expected_order)
    assert not dataset.sample_costs.flags.writeable
    assert not dataset.sample_order.flags.writeable
    assert (dataset.processed_dir / "balance_meta.npz").exists()
    assert (dataset.processed_dir / "balance_order.npy").exists()

    loader = dataset.to_dataloader(batch_size=2, shuffle=True)
    assert isinstance(loader.batch_sampler, BalancedBatchSampler)
    observed_ids = sorted(int(value) for batch in loader for value in batch["id"])
    assert set(observed_ids) == set(range(costs.shape[0]))
    assert len(observed_ids) == 6


def test_balance_can_compute_costs_from_raw_blobs_without_parsing(tmp_path: Path):
    root = tmp_path / "dataset"
    costs = np.asarray([4.0, 1.0, 7.0], dtype=np.float64)
    _write_lmdb(root / "raw" / "data.lmdb", _samples(costs.tolist()))

    dataset = _RawCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="runtime",
    )

    assert np.array_equal(dataset.sample_costs, costs)
    assert dataset._transactions is None
    assert dataset._environments is None


def test_balance_closes_raw_storage_when_cost_extraction_fails(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _FailingRawCostDataset.__new__(_FailingRawCostDataset)

    with pytest.raises(RuntimeError, match="cost extraction failed"):
        dataset.__init__(
            root,
            ["raw/data.lmdb"],
            balance_mode="runtime",
        )

    assert dataset.close_calls >= 1
    assert dataset._transactions is None
    assert dataset._environments is None


def test_balanced_subset_uses_local_cost_and_order_coordinates(tmp_path: Path):
    root = tmp_path / "dataset"
    costs = np.asarray([9.0, 1.0, 8.0, 2.0, 7.0], dtype=np.float64)
    _write_lmdb(root / "raw" / "data.lmdb", _samples(costs.tolist()))
    dataset = _BalancedDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="runtime",
        balance_seed=7,
    )

    subset_indices = np.asarray([4, 1, 3], dtype=np.int64)
    subset = dataset[subset_indices]
    full_order = dataset.sample_order
    global_positions = np.empty(full_order.shape[0], dtype=np.int64)
    global_positions[full_order] = np.arange(full_order.shape[0], dtype=np.int64)
    expected_local_order = np.argsort(global_positions[subset_indices], kind="stable")

    assert np.array_equal(subset.sample_costs, costs[subset_indices])
    assert np.array_equal(subset.sample_order, expected_local_order)
    loader = subset.to_dataloader(batch_size=2, shuffle=True)
    observed_ids = {int(value) for batch in loader for value in batch["id"]}
    assert observed_ids == {1, 3, 4}


@pytest.mark.parametrize("invalid_cost", [True, -1.0, float("nan"), "large"])
def test_balance_rejects_invalid_sample_costs(tmp_path: Path, invalid_cost):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", [{"id": 0, "cost": invalid_cost}])

    with pytest.raises((TypeError, ValueError), match="Sample cost"):
        _BalancedDataset(root, ["raw/data.lmdb"], balance_mode="runtime")


def test_rewrite_preserves_source_order_asset_and_aligns_effective_costs(tmp_path: Path):
    root = tmp_path / "dataset"
    costs = np.asarray([9.0, 1.0, 8.0, 2.0, 7.0], dtype=np.float64)
    samples = _samples(costs.tolist())
    source_path = root / "raw" / "data.lmdb"
    _write_lmdb(source_path, samples)
    source_environment = lmdb.open(
        str(source_path),
        subdir=False,
        readonly=True,
        lock=False,
    )
    try:
        with source_environment.begin(write=False) as transaction:
            source_blobs = [
                transaction.get(str(idx).encode("ascii"))
                for idx in range(len(samples))
            ]
    finally:
        source_environment.close()

    dataset = _BalancedDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        balance_seed=11,
        balance_strategy="v1",
    )
    stored_order = np.load(dataset.processed_dir / "balance_order.npy")

    assert dataset.rewrite_path.exists()
    assert dataset._lmdb_paths == (dataset.rewrite_path,)
    assert [dataset[idx]["id"] for idx in range(len(dataset))] == stored_order.tolist()
    assert np.array_equal(dataset.sample_costs, costs[stored_order])
    assert np.array_equal(dataset.sample_order, np.arange(len(dataset)))
    assert np.array_equal(
        np.load(dataset.processed_dir / "balance_order.npy"),
        stored_order,
    )

    environment = lmdb.open(str(dataset.rewrite_path), subdir=False, readonly=True, lock=False)
    try:
        with environment.begin(write=False) as transaction:
            assert pickle.loads(transaction.get(b"length")) == len(samples)
            assert pickle.loads(transaction.get(b"0"))["id"] == int(stored_order[0])
            assert transaction.get(b"0") == source_blobs[int(stored_order[0])]
    finally:
        environment.close()

    reloaded = _BalancedDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        balance_seed=11,
        balance_strategy="v1",
    )
    assert np.array_equal(reloaded.sample_costs, costs[stored_order])
    assert np.array_equal(reloaded.sample_order, np.arange(len(reloaded)))


def test_sequential_staged_rewrite_matches_direct_across_shards(tmp_path: Path):
    direct_root = tmp_path / "direct"
    staged_root = tmp_path / "staged"
    costs = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0]
    for root in (direct_root, staged_root):
        _write_lmdb(
            root / "raw" / "b.lmdb",
            [{"id": idx, "cost": cost} for idx, cost in enumerate(costs[3:])],
        )
        _write_lmdb(
            root / "raw" / "a.lmdb",
            [{"id": idx, "cost": cost} for idx, cost in enumerate(costs[:3])],
        )

    direct = _BalancedDataset(
        direct_root,
        ["raw/b.lmdb", "raw/a.lmdb"],
        balance_mode="rewrite",
        balance_seed=17,
    )
    staged = _BalancedDataset(
        staged_root,
        ["raw/b.lmdb", "raw/a.lmdb"],
        balance_mode="rewrite",
        rewrite_staging_dir=staged_root / "scratch",
        balance_seed=17,
    )

    assert [direct.get_raw_blob(i) for i in range(len(direct))] == [
        staged.get_raw_blob(i) for i in range(len(staged))
    ]
    assert pickle.loads(staged.get_raw_blob(-1))
    assert staged.rewrite_path.exists()
    assert list((staged_root / "scratch").glob("**/staging-*.lmdb")) == []


def test_sequential_staged_reuses_cursor_blob_through_single_cost_hook(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([4.0, 1.0, 7.0]))
    dataset = _PrefetchedCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        rewrite_staging_dir=root / "scratch",
    )

    assert dataset.cost_calls == [0, 1, 2]
    assert dataset.storage_open_calls == 0
    assert dataset._transactions is None
    assert dataset._prefetched_raw_blob is None
    source_costs = np.asarray([4.0, 1.0, 7.0])
    expected_order = compute_global_even_sort_order(source_costs, strategy="v3")
    assert np.array_equal(dataset.sample_costs, source_costs[expected_order])


def test_prefetched_blob_scope_does_not_leak_through_copy_or_pickle(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", [{"id": 0}])
    dataset = _PlainDataset(root, ["raw/data.lmdb"])
    prefetched = pickle.dumps({"id": "prefetched"})

    with dataset._use_prefetched_raw_blob(0, prefetched):
        assert dataset.get_raw_blob(0) == prefetched
        copied = copy.copy(dataset)
        restored = pickle.loads(pickle.dumps(dataset))

    assert dataset._prefetched_raw_blob is None
    assert copied._prefetched_raw_blob is None
    assert restored._prefetched_raw_blob is None
    assert pickle.loads(copied.get_raw_blob(0)) == {"id": 0}
    assert pickle.loads(restored.get_raw_blob(0)) == {"id": 0}


def test_sequential_staged_rejects_noncanonical_keys_but_direct_keeps_compatibility(
    tmp_path: Path,
):
    direct_root = tmp_path / "direct"
    staged_root = tmp_path / "staged"
    for root in (direct_root, staged_root):
        path = root / "raw" / "data.lmdb"
        _write_lmdb(path, [{"id": 0, "cost": 1.0}, {"id": 1, "cost": 2.0}])
        environment = lmdb.open(str(path), subdir=False, map_size=1 << 26)
        try:
            with environment.begin(write=True) as transaction:
                transaction.put(b"01", pickle.dumps({"id": 1, "cost": 2.0}))
        finally:
            environment.close()

    direct = _PrefetchedCostDataset(
        direct_root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
    )
    assert len(direct) == 2
    assert direct.cost_calls == [0, 1]
    with pytest.raises(RuntimeError, match="canonical ASCII sample keys"):
        _PrefetchedCostDataset(
            staged_root,
            ["raw/data.lmdb"],
            balance_mode="rewrite",
            rewrite_staging_dir=staged_root / "scratch",
        )
    assert not (staged_root / "processed" / "balance_rewrite.lmdb").exists()


def test_rewrite_artifact_is_reused_when_staging_path_is_added(tmp_path: Path):
    root = tmp_path / "dataset"
    path = root / "raw" / "data.lmdb"
    _write_lmdb(path, [{"id": 0, "cost": 1.0}, {"id": 1, "cost": 2.0}])
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(b"01", pickle.dumps({"id": 1, "cost": 2.0}))
    finally:
        environment.close()

    direct = _PrefetchedCostDataset(root, ["raw/data.lmdb"], balance_mode="rewrite")
    direct_blobs = [direct.get_raw_blob(idx) for idx in range(len(direct))]
    direct.close()

    reused = _PrefetchedCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        rewrite_staging_dir=root / "scratch",
    )

    assert [reused.get_raw_blob(idx) for idx in range(len(reused))] == direct_blobs
    assert not (root / "scratch").exists()


def test_balance_only_keeps_direct_key_compatibility_without_rewrite(tmp_path: Path):
    root = tmp_path / "dataset"
    path = root / "raw" / "data.lmdb"
    _write_lmdb(path, [{"id": 0, "cost": 1.0}, {"id": 1, "cost": 2.0}])
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(b"01", pickle.dumps({"id": 1, "cost": 2.0}))
    finally:
        environment.close()

    dataset = _PrefetchedCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="runtime",
    )

    assert dataset.cost_calls == [0, 1]
    assert dataset._transactions is None
    assert (root / "processed" / "balance_meta.npz").exists()
    assert (root / "processed" / "balance_order.npy").exists()
    assert not dataset.rewrite_path.exists()


def test_sequential_staged_scan_failure_cleans_unreturned_staging_lmdb(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    original_scan = _BalancedDataset._iter_source_blobs_sequential
    scan_calls = 0

    def fail_source_scan(self):
        nonlocal scan_calls
        scan_calls += 1
        if scan_calls == 2:
            raise RuntimeError("injected source scan failure")
        yield from original_scan(self)

    monkeypatch.setattr(
        _BalancedDataset,
        "_iter_source_blobs_sequential",
        fail_source_scan,
    )
    with pytest.raises(RuntimeError, match="Failed to materialize staged"):
        _BalancedDataset(
            root,
            ["raw/data.lmdb"],
            balance_mode="rewrite",
            rewrite_staging_dir=root / "scratch",
        )

    workspace = next((root / "scratch").iterdir())
    assert not list(workspace.glob("staging-*.lmdb"))
    assert not list(workspace.glob("staging-*.lmdb-lock"))
    assert not list(workspace.glob("inverse-*.mmap"))


def test_sequential_staged_clears_prefetched_blob_when_cost_extraction_fails(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = object.__new__(_FailingRawCostDataset)

    with pytest.raises(RuntimeError, match="cost extraction failed"):
        dataset.__init__(
            root,
            ["raw/data.lmdb"],
            balance_mode="rewrite",
            rewrite_staging_dir=root / "scratch",
        )

    assert dataset._prefetched_raw_blob is None


def test_empty_staged_dataset_skips_large_space_preflight(tmp_path: Path, monkeypatch):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", [])
    monkeypatch.setattr(
        "qqtools.torch.qlmdbdataset.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 0})(),
    )

    dataset = _PrefetchedCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        rewrite_staging_dir=root / "scratch",
    )
    assert len(dataset) == 0
    assert dataset.rewrite_path.exists()


def test_sequential_staged_empty_dataset_publishes_length_zero_without_staging_scan(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", [])
    dataset = _PrefetchedCostDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="rewrite",
        rewrite_staging_dir=root / "scratch",
    )

    assert len(dataset) == 0
    assert dataset.cost_calls == []
    environment = lmdb.open(str(dataset.rewrite_path), subdir=False, readonly=True, lock=False)
    try:
        with environment.begin(write=False) as transaction:
            assert pickle.loads(transaction.get(b"length")) == 0
    finally:
        environment.close()
    workspace = next((root / "scratch").iterdir())
    assert not list(workspace.glob("inverse-*.mmap"))
    assert not list(workspace.glob("staging-*.lmdb"))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"balance_mode": "unknown"}, "balance_mode"),
        ({"rewrite_staging_dir": "scratch"}, "balance_mode='rewrite'"),
        (
            {
                "balance_mode": "runtime",
                "rewrite_staging_dir": "scratch",
            },
            "balance_mode='rewrite'",
        ),
    ],
)
def test_balance_mode_configuration_is_validated(tmp_path: Path, kwargs, message):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    with pytest.raises(ValueError, match=message):
        _BalancedDataset(root, ["raw/data.lmdb"], **kwargs)


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"enable_balance": True},
        {"enable_rewrite": True},
        {"rewrite_io_strategy": "sequential_staged"},
    ],
)
def test_legacy_balance_configuration_arguments_are_removed(tmp_path: Path, legacy_kwargs):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _BalancedDataset(root, ["raw/data.lmdb"], **legacy_kwargs)


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_file_lock_write_guard_serializes_processes(
    tmp_path: Path,
    start_method: str,
    checkout_subprocess_env,
):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")

    probe_path = (
        Path(__file__).parents[3] / "fixtures" / "qlmdbdataset_file_lock_probe.py"
    )
    worker_env = checkout_subprocess_env
    if os.name != "nt":
        worker_env.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "TMP": "/tmp"})

    subprocess.run(
        [sys.executable, str(probe_path), start_method, str(tmp_path)],
        check=True,
        env=worker_env,
        timeout=60,
    )


@pytest.mark.filterwarnings("ignore:This process.*fork")
def test_plain_dataloader_reads_with_multiple_workers(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0, 2.0, 3.0, 4.0]))
    dataset = _PlainDataset(root, ["raw/data.lmdb"])

    loader = dataset.to_dataloader(
        batch_size=2,
        shuffle=False,
        num_workers=2,
        multiprocessing_context="fork",
        collate_fn=_collate_ids,
    )
    observed_ids = [value for batch in loader for value in batch]

    assert observed_ids == [0, 1, 2, 3]
    assert dataset._environments is None
    assert loader.multiprocessing_context.get_start_method() == "fork"


def test_dataloader_prefers_forkserver_when_available(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _PlainDataset(root, ["raw/data.lmdb"])

    loader = dataset.to_dataloader(batch_size=1, num_workers=1)

    assert loader.multiprocessing_context.get_start_method() == "forkserver"


def test_dataloader_context_falls_back_to_spawn(tmp_path: Path, monkeypatch):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _PlainDataset(root, ["raw/data.lmdb"])
    monkeypatch.setattr(
        "qqtools.torch.qlmdbdataset.multiprocessing.get_all_start_methods",
        lambda: ["spawn"],
    )

    loader = dataset.to_dataloader(batch_size=1, num_workers=1)

    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_dataloader_respects_explicit_context(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _PlainDataset(root, ["raw/data.lmdb"])

    loader = dataset.to_dataloader(
        batch_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
    )

    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_balanced_dataloader_rejects_unpickleable_collate_before_worker_start(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _BalancedDataset(
        root,
        ["raw/data.lmdb"],
        balance_mode="runtime",
    )

    with pytest.raises(TypeError, match="module-level function"):
        dataset.to_dataloader(
            batch_size=1,
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=lambda batch: batch,
        )


def test_dataloader_validates_worker_configuration(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_lmdb(root / "raw" / "data.lmdb", _samples([1.0]))
    dataset = _PlainDataset(root, ["raw/data.lmdb"])

    with pytest.raises(ValueError, match="batch_size"):
        dataset.to_dataloader(batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        dataset.to_dataloader(batch_size=1, num_workers=-1)
    with pytest.raises(ValueError, match="persistent_workers"):
        dataset.to_dataloader(batch_size=1, persistent_workers=True)
    with pytest.raises(ValueError, match="multiprocessing_context"):
        dataset.to_dataloader(
            batch_size=1,
            multiprocessing_context="forkserver",
        )
