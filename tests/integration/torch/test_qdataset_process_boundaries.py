import multiprocessing
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from qqtools import qLmdbDataset

lmdb = pytest.importorskip("lmdb")

pytestmark = pytest.mark.integration


class _PlainDataset(qLmdbDataset):
    def __init__(self, root: Path, files: list[str], **kwargs) -> None:
        self._files = files
        super().__init__(root=root, **kwargs)

    @property
    def lmdb_files(self):
        return self._files


def _write_lmdb(path: Path, samples: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 26)
    try:
        with environment.begin(write=True) as transaction:
            for idx, sample in enumerate(samples):
                transaction.put(str(idx).encode("ascii"), pickle.dumps(sample))
            transaction.put(b"length", pickle.dumps(len(samples)))
        environment.sync()
    finally:
        environment.close()


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


def _probe_environment(checkout_subprocess_env: dict[str, str]) -> dict[str, str]:
    environment = checkout_subprocess_env.copy()
    if os.name != "nt":
        test_tmp = environment.get("TMPDIR", "/tmp")
        environment.update({"TMPDIR": test_tmp, "TEMP": test_tmp, "TMP": test_tmp})
    return environment


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_qdictdataloader_graph_collate_with_pickle_based_worker(
    start_method: str,
    checkout_subprocess_env: dict[str, str],
) -> None:
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")

    probe_path = Path(__file__).parents[2] / "fixtures" / "qdataset_graph_worker_probe.py"
    subprocess.run(
        [sys.executable, str(probe_path), start_method],
        check=True,
        env=_probe_environment(checkout_subprocess_env),
        timeout=60,
    )


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_file_lock_write_guard_serializes_processes(
    tmp_path: Path,
    start_method: str,
    checkout_subprocess_env: dict[str, str],
) -> None:
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is not available on this platform")

    probe_path = Path(__file__).parents[2] / "fixtures" / "qlmdbdataset_file_lock_probe.py"
    subprocess.run(
        [sys.executable, str(probe_path), start_method, str(tmp_path)],
        check=True,
        env=_probe_environment(checkout_subprocess_env),
        timeout=60,
    )


@pytest.mark.filterwarnings("ignore:This process.*fork")
def test_plain_dataloader_reads_with_multiple_workers(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is not available on this platform")

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
