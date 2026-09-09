import copy
import importlib
import multiprocessing
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from qqtools.torch.qlmdbdataset import qLmdbDataset

lmdb = pytest.importorskip("lmdb")


class _Dataset(qLmdbDataset):
    @property
    def lmdb_files(self):
        return ["data.lmdb"]


def _read_in_child(dataset, queue):
    try:
        storage = importlib.import_module("qqtools.torch.qlmdbdataset")
        assert storage._environment_pid == os.getpid()
        assert storage._environments == {}
        queue.put(dataset.get(0))
    finally:
        dataset.close()


@pytest.fixture
def root(tmp_path):
    with lmdb.open(str(tmp_path / "data.lmdb"), subdir=False, map_size=1 << 20) as env:
        with env.begin(write=True) as txn:
            txn.put(b"0", pickle.dumps({"id": 7}))
            txn.put(b"length", pickle.dumps(1))
    return tmp_path


@pytest.mark.parametrize("first_to_close", [0, 1, 2])
def test_readers_share_storage_and_close_independently(root, first_to_close):
    original = _Dataset(root)
    assert original.get(0) == {"id": 7}
    readers = [original, copy.copy(original), _Dataset(root)]
    try:
        for reader in readers:
            assert len(reader) == 1
            assert reader.get(0) == {"id": 7}
        readers[first_to_close].close()
        readers[first_to_close].close()
        for index, reader in enumerate(readers):
            if index != first_to_close:
                assert reader.get(0) == {"id": 7}
        assert readers[first_to_close].get(0) == {"id": 7}
    finally:
        for reader in readers:
            reader.close()
    with lmdb.open(str(root / "data.lmdb"), subdir=False) as env:
        with env.begin(write=True) as txn:
            txn.put(b"0", pickle.dumps({"id": 8}))
    try:
        assert original.get(0) == {"id": 8}
    finally:
        original.close()


@pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
def test_child_reopens_parent_readers_without_invalidating_parent(root, method):
    dataset = _Dataset(root)
    assert dataset.get(0) == {"id": 7}
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    process = context.Process(target=_read_in_child, args=(dataset, queue))
    try:
        process.start()
        process.join(timeout=45)
        assert process.exitcode == 0
        assert queue.get(timeout=5) == {"id": 7}
        assert dataset.get(0) == {"id": 7}
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        queue.close()
        queue.join_thread()
        dataset.close()


def test_concurrent_readers_and_symlink_alias_share_environment(root):
    alias = root / "alias"
    alias.symlink_to(root, target_is_directory=True)
    original = _Dataset(root)
    assert original.get(0) == {"id": 7}

    def read_one(index):
        reader = _Dataset(alias if index % 2 else root)
        try:
            return reader.get(0)
        finally:
            reader.close()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            assert list(executor.map(read_one, range(12))) == [{"id": 7}] * 12
        assert original.get(0) == {"id": 7}
    finally:
        original.close()


def test_failed_transaction_acquisition_releases_its_environment(root, monkeypatch):
    from qqtools.torch.qlmdbdataset import _ReadonlyEnvironmentLease

    dataset = _Dataset(root)
    assert len(dataset) == 1

    def fail_begin(self, *, write=False):
        raise RuntimeError("injected begin failure")

    with monkeypatch.context() as patch:
        patch.setattr(_ReadonlyEnvironmentLease, "begin", fail_begin)
        with pytest.raises(RuntimeError, match="injected begin failure"):
            dataset.get(0)
    assert dataset._environments is None
    with lmdb.open(str(root / "data.lmdb"), subdir=False) as env:
        with env.begin() as txn:
            assert pickle.loads(txn.get(b"0")) == {"id": 7}
    try:
        assert dataset.get(0) == {"id": 7}
    finally:
        dataset.close()
