import copy
import gc
import importlib
import multiprocessing
import os
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from qqtools.torch.qlmdbdataset import qLmdbDataset

lmdb = pytest.importorskip("lmdb")
pytestmark = pytest.mark.integration


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


def _collect_reader_during_registry_update(root, phase):
    storage = importlib.import_module("qqtools.torch.qlmdbdataset")
    gc.disable()
    try:
        reader = _Dataset(root)
        assert reader.get(0) == {"id": 7}
        reader.cycle = reader
        previous = weakref.ref(reader)
        del reader

        class CollectingRegistry(dict):
            has_collected = False

            def collect_once(self, operation):
                if operation == phase and not self.has_collected:
                    self.has_collected = True
                    gc.collect()
                    assert previous() is None

            def get(self, key, default=None):
                entry = super().get(key, default)
                self.collect_once("lookup")
                return entry

            def __setitem__(self, key, value):
                self.collect_once("increment")
                super().__setitem__(key, value)

        storage._environments = CollectingRegistry(storage._environments)
        lease = storage._ReadonlyEnvironmentLease(root / "data.lmdb")
        try:
            assert storage._environments.has_collected
            assert storage._environments[(root / "data.lmdb").resolve()][1] == 1
            with lease.begin() as transaction:
                assert pickle.loads(transaction.get(b"0")) == {"id": 7}
        finally:
            lease.close()
        assert not storage._environments
        # Native handle was actually released, not just removed from bookkeeping.
        with lmdb.open(str(root / "data.lmdb"), subdir=False) as environment:
            with environment.begin(write=True) as transaction:
                transaction.put(b"0", pickle.dumps({"id": 8}))
    finally:
        gc.enable()


@pytest.mark.parametrize("phase", ["lookup", "increment"])
def test_gc_finalizer_cannot_deadlock_or_invalidate_inflight_lease(root, phase):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_collect_reader_during_registry_update, args=(root, phase))
    try:
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0, "GC reentered an unfinished LMDB registry update"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
