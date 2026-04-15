from __future__ import annotations

import multiprocessing
import time

import pytest

from qqtools.plugins.qexp.v2.layout import init_shared_root
from qqtools.plugins.qexp.v2.locking import (
    FileLock,
    LockTimeout,
    batch_lock,
    clean_lock,
    submit_lock,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1", runtime_root=tmp_path / "runtime")


class TestFileLock:
    def test_acquire_release(self, cfg):
        lock = submit_lock(cfg)
        lock.acquire()
        assert lock._fd is not None
        lock.release()
        assert lock._fd is None

    def test_context_manager(self, cfg):
        with submit_lock(cfg) as lock:
            assert lock._fd is not None
        assert lock._fd is None

    def test_double_release_safe(self, cfg):
        lock = submit_lock(cfg)
        lock.acquire()
        lock.release()
        lock.release()

    def test_exclusive(self, cfg):
        def _hold_lock(lock_path_str, acquired_event, release_event):
            lock = FileLock(lock_path_str, timeout=5.0)
            lock.acquire()
            acquired_event.set()
            release_event.wait(timeout=10.0)
            lock.release()

        from qqtools.plugins.qexp.v2.layout import submit_lock_path
        lp = submit_lock_path(cfg)

        acquired = multiprocessing.Event()
        release = multiprocessing.Event()
        proc = multiprocessing.Process(
            target=_hold_lock, args=(lp, acquired, release)
        )
        proc.start()
        acquired.wait(timeout=5.0)

        try:
            with pytest.raises(LockTimeout):
                FileLock(lp, timeout=0.2).acquire()
        finally:
            release.set()
            proc.join(timeout=5.0)

    def test_is_stale(self, tmp_path):
        lp = tmp_path / "test.lock"
        lp.touch()
        lock = FileLock(lp)
        assert not lock.is_stale(max_age_seconds=300.0)
        assert lock.is_stale(max_age_seconds=0.0)


class TestLockFactories:
    def test_submit_lock(self, cfg):
        with submit_lock(cfg):
            pass

    def test_batch_lock(self, cfg):
        with batch_lock(cfg):
            pass

    def test_clean_lock(self, cfg):
        with clean_lock(cfg):
            pass
