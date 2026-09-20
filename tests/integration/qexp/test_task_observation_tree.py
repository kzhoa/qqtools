"""Ordered observation partitions obey bounded seek and mutation semantics."""

import random

import pytest

from qqtools.plugins.qexp.runtime.observation.tree import IndexTree

pytestmark = pytest.mark.integration


def collect(tree, size=17):
    result = []
    after = None
    for _ in range(10000):
        batch = tree.candidates(after, max_candidates=size)
        assert batch.pages <= 1024
        assert batch.bytes_read <= 8 * 1024 * 1024
        assert len(batch.keys) <= size
        assert list(batch.keys) == sorted(set(batch.keys))
        assert all(after is None or key > after for key in batch.keys)
        result.extend(batch.keys)
        if batch.exhausted:
            return result
        assert batch.keys, "nonterminal pages must advance"
        after = batch.keys[-1]
    raise AssertionError("tree traversal failed to terminate")


def test_seek_split_prefix_keys_and_delete_match_ordered_set(tmp_path):
    tree = IndexTree(tmp_path)
    expected = {f"task-{n:04d}" for n in range(320)} | {"task", "task-", "A", "Z", "a", "z", "_", "."}
    order = list(sorted(expected))
    random.Random(71).shuffle(order)
    for key in order:
        tree.add(key)
    assert collect(tree) == sorted(expected)
    for key in order[::3]:
        tree.discard(key)
        expected.remove(key)
    for key in ["task-000", "task-0000x", "task-0000", "task-0000xy", "task-0000w"]:
        tree.add(key)
        expected.add(key)
    assert collect(tree, 1) == sorted(expected)
    for key in list(expected):
        tree.discard(key)
    assert collect(tree) == []
    tree.add("recreated")
    assert collect(tree) == ["recreated"]


def test_sparse_and_absent_partitions_do_not_enumerate_directories(tmp_path, monkeypatch):
    import os

    broad = IndexTree(tmp_path, group="big")
    sparse = IndexTree(tmp_path, phase="running", group="big")
    for n in range(500):
        broad.add(f"task-{n:05d}")
    for key in ("task-00100", "task-00499"):
        sparse.add(key)

    def forbidden(*args, **kwargs):
        raise AssertionError("query enumerated a directory")

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", forbidden)
        guard.setattr(os, "listdir", forbidden)
        assert collect(sparse) == ["task-00100", "task-00499"]
        absent = IndexTree(tmp_path, phase="missing", group="big")
        batch = absent.candidates(None)
        assert batch.keys == () and batch.exhausted
        assert batch.pages <= 1
        batch = broad.candidates("task-00495", max_candidates=2)
        assert batch.keys == ("task-00496", "task-00497")
        assert batch.pages <= 8


def test_budget_and_corruption_never_return_false_empty_success(tmp_path):
    tree = IndexTree(tmp_path)
    for n in range(260):
        tree.add(f"item-{n:04d}")
    with pytest.raises((ValueError, OSError)):
        tree.candidates(None, max_pages=1, max_bytes=1)
    roots = list(tmp_path.rglob("root.json"))
    assert len(roots) == 1
    roots[0].write_text('{"version":1,"keys":["z","a"]}')
    with pytest.raises((ValueError, OSError)):
        tree.candidates(None)


@pytest.mark.parametrize("key", ["../escape", "bad/key", "", "x" * 251, "\x00", "汉字"])
def test_keys_cannot_address_paths_or_exceed_record_bound(tmp_path, key):
    tree = IndexTree(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        tree.add(key)
