"""Collect the complete phase before selecting a deterministic CI shard."""

from pathlib import Path

import pytest


def select_nodeids(nodeids: list[str], index: int, count: int) -> list[str]:
    """Distribute sorted node IDs evenly, independent of collection order."""
    return sorted(nodeids)[index::count]


class CollectionShard:
    def __init__(self, config, index: int, count: int, directory: Path):
        if count < 1 or not 0 <= index < count:
            raise pytest.UsageError("qexp shard requires 0 <= index < count")
        self.config = config
        self.index = index
        self.count = count
        self.directory = directory

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        result = yield
        full = [item.nodeid for item in items]
        if not full or len(full) != len(set(full)):
            raise pytest.UsageError("qexp shard requires a nonempty, unique full collection")
        config._qexp_full_collection = full
        selected = set(select_nodeids(full, self.index, self.count))
        deselected = [item for item in items if item.nodeid not in selected]
        items[:] = [item for item in items if item.nodeid in selected]
        config.hook.pytest_deselected(items=deselected)
        worker = getattr(config, "workerinput", {}).get("workerid", "main")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"full-collection-{worker}.txt").write_text(
            "".join(f"{nodeid}\n" for nodeid in sorted(full)), encoding="utf-8"
        )
        return result
