"""Process-local launch handoffs and their membership revision."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, MutableMapping, ValuesView
from contextlib import contextmanager
from threading import RLock
from typing import Any

_HandoffKey = tuple[str, str]


class LaunchHandoffState(MutableMapping[_HandoffKey, Any]):
    """Track pending identities without coupling Idle Job to their record shape."""

    def __init__(self) -> None:
        self._entries: dict[_HandoffKey, Any] = {}
        self._membership_revision = 0
        self._lock = RLock()

    def __getitem__(self, key: _HandoffKey) -> Any:
        with self._lock:
            return self._entries[key]

    def __setitem__(self, key: _HandoffKey, value: Any) -> None:
        with self._lock:
            is_new = key not in self._entries
            self._entries[key] = value
            if is_new:
                self._membership_revision += 1

    def __delitem__(self, key: _HandoffKey) -> None:
        with self._lock:
            del self._entries[key]
            self._membership_revision += 1

    def __iter__(self) -> Iterator[_HandoffKey]:
        with self._lock:
            return iter(tuple(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def items(self) -> ItemsView[_HandoffKey, Any]:
        with self._lock:
            return self._entries.copy().items()

    def values(self) -> ValuesView[Any]:
        with self._lock:
            return self._entries.copy().values()

    def clear(self) -> None:
        with self._lock:
            if self._entries:
                self._entries.clear()
                self._membership_revision += 1

    def pop(self, key: _HandoffKey, /, *default: Any) -> Any:
        with self._lock:
            return super().pop(key, *default)

    def popitem(self) -> tuple[_HandoffKey, Any]:
        with self._lock:
            item = self._entries.popitem()
            self._membership_revision += 1
            return item

    def setdefault(self, key: _HandoffKey, default: Any = None) -> Any:
        with self._lock:
            return super().setdefault(key, default)

    def snapshot(self) -> tuple[int, bool]:
        """Read the membership revision and pending state together."""
        with self._lock:
            return self._membership_revision, bool(self._entries)

    @contextmanager
    def read_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Hold membership stable through a machine-local Idle Job decision."""
        acquired = self._lock.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()
