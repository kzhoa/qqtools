"""Single-slot background reader for deadline-bound machine observations."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future
from threading import Condition, Thread
from typing import TypeVar

_T = TypeVar("_T")


class IdleLocalReader:
    """Discard late reads without accumulating queued work or threads."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._work: Callable[[], None] | None = None
        self._closed = False
        Thread(target=self._run, name="qexp-idle-local-read", daemon=True).start()

    def read_before(self, read: Callable[[], _T], *, deadline: float) -> _T | None:
        """Run one read-only operation. Late results are private to the expired call."""
        result: Future[_T] = Future()

        def work() -> None:
            if time.monotonic() < deadline:
                try:
                    result.set_result(read())
                except Exception as exc:
                    result.set_exception(exc)

        with self._condition:
            if self._closed or self._work is not None or time.monotonic() >= deadline:
                return None
            self._work = work
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: self._closed or self._work is not work,
                timeout=max(0, deadline - time.monotonic()),
            )
            if self._closed or time.monotonic() >= deadline or not result.done():
                return None
            return result.result()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._work is not None or self._closed)
                if self._closed:
                    return
                work = self._work
            assert work is not None
            work()
            with self._condition:
                self._work = None
                self._condition.notify_all()
            del work
