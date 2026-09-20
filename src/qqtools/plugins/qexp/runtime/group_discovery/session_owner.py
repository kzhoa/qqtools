"""Machine-resident ownership for one cooperative Group source session."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Protocol


class _ProjectionDriver(Protocol):
    @property
    def is_closed(self) -> bool: ...

    @property
    def summary(self) -> object | None: ...

    def request_checkpoint(self) -> None: ...

    def request_close(self) -> None: ...

    def advance(
        self,
        io: Any,
        *,
        max_processed_bytes: int = 65_536,
        soft_deadline: float | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """Identity of one source projection session."""

    project_id: str
    source: Path
    scratch: Path
    operation_id: str
    group: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.project_id, "project_id"),
            (self.operation_id, "operation_id"),
            (self.group, "group"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(self, "source", _normalized_absolute_path(self.source, "source"))
        object.__setattr__(self, "scratch", _normalized_absolute_path(self.scratch, "scratch"))


@dataclass(frozen=True, slots=True)
class SourceServiceStep:
    """Result exposed by the resident source owner."""

    state: str
    reason: str | None
    summary: object | None


def _normalized_absolute_path(value: Path, name: str) -> Path:
    try:
        candidate = Path(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a path") from exc
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be absolute")
    normalized = Path(os.path.normpath(str(candidate)))
    if candidate != normalized:
        raise ValueError(f"{name} must be normalized")
    return normalized


def _default_driver_factory(request: SourceRequest) -> _ProjectionDriver:
    """Construct the production driver only when a source session is needed."""
    from .driver import ProjectionDriver

    return ProjectionDriver(request.source, request.scratch, request.operation_id, request.group)


class GroupSourceOwner:
    """Own one source driver across bounded, ephemeral upgrade invocations."""

    _CLEANUP_IO_BYTES = 262_144
    _CLEANUP_OPERATIONS = 32
    _CLEANUP_THREAD_NAME = "qexp-group-source-close"

    def __init__(
        self,
        *,
        driver_factory: Callable[[SourceRequest], _ProjectionDriver] | None = None,
    ) -> None:
        if driver_factory is not None and not callable(driver_factory):
            raise TypeError("driver_factory must be callable or None")
        self._driver_factory = driver_factory if driver_factory is not None else _default_driver_factory
        self._mutex = Lock()
        self._closed_event = Event()
        self._driver: _ProjectionDriver | None = None
        self._current_request: SourceRequest | None = None
        self._executor_assigned = False
        self._closing = False
        self._cleanup_started = False
        self._cleanup_thread: Thread | None = None
        self._driver_close_requested = False
        self._cleanup_error: BaseException | None = None

    @property
    def is_closed(self) -> bool:
        """Return whether permanent owner shutdown has completed."""
        return self._closed_event.is_set()

    @property
    def cleanup_error(self) -> BaseException | None:
        """Return the first error observed by permanent cleanup, if any."""
        return self._cleanup_error

    @property
    def current_request(self) -> SourceRequest | None:
        """Return the resident request identity, if a session is retained."""
        return self._current_request

    def advance(
        self,
        request: SourceRequest,
        io: Any,
        *,
        max_processed_bytes: int = 65_536,
        soft_deadline: float | None = None,
    ) -> SourceServiceStep:
        """Advance the exact resident request using caller-owned I/O budget."""
        _require_source_request(request)
        driver, status = self._reserve_advance(request)
        if status is not None:
            return status

        try:
            if driver is None:
                # Shutdown can win the race after the executor reservation.  The
                # reservation's finally handoff will complete permanent cleanup.
                if self._closing:
                    return self._closing_step()
                driver = self._driver_factory(request)
                self._publish_driver(request, driver)
            if self._closing:
                return self._closing_step()
            result = driver.advance(
                io,
                max_processed_bytes=max_processed_bytes,
                soft_deadline=soft_deadline,
            )
            return self._service_step(driver, result)
        finally:
            self._finish_executor(driver)

    def request_checkpoint(self, request: SourceRequest) -> bool:
        """Request a checkpoint without entering the driver's I/O executor."""
        _require_source_request(request)
        if not self._mutex.acquire(blocking=False):
            return False
        try:
            if (
                self._closed_event.is_set()
                or self._closing
                or self._executor_assigned
                or self._driver is None
                or self._current_request != request
                or self._driver_close_requested
            ):
                return False
            self._driver.request_checkpoint()
            return True
        finally:
            self._mutex.release()

    def release(self, io: Any) -> SourceServiceStep:
        """Request resident-driver cleanup and spend one caller-provided slice on it."""
        driver, status = self._reserve_release()
        if status is not None:
            return status
        if driver is None:
            return SourceServiceStep("closed", None, None)

        try:
            if not self._driver_is_closed(driver):
                self._request_close_once(driver)
            if self._driver_is_closed(driver):
                return SourceServiceStep("closed", None, self._driver_summary(driver))
            result = driver.advance(
                io,
                max_processed_bytes=65_536,
                soft_deadline=None,
            )
            return self._service_step(driver, result)
        finally:
            self._finish_executor(driver)

    def shutdown(self) -> None:
        """Request permanent owner shutdown without performing caller-thread I/O."""
        cleanup_driver: _ProjectionDriver | None = None
        cleanup_thread: Thread | None = None
        if not self._mutex.acquire():
            return
        try:
            if self._closed_event.is_set():
                return
            self._closing = True
            if self._executor_assigned or self._cleanup_started:
                return
            cleanup_driver = self._driver
            if cleanup_driver is None:
                self._clear_session_locked()
                self._closed_event.set()
                return
            self._executor_assigned = True
            self._cleanup_started = True
            cleanup_thread = Thread(
                target=self._cleanup_thread_main,
                args=(cleanup_driver,),
                name=self._CLEANUP_THREAD_NAME,
                daemon=True,
            )
            self._cleanup_thread = cleanup_thread
        finally:
            self._mutex.release()

        if cleanup_thread is not None:
            try:
                cleanup_thread.start()
            except BaseException as exc:
                self._record_cleanup_error(exc)
                self._finish_cleanup(cleanup_driver)

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Wait for permanent cleanup to finish."""
        return self._closed_event.wait(timeout)

    def _reserve_advance(self, request: SourceRequest) -> tuple[_ProjectionDriver | None, SourceServiceStep | None]:
        if not self._mutex.acquire(blocking=False):
            return None, SourceServiceStep("waiting", "executor_busy", None)
        try:
            if self._closed_event.is_set():
                return None, SourceServiceStep("closed", None, None)
            if self._closing:
                return None, SourceServiceStep("waiting", "closing", None)
            if self._executor_assigned:
                return None, SourceServiceStep("waiting", "executor_busy", None)
            if self._driver is not None and self._current_request != request:
                return None, SourceServiceStep("waiting", "source_busy", None)
            if self._driver_close_requested:
                return None, SourceServiceStep("waiting", "closing", None)
            if self._driver is None:
                self._current_request = request
                self._driver_close_requested = False
            self._executor_assigned = True
            return self._driver, None
        finally:
            self._mutex.release()

    def _reserve_release(self) -> tuple[_ProjectionDriver | None, SourceServiceStep | None]:
        if not self._mutex.acquire(blocking=False):
            return None, SourceServiceStep("waiting", "executor_busy", None)
        try:
            if self._closed_event.is_set():
                return None, SourceServiceStep("closed", None, None)
            if self._closing:
                return None, SourceServiceStep("waiting", "closing", None)
            if self._executor_assigned:
                return None, SourceServiceStep("waiting", "executor_busy", None)
            if self._driver is None:
                return None, SourceServiceStep("closed", None, None)
            self._executor_assigned = True
            return self._driver, None
        finally:
            self._mutex.release()

    def _publish_driver(self, request: SourceRequest, driver: _ProjectionDriver) -> None:
        if driver is None:
            raise TypeError("driver_factory returned None")
        if not self._mutex.acquire():
            raise RuntimeError("unable to publish source driver")
        try:
            if self._driver is not None and self._driver is not driver:
                raise RuntimeError("source driver was replaced while constructing a session")
            self._driver = driver
            self._current_request = request
        finally:
            self._mutex.release()

    def _finish_executor(self, driver: _ProjectionDriver | None) -> None:
        cleanup_driver: _ProjectionDriver | None = None
        should_cleanup = False
        closed = self._driver_is_closed(driver) if driver is not None else True
        if not self._mutex.acquire():
            return
        try:
            if self._closing:
                cleanup_driver = driver if driver is not None else self._driver
                if not self._cleanup_started:
                    self._cleanup_started = True
                    should_cleanup = True
                # Keep executor ownership until cleanup has completed.  This is
                # the handoff that prevents shutdown from racing a final advance.
            else:
                if driver is None:
                    self._clear_session_locked()
                elif self._driver is driver and closed:
                    self._clear_session_locked()
                self._executor_assigned = False
        finally:
            self._mutex.release()

        if should_cleanup:
            self._run_cleanup(cleanup_driver)
            self._finish_cleanup(cleanup_driver)

    def _cleanup_thread_main(self, driver: _ProjectionDriver) -> None:
        self._run_cleanup(driver)
        self._finish_cleanup(driver)

    def _run_cleanup(self, driver: _ProjectionDriver | None) -> None:
        if driver is None:
            return
        if self._driver_is_closed(driver, record_error=True):
            return
        try:
            self._request_close_once(driver)
        except BaseException as exc:
            self._record_cleanup_error(exc)
            if not self._driver_is_closed(driver, record_error=True):
                return
        while not self._driver_is_closed(driver, record_error=True):
            try:
                from .slice_io import SliceIO

                io = SliceIO(
                    max_io_bytes=self._CLEANUP_IO_BYTES,
                    max_operations=self._CLEANUP_OPERATIONS,
                )
                result = driver.advance(
                    io,
                    max_processed_bytes=65_536,
                    soft_deadline=None,
                )
            except BaseException as exc:
                self._record_cleanup_error(exc)
                if not self._driver_is_closed(driver, record_error=True):
                    # An exception while the driver remains open violates the
                    # driver contract.  Keep the owner open so this is visible
                    # through wait_closed() and cleanup_error.
                    return
                return
            if self._driver_is_closed(driver, record_error=True):
                return
            if getattr(result, "state", None) == "closed":
                self._record_cleanup_error(RuntimeError("driver reported closed without is_closed"))
                return

    def _finish_cleanup(self, driver: _ProjectionDriver | None) -> None:
        closed = driver is None or self._driver_is_closed(driver, record_error=True)
        if not self._mutex.acquire():
            return
        try:
            if closed:
                if driver is None or self._driver is driver:
                    self._clear_session_locked()
                self._executor_assigned = False
                self._cleanup_started = False
                self._closed_event.set()
            else:
                self._executor_assigned = False
                self._cleanup_started = False
            self._cleanup_thread = None
        finally:
            self._mutex.release()

    def _request_close_once(self, driver: _ProjectionDriver) -> None:
        if self._driver_close_requested:
            return
        self._driver_close_requested = True
        driver.request_close()

    def _service_step(self, driver: _ProjectionDriver, result: object) -> SourceServiceStep:
        state = getattr(result, "state")
        reason = getattr(result, "reason", None)
        return SourceServiceStep(state, reason, self._driver_summary(driver))

    @staticmethod
    def _driver_summary(driver: _ProjectionDriver) -> object | None:
        return getattr(driver, "summary", None)

    def _closing_step(self) -> SourceServiceStep:
        if self._closed_event.is_set():
            return SourceServiceStep("closed", None, None)
        return SourceServiceStep("waiting", "closing", None)

    def _clear_session_locked(self) -> None:
        self._driver = None
        self._current_request = None
        self._driver_close_requested = False

    def _record_cleanup_error(self, error: BaseException) -> None:
        if not self._mutex.acquire():
            return
        try:
            if self._cleanup_error is None:
                self._cleanup_error = error
        finally:
            self._mutex.release()

    def _driver_is_closed(self, driver: _ProjectionDriver | None, *, record_error: bool = False) -> bool:
        if driver is None:
            return True
        try:
            value = driver.is_closed
        except BaseException as exc:
            if record_error:
                # A missing or failing lifecycle property is a driver-contract
                # failure; cleanup cannot claim closure from it.
                self._record_cleanup_error(exc)
            return False
        if not isinstance(value, bool):
            if record_error:
                self._record_cleanup_error(TypeError("driver.is_closed must be a bool"))
            return False
        return value


def _require_source_request(request: SourceRequest) -> None:
    if not isinstance(request, SourceRequest):
        raise TypeError("request must be a SourceRequest")


__all__ = ["GroupSourceOwner", "SourceRequest", "SourceServiceStep"]
