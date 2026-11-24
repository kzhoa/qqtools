"""Receive live Idle Job activation locks from a typed Project I/O worker."""

from __future__ import annotations

import os
import re
import select
import socket
import time
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from .idle_guard_digest import LOCKS_PER_PROJECT
from .project_io_protocol import PROJECT_IO_OVERDUE_SECONDS, ProjectIORequest
from .project_io_transport_support import ProjectIOProtocolError

if TYPE_CHECKING:
    from .project_io_executor import ProjectIOExecutor

_OPERATION = "idle_decision_guard"
_MAX_DESCRIPTORS_PER_PACKET = 100
_DESCRIPTOR_SIZE = array("i").itemsize
_ANCILLARY_SIZE = socket.CMSG_SPACE(_MAX_DESCRIPTORS_PER_PACKET * _DESCRIPTOR_SIZE)
_RESULT_SIZE = len(b"ok:") + 64
_RESULT_PATTERN = re.compile(rb"ok:([0-9a-f]{64})")


def _close_descriptors(descriptors: list[int]) -> None:
    while descriptors:
        with suppress(OSError):
            os.close(descriptors.pop())


def _receive_packet(channel: socket.socket, descriptors: list[int]) -> tuple[bytes, int]:
    packet, ancillary, flags, _address = channel.recvmsg(_RESULT_SIZE, _ANCILLARY_SIZE, socket.MSG_CMSG_CLOEXEC)
    is_valid = bool(packet) and not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
    initial_count = len(descriptors)
    for level, message_type, data in ancillary:
        if level != socket.SOL_SOCKET or message_type != socket.SCM_RIGHTS:
            is_valid = False
            continue
        complete_size = len(data) - len(data) % _DESCRIPTOR_SIZE
        received = array("i")
        received.frombytes(data[:complete_size])
        descriptors.extend(received)
        if complete_size != len(data):
            is_valid = False
    count = len(descriptors) - initial_count
    if not is_valid or count > _MAX_DESCRIPTORS_PER_PACKET:
        raise ProjectIOProtocolError("Idle Job guard channel closed or returned a malformed packet.")
    return packet, count


def _wait_for(channel: socket.socket, event: int, deadline: float) -> None:
    waiter = select.poll()
    waiter.register(channel, event)
    while (remaining := deadline - time.monotonic()) > 0:
        # poll uses integer milliseconds. Round down to stay within the caller's budget.
        if waiter.poll(int(min(remaining, 1.0) * 1000)):
            return
    raise TimeoutError("Idle Job guard response missed the decision deadline.")


def _receive_before(channel: socket.socket, descriptors: list[int], deadline: float) -> tuple[bytes, int]:
    while time.monotonic() < deadline:
        try:
            return _receive_packet(channel, descriptors)
        except BlockingIOError:
            _wait_for(channel, select.POLLIN, deadline)
    raise TimeoutError("Idle Job guard response missed the decision deadline.")


def _send_check(channel: socket.socket, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            sent = channel.send(b"check")
        except BlockingIOError:
            _wait_for(channel, select.POLLOUT, deadline)
            continue
        if sent != len(b"check"):
            raise ProjectIOProtocolError("Idle Job guard channel accepted an incomplete check.")
        return
    raise TimeoutError("Idle Job guard check missed the decision deadline.")


def _receive_check(
    channel: socket.socket, descriptors: list[int], descriptor_limit: int, deadline: float
) -> bytes | None:
    while True:
        packet, count = _receive_before(channel, descriptors, deadline)
        if packet == b"fds" and count > 0 and len(descriptors) <= descriptor_limit:
            continue
        if count == 0:
            if packet == b"unknown":
                return None
            if len(descriptors) == descriptor_limit and (matched := _RESULT_PATTERN.fullmatch(packet)):
                return matched[1]
        raise ProjectIOProtocolError("Idle Job guard returned invalid activation lock evidence.")


class IdleGuardClient:
    """Own one scheduler-thread session and its exact durable request."""

    def __init__(self, executor: ProjectIOExecutor) -> None:
        self._executor = executor
        self._request: ProjectIORequest | None = None
        self._channel: socket.socket | None = None
        self._is_ready = False
        self._last_response_at = 0.0
        self._has_orphan = False

    @property
    def needs_start(self) -> bool:
        """Return whether no known guard request or channel needs draining."""
        return self._request is None and not self._has_orphan

    def live(self, revision: int, epoch: str) -> bool:
        """Return whether an exact session can be asked for fresh evidence."""
        request = self._request
        return bool(
            self._channel is not None
            and request is not None
            and request.registry_revision == revision
            and request.executor_epoch == epoch
            and time.monotonic() - self._last_response_at < PROJECT_IO_OVERDUE_SECONDS
        )

    def ready(self, revision: int, epoch: str) -> bool:
        """Return whether the live guard is ready and has no pending channel events."""
        channel = self._channel
        if channel is None or not self._is_ready or not self.live(revision, epoch):
            return False
        waiter = select.poll()
        try:
            waiter.register(channel, select.POLLIN)
            return not waiter.poll(0)
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        """Disconnect the worker promptly, preserving its exact request for retirement."""
        channel = self._channel
        self._channel = None
        self._is_ready = False
        if channel is not None:
            with suppress(OSError):
                channel.close()

    def start(self, revision: int) -> None:
        """Prepare and start one worker under the normal executor lifecycle."""
        if not self.needs_start:
            return
        try:
            channel, child = socket.socketpair(type=socket.SOCK_SEQPACKET)
        except OSError:
            return
        try:
            with child:
                channel.setblocking(False)
                request = self._executor.prepare_idle_decision_guard(revision)
                self._request = request
                process = self._executor.start(request.request_id, idle_guard_socket=child)
            if process is not None and process.request == request:
                self._last_response_at = time.monotonic()
                self._channel = channel
        except (OSError, RuntimeError, ValueError):
            return
        finally:
            if self._channel is not channel:
                channel.close()

    def _note_response(self, request: ProjectIORequest) -> None:
        self._last_response_at = time.monotonic()
        self._executor.note_idle_guard_response(request.request_id)

    def _poll_channel(self, channel: socket.socket, request: ProjectIORequest) -> None:
        descriptors: list[int] = []
        try:
            packet, count = _receive_packet(channel, descriptors)
            if self._is_ready or packet != b"ready" or count:
                raise ProjectIOProtocolError("Idle Job guard returned an unsolicited response.")
            self._is_ready = True
            self._note_response(request)
        except BlockingIOError:
            return
        except (OSError, ProjectIOProtocolError):
            self.close()
        finally:
            _close_descriptors(descriptors)

    def service(self, revision: int, epoch: str | None, unresolved: Sequence[ProjectIORequest]) -> None:
        """Observe readiness and retire disconnected, obsolete, or orphaned requests."""
        active = self._request
        if active is not None and (
            active.registry_revision != revision
            or (epoch is not None and active.executor_epoch != epoch)
            or time.monotonic() - self._last_response_at >= PROJECT_IO_OVERDUE_SECONDS
        ):
            self.close()
        if epoch is None:
            return

        guards = [request for request in unresolved if request.operation_kind == _OPERATION]
        if active is not None:
            if active not in guards:
                self.close()
                guards.append(active)
            elif self._channel is not None:
                self._poll_channel(self._channel, active)
                if self._channel is not None and self._is_ready:
                    # Losing the optional cross-process status sample cannot revoke
                    # verified evidence on the live session.
                    with suppress(OSError, ValueError, RuntimeError):
                        self._executor.publish_idle_guard_response(active)

        self._has_orphan = False
        for request in guards:
            if request == active and self._channel is not None:
                continue
            request_id = request.request_id
            is_resolved = False
            try:
                # A disconnected guard has no reusable business result. Retire only
                # this exact, exited worker without reconciling or waiting for peers.
                is_resolved = self._executor.resolve_stale_idle_decision_guard(request_id, request)
            except (OSError, RuntimeError, ValueError):
                pass
            if request == active:
                if is_resolved:
                    self._request = None
            elif not is_resolved:
                self._has_orphan = True

    @contextmanager
    def authorize(self, revision: int, epoch: str, digest: str, project_count: int, deadline: float) -> Iterator[bool]:
        """Grant only fresh matching evidence and hold every lock through the caller's scope."""
        channel, request = self._channel, self._request
        if not self.live(revision, epoch):
            self.close()
        if channel is None or request is None or self._channel is not channel or not time.monotonic() < deadline:
            yield False
            return

        descriptors: list[int] = []
        try:
            is_authorized = False
            check_sent = False
            try:
                if not self._is_ready:
                    packet, count = _receive_before(channel, descriptors, deadline)
                    if packet != b"ready" or count:
                        raise ProjectIOProtocolError("Idle Job guard did not complete its readiness handshake.")
                    self._is_ready = True
                    self._note_response(request)
                _send_check(channel, deadline)
                check_sent = True
                received = _receive_check(channel, descriptors, project_count * LOCKS_PER_PROJECT, deadline)
                if time.monotonic() >= deadline or self._channel is not channel:
                    raise TimeoutError("Idle Job guard decision expired before its response was verified.")
                self._note_response(request)
                if time.monotonic() >= deadline or self._channel is not channel:
                    raise TimeoutError("Idle Job guard decision expired before it could be granted.")
                is_authorized = received is not None and received.decode("ascii") == digest
            except TimeoutError:
                # Startup can legitimately take longer than one decision turn.
                # Once a check has been sent, its late response must never be reused.
                if check_sent and self._channel is channel:
                    self.close()
            except (OSError, ProjectIOProtocolError):
                if self._channel is channel:
                    self.close()
            if not is_authorized:
                _close_descriptors(descriptors)
            yield is_authorized
        finally:
            _close_descriptors(descriptors)
