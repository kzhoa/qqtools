"""Private, one-packet handoff between the idle controller and supervisor."""

from __future__ import annotations

import math
import re
import socket
import time

_READY = b"ready"
_MAX_DEADLINE_BYTES = 64
_MAX_PERMIT_BYTES = 1_024 + 1 + _MAX_DEADLINE_BYTES
_PERMIT_PATTERN = re.compile(rb"([!-~]{1,1024}) ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def create_idle_launch_permit() -> tuple[socket.socket, socket.socket]:
    """Return the controller and supervisor endpoints of an anonymous socket pair."""
    controller, supervisor = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET, 0)
    controller.setblocking(False)
    supervisor.setblocking(False)
    return controller, supervisor


def validate_idle_launch_permit_socket(endpoint: socket.socket) -> None:
    if (
        not isinstance(endpoint, socket.socket)
        or endpoint.fileno() < 3
        or endpoint.family != socket.AF_UNIX
        or endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET
        or endpoint.getsockname() != ""
        or endpoint.getpeername() != ""
    ):
        raise ValueError("idle launch permit requires a private Unix sequenced-packet socket")


def _send_packet(endpoint: socket.socket, packet: bytes) -> None:
    if endpoint.send(packet, socket.MSG_DONTWAIT) != len(packet):
        raise OSError("incomplete idle launch permit packet write")


def _poll_packet(endpoint: socket.socket, size: int) -> bytes | None:
    try:
        packet, _ancillary, flags, _address = endpoint.recvmsg(size, 0, socket.MSG_DONTWAIT)
    except BlockingIOError:
        return None
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not packet:
        raise ValueError("incomplete idle launch permit packet")
    return packet


def send_idle_launch_ready(endpoint: socket.socket) -> None:
    _send_packet(endpoint, _READY)


def poll_idle_launch_ready(endpoint: socket.socket) -> bool:
    """Return readiness without waiting, or raise on an invalid or closed channel."""
    packet = _poll_packet(endpoint, len(_READY))
    if packet is None:
        return False
    if packet != _READY:
        raise ValueError("invalid idle launch readiness packet")
    return True


def send_idle_launch_permit(endpoint: socket.socket, instance_id: str, deadline: float) -> None:
    """Queue the single launch permit while its monotonic deadline is still live."""
    packet = f"{instance_id} {deadline!r}".encode("ascii")
    if len(packet) > _MAX_PERMIT_BYTES or _PERMIT_PATTERN.fullmatch(packet) is None:
        raise ValueError("invalid idle launch permit packet")
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ValueError("idle launch permit deadline expired")
    _send_packet(endpoint, packet)


def poll_idle_launch_permit(endpoint: socket.socket, instance_id: str) -> float | None:
    packet = _poll_packet(endpoint, _MAX_PERMIT_BYTES)
    if packet is None:
        return None
    fields = _PERMIT_PATTERN.fullmatch(packet)
    if fields is None or len(fields[2]) > _MAX_DEADLINE_BYTES or fields[1].decode("ascii") != instance_id:
        raise ValueError("invalid idle launch permit packet")
    deadline = float(fields[2])
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ValueError("idle launch permit deadline expired")
    return deadline
