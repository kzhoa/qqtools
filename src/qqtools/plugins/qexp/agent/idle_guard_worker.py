"""Isolate the shared Project reads for a machine-local Idle Job decision."""

from __future__ import annotations

import array
import fcntl
import os
import select
import socket
import stat
import time
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config_types import RootConfig
from ..layout import is_idle_job_activation_writers_root
from ..lease import holder_safe_deadline, load_lease_policy
from ..runtime import project_activation
from ..runtime.paths import lock_path, shared_paths
from ..runtime.records import validate_identifier
from ..runtime.registration_protocol import read_registration_document, validate_registration
from .bindings import ProjectBinding
from .idle_guard_digest import (
    IDLE_DECISION_SECONDS,
    MAX_REGISTRATION_HORIZON_NS,
    IdleGuardCheckpoint,
    idle_guard_digest,
)
from .project_io.protocol import ProjectIORequest

_IDLE_MILLISECONDS = 15_000
_ACTIVATION_LOCK_NAME = "project-activation-v1.lock"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_LOCK_FLAGS = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_UNAVAILABLE_ERRORS = (OSError, RuntimeError, ValueError, KeyError, TypeError)


def _send_project_locks(channel: socket.socket, root: Path) -> tuple[os.stat_result, list[tuple[Path, os.stat_result]]]:
    if root.resolve(strict=True) != root:
        raise ValueError("Idle Job guard requires a canonical Project root.")
    schema_lock = lock_path(root, "schema")
    locks = schema_lock.parent
    activation_lock = locks / _ACTIVATION_LOCK_NAME
    with ExitStack() as directories:
        root_fd = os.open(root, _DIRECTORY_FLAGS)
        directories.callback(os.close, root_fd)
        root_stat = os.fstat(root_fd)
        locks_fd = os.open(locks.name, _DIRECTORY_FLAGS, dir_fd=root_fd)
        directories.callback(os.close, locks_fd)
        opened = []
        for path, flag, open_flags in (
            (schema_lock, fcntl.LOCK_EX, _LOCK_FLAGS),
            (locks / "registrations.lock", fcntl.LOCK_SH, _LOCK_FLAGS),
            (activation_lock, fcntl.LOCK_SH, _LOCK_FLAGS | os.O_CREAT),
        ):
            descriptor = os.open(path.name, open_flags, 0o600, dir_fd=locks_fd)
            try:
                lock_stat = os.fstat(descriptor)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise ValueError("Idle Job guard lock must be a regular file.")
                fcntl.flock(descriptor, flag | fcntl.LOCK_NB)
                channel.sendmsg([b"fds"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [descriptor]))])
                opened.append((path, lock_stat))
            finally:
                # The receiver or its socket queue now owns the lease. A timeout
                # can release it even if a later Project read never returns.
                os.close(descriptor)
    return root_stat, opened


def _read_checkpoint(binding: ProjectBinding) -> tuple[IdleGuardCheckpoint, int]:
    root = binding.shared_root
    project = shared_paths(root)["project"]
    if not stat.S_ISDIR(project.lstat().st_mode):
        raise ValueError("Idle Job guard Project identity directory is invalid.")
    identity = project_activation._read_regular_json(project / "identity.json")
    if identity != {"project": {"project_id": binding.project_id, "shared_root": str(root)}}:
        raise ValueError("Idle Job guard Project identity differs from its registry binding.")
    cfg = RootConfig.from_canonical_paths(root, root.parent, binding.machine_name, Path(binding.runtime_root))
    if not is_idle_job_activation_writers_root(cfg):
        raise ValueError("Idle Job guard requires Project activation writer support.")
    envelope = read_registration_document(cfg)
    record = envelope.get("registration") if envelope else None
    validate_registration(record, binding.project_id, cfg, allow_source=True)
    if (
        record["generation"] != binding.registration_generation
        or record["runtime_instance_id"] != binding.runtime_instance_id
        or record["runtime_root"] != binding.runtime_root
        or record["state"] != "eligible"
    ):
        raise ValueError("Idle Job guard registration differs from its eligible registry binding.")
    policy = load_lease_policy(cfg)
    try:
        safe_expiry = holder_safe_deadline(record["eligibility_expires_at"], policy.max_clock_skew_seconds)
    except OverflowError as exc:
        raise ValueError("Idle Job guard registration deadline is out of range.") from exc
    # Sample monotonic time before UTC so a pause between reads shortens the proof.
    monotonic_now_ns = time.monotonic_ns()
    remaining = safe_expiry - datetime.now(timezone.utc)
    if remaining <= timedelta(seconds=IDLE_DECISION_SECONDS):
        raise ValueError("Idle Job guard requires registration eligibility through the decision deadline.")
    # Account for UTC sampling precision when converting to integer nanoseconds.
    remaining_us = remaining // timedelta(microseconds=1) - 1
    registration_horizon_ns = monotonic_now_ns + remaining_us * 1_000
    if not 0 < registration_horizon_ns <= MAX_REGISTRATION_HORIZON_NS:
        raise ValueError("Idle Job guard registration deadline is out of range.")
    checkpoint = project_activation.read_project_activation(root)
    try:
        project_activation.activation_pending_path(root).lstat()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("Idle Job guard found a pending Project activation.")
    current = checkpoint["project_activation"] if checkpoint is not None else {}
    row = (
        binding.project_id,
        binding.registration_generation,
        str(root),
        current.get("epoch"),
        current.get("sequence"),
    )
    return row, registration_horizon_ns


def _answer_check(channel: socket.socket, bindings: tuple[ProjectBinding, ...]) -> None:
    try:
        opened = [_send_project_locks(channel, binding.shared_root) for binding in bindings]
        observations = [_read_checkpoint(binding) for binding in bindings]
        digest = idle_guard_digest(row for row, _ in observations)
        horizon_ns = min((deadline_ns for _, deadline_ns in observations), default=None)
        for binding, (root_stat, locks) in zip(bindings, opened, strict=True):
            if not os.path.samestat(root_stat, binding.shared_root.lstat()):
                raise RuntimeError("Idle Job guard Project root changed during the check.")
            for path, lock_stat in locks:
                if not os.path.samestat(lock_stat, path.lstat()):
                    raise RuntimeError("Idle Job guard Project lock changed during the check.")
    except _UNAVAILABLE_ERRORS:
        channel.send(b"unknown")
        return
    response = f"ok:{digest}" if horizon_ns is None else f"ok:{digest}:{horizon_ns}"
    channel.send(response.encode("ascii"))


def _enabled_bindings(
    request: ProjectIORequest, roster: tuple[ProjectBinding, ...], root: Path
) -> tuple[ProjectBinding, ...]:
    bindings = tuple(binding for binding in roster if binding.enabled)
    if len({binding.project_id for binding in bindings}) != len(bindings) or len(
        {binding.shared_root for binding in bindings}
    ) != len(bindings):
        raise ValueError("Idle Job guard requires unique Projects and shared roots.")
    for binding in bindings:
        validate_identifier(binding.project_id, "project_id")
        validate_identifier(binding.registration_generation, "registration_generation")
        if binding.runtime_instance_id != request.runtime_id or binding.runtime_root != str(root):
            raise ValueError("Idle Job guard binding belongs to a different machine runtime.")
    return tuple(sorted(bindings, key=lambda binding: str(binding.shared_root)))


def serve_idle_decision_guard(
    channel: socket.socket,
    request: ProjectIORequest,
    runtime_root: Path,
    *,
    read_registry: Callable[[], tuple[int, tuple[ProjectBinding, ...]]],
    is_worker_current: Callable[[], bool],
) -> None:
    """Serve checks for the immutable registry roster over the inherited channel."""
    channel.setblocking(False)
    try:
        registry = read_registry()
        if registry[0] != request.registry_revision or not is_worker_current():
            raise RuntimeError("Idle Job guard request is no longer current.")
        bindings = _enabled_bindings(request, registry[1], runtime_root)
    except _UNAVAILABLE_ERRORS:
        channel.send(b"unknown")
        raise
    channel.send(b"ready")

    incoming = select.poll()
    incoming.register(channel, select.POLLIN)
    while incoming.poll(_IDLE_MILLISECONDS):
        packet, _ancillary, flags, _peer = channel.recvmsg(5)
        if not packet:
            return
        try:
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or packet != b"check":
                raise ValueError("Idle Job guard check is invalid.")
            if not is_worker_current() or read_registry() != registry:
                raise RuntimeError("Idle Job guard check is invalid or no longer current.")
        except _UNAVAILABLE_ERRORS:
            channel.send(b"unknown")
            raise
        _answer_check(channel, bindings)
