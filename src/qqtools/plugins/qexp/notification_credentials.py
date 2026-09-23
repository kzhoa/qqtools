"""Owner-private immutable Feishu webhook credential storage."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .runtime.records import utc_now

_SCHEMA_VERSION = 1
_CREDENTIAL_ID_RE = re.compile(r"[a-f0-9]{32}", re.ASCII)
_CREATED_AT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", re.ASCII)
_WEBHOOK_PATH_PREFIX = "/open-apis/bot/v2/hook/"
_MAX_CREDENTIAL_BYTES = 65_536


def _runtime_root_path(runtime_root: Path) -> Path:
    if not isinstance(runtime_root, Path):
        raise ValueError("runtime_root must be a pathlib.Path.")
    expanded = runtime_root.expanduser()
    if "\x00" in os.fspath(expanded):
        raise ValueError("runtime_root is invalid.")
    return Path(os.path.abspath(expanded))


def _validate_credential_id(credential_id: str) -> str:
    if not isinstance(credential_id, str) or _CREDENTIAL_ID_RE.fullmatch(credential_id) is None:
        raise ValueError("credential_id must be 32 lowercase UUID hex characters.")
    return credential_id


def credential_path(runtime_root: Path, credential_id: str) -> Path:
    """Return the private storage path for one canonical credential ID."""
    root = _runtime_root_path(runtime_root)
    identifier = _validate_credential_id(credential_id)
    return root / "notifications" / "credentials" / f"{identifier}.json"


def _validate_webhook(webhook: str) -> str:
    if not isinstance(webhook, str) or not webhook:
        raise ValueError("webhook must be a valid Feishu bot URL.")
    if webhook != webhook.strip() or any(character.isspace() or ord(character) < 0x20 for character in webhook):
        raise ValueError("webhook must be a valid Feishu bot URL.")
    try:
        if len(webhook.encode("utf-8")) > _MAX_CREDENTIAL_BYTES - 256:
            raise ValueError("webhook exceeds the supported credential size.")
    except UnicodeEncodeError:
        raise ValueError("webhook must be a valid Feishu bot URL.") from None
    if "?" in webhook or "#" in webhook:
        raise ValueError("webhook must not contain a query or fragment.")
    try:
        parsed = urlsplit(webhook)
        scheme = parsed.scheme
        network_location = parsed.netloc
        host = parsed.hostname
        username = parsed.username
        password = parsed.password
        path = parsed.path
    except ValueError:
        raise ValueError("webhook must be a valid Feishu bot URL.") from None
    if (
        scheme != "https"
        or network_location != "open.feishu.cn"
        or host != "open.feishu.cn"
        or username is not None
        or password is not None
        or not path.startswith(_WEBHOOK_PATH_PREFIX)
    ):
        raise ValueError("webhook must use the canonical Feishu HTTPS bot URL.")
    token = path[len(_WEBHOOK_PATH_PREFIX) :]
    if not token or "/" in token:
        raise ValueError("webhook must contain a non-empty bot token.")
    return webhook


def _validate_created_at(value: Any) -> str:
    if not isinstance(value, str) or _CREATED_AT_RE.fullmatch(value) is None:
        raise ValueError("credential created_at must be a UTC timestamp with second precision.")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError("credential created_at must be a valid UTC timestamp.") from None
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("credential created_at must be canonical UTC.")
    return value


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return flags | getattr(os, "O_CLOEXEC", 0)


def _open_runtime_root(runtime_root: Path) -> int:
    descriptor = os.open(runtime_root, _directory_flags())
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid():
        os.close(descriptor)
        raise ValueError("runtime_root must be a directory owned by the current user.")
    return descriptor


def _open_private_directory(parent_fd: int, name: str, *, create: bool) -> int | None:
    created = False
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid():
            raise ValueError(f"notification storage directory is unsafe: {name}")
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            value = os.fstat(descriptor)
        elif stat.S_IMODE(value.st_mode) != 0o700:
            raise ValueError(f"notification storage directory is unsafe: {name}")
        if stat.S_IMODE(value.st_mode) != 0o700:
            raise ValueError(f"notification storage directory is unsafe: {name}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _private_credential_stat(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise ValueError("credential file must be a regular owner-only file.")


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("credential write did not complete.")
        remaining = remaining[written:]


def _remove_partial_file(directory_fd: int, name: str, witness: os.stat_result) -> None:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (witness.st_dev, witness.st_ino):
            os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except OSError:
        pass


def stage_webhook(runtime_root: Path, webhook: str) -> str:
    """Durably stage a validated webhook under a fresh immutable identifier."""
    validated_webhook = _validate_webhook(webhook)
    root = _runtime_root_path(runtime_root)
    with ExitStack() as stack:
        root_fd = _open_runtime_root(root)
        stack.callback(os.close, root_fd)
        notifications_fd = _open_private_directory(root_fd, "notifications", create=True)
        if notifications_fd is None:
            raise ValueError("notification storage directory is unavailable.")
        stack.callback(os.close, notifications_fd)
        credentials_fd = _open_private_directory(notifications_fd, "credentials", create=True)
        if credentials_fd is None:
            raise ValueError("credential storage directory is unavailable.")
        stack.callback(os.close, credentials_fd)

        for _ in range(32):
            credential_id = secrets.token_hex(16)
            filename = f"{credential_id}.json"
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "credential_id": credential_id,
                "created_at": utc_now(),
                "webhook": validated_webhook,
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            if len(encoded) > _MAX_CREDENTIAL_BYTES:
                raise ValueError("webhook credential record is too large.")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            try:
                file_fd = os.open(filename, flags, 0o600, dir_fd=credentials_fd)
            except FileExistsError:
                continue
            try:
                os.fchmod(file_fd, 0o600)
                file_stat = os.fstat(file_fd)
                _private_credential_stat(file_stat)
                _write_all(file_fd, encoded)
                os.fsync(file_fd)
                os.fsync(credentials_fd)
                final_stat = os.fstat(file_fd)
                _private_credential_stat(final_stat)
                published_stat = os.stat(filename, dir_fd=credentials_fd, follow_symlinks=False)
                if (published_stat.st_dev, published_stat.st_ino) != (final_stat.st_dev, final_stat.st_ino):
                    raise OSError("staged credential could not be verified.")
            except BaseException:
                try:
                    file_stat = os.fstat(file_fd)
                    _remove_partial_file(credentials_fd, filename, file_stat)
                except OSError:
                    pass
                raise
            finally:
                os.close(file_fd)
            return credential_id
    raise FileExistsError("could not allocate a unique notification credential ID.")


def _read_all(descriptor: int) -> bytes:
    content = bytearray()
    while len(content) <= _MAX_CREDENTIAL_BYTES:
        chunk = os.read(descriptor, min(8192, _MAX_CREDENTIAL_BYTES + 1 - len(content)))
        if not chunk:
            break
        content.extend(chunk)
    if len(content) > _MAX_CREDENTIAL_BYTES:
        raise ValueError("credential record is too large.")
    return bytes(content)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _decode_webhook_record(encoded: bytes, credential_id: str) -> str:
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise ValueError("credential record is invalid.") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "credential_id", "created_at", "webhook"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != _SCHEMA_VERSION
        or value["credential_id"] != credential_id
        or not isinstance(value["webhook"], str)
    ):
        raise ValueError("credential record is invalid.")
    try:
        _validate_created_at(value["created_at"])
        return _validate_webhook(value["webhook"])
    except ValueError:
        raise ValueError("credential record is invalid.") from None


def read_webhook(runtime_root: Path, credential_id: str) -> str:
    """Read one credential through no-follow descriptors after checking privacy."""
    root = _runtime_root_path(runtime_root)
    identifier = _validate_credential_id(credential_id)
    with ExitStack() as stack:
        root_fd = _open_runtime_root(root)
        stack.callback(os.close, root_fd)
        notifications_fd = _open_private_directory(root_fd, "notifications", create=False)
        if notifications_fd is None:
            raise FileNotFoundError("notification storage directory does not exist.")
        stack.callback(os.close, notifications_fd)
        credentials_fd = _open_private_directory(notifications_fd, "credentials", create=False)
        if credentials_fd is None:
            raise FileNotFoundError("credential storage directory does not exist.")
        stack.callback(os.close, credentials_fd)

        filename = f"{identifier}.json"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(filename, flags, dir_fd=credentials_fd)
        try:
            initial_stat = os.fstat(descriptor)
            _private_credential_stat(initial_stat)
            if initial_stat.st_size > _MAX_CREDENTIAL_BYTES:
                raise ValueError("credential record is too large.")
            encoded = _read_all(descriptor)
            final_stat = os.fstat(descriptor)
            _private_credential_stat(final_stat)
            initial_witness = (
                initial_stat.st_dev,
                initial_stat.st_ino,
                initial_stat.st_size,
                initial_stat.st_mtime_ns,
                initial_stat.st_ctime_ns,
            )
            final_witness = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            )
            if initial_witness != final_witness:
                raise ValueError("credential file changed while reading.")
            try:
                published_stat = os.stat(filename, dir_fd=credentials_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise ValueError("credential file changed while reading.") from None
            if (published_stat.st_dev, published_stat.st_ino) != (final_stat.st_dev, final_stat.st_ino):
                raise ValueError("credential file changed while reading.")
        finally:
            os.close(descriptor)
    return _decode_webhook_record(encoded, identifier)
