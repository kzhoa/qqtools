"""Bounded Linux directory-entry capture for durable directory cursors."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


class _Dirent(ctypes.Structure):
    """Linux dirent layout used by the durable directory cursor."""

    _fields_ = [
        ("d_ino", ctypes.c_ulong),
        ("d_off", ctypes.c_long),
        ("d_reclen", ctypes.c_ushort),
        ("d_type", ctypes.c_ubyte),
        ("d_name", ctypes.c_char * 256),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.fdopendir.argtypes = [ctypes.c_int]
_LIBC.fdopendir.restype = ctypes.c_void_p
_LIBC.closedir.argtypes = [ctypes.c_void_p]
_LIBC.closedir.restype = ctypes.c_int
_LIBC.readdir.argtypes = [ctypes.c_void_p]
_LIBC.readdir.restype = ctypes.POINTER(_Dirent)
_LIBC.seekdir.argtypes = [ctypes.c_void_p, ctypes.c_long]
_LIBC.seekdir.restype = None
_LIBC.telldir.argtypes = [ctypes.c_void_p]
_LIBC.telldir.restype = ctypes.c_long

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_DOT_ENTRY_LIMIT = 2


def read_directory_entry(path: Path, offset: int) -> tuple[str | None, int]:
    """Read one directory entry after ``offset`` without following symlinks.

    The returned offset is the ``telldir`` cookie after the returned entry, or
    the end-of-directory cookie when no entry remains. Every invocation owns
    and closes its directory stream so persisted cursors never retain process
    resources across maintenance slices.
    """

    if type(offset) is not int or not 0 <= offset <= (1 << 63) - 1:
        raise ValueError("directory cursor offset must be a non-negative integer")
    descriptor = -1
    stream: int | None = None
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
        stream = _LIBC.fdopendir(descriptor)
        if not stream:
            stream = None
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), os.fspath(path))
        # fdopendir owns the descriptor after success; closedir must release it.
        descriptor = -1
        if offset:
            _LIBC.seekdir(stream, ctypes.c_long(offset))
        skipped = 0
        while True:
            ctypes.set_errno(0)
            item = _LIBC.readdir(stream)
            if not item:
                error = ctypes.get_errno()
                if error:
                    raise OSError(error, os.strerror(error), os.fspath(path))
                return None, int(_LIBC.telldir(stream))
            raw_name = bytes(item.contents.d_name).split(b"\0", 1)[0]
            try:
                name = os.fsdecode(raw_name)
            except UnicodeDecodeError as exc:
                raise ValueError("directory entry name is not valid filesystem text") from exc
            new_offset = int(_LIBC.telldir(stream))
            if name in {".", ".."}:
                skipped += 1
                if skipped > _DOT_ENTRY_LIMIT:
                    raise RuntimeError("directory returned more than two dot entries")
                continue
            if not name or "/" in name or "\x00" in name:
                raise ValueError("directory entry name is invalid")
            return name, new_offset
    finally:
        if stream is not None:
            _LIBC.closedir(stream)
        elif descriptor >= 0:
            os.close(descriptor)


__all__ = ["read_directory_entry"]
