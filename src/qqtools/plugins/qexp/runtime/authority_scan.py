"""Resumable, advisory discovery of machine-local authority evidence."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidencePage:
    paths: tuple[Path, ...]
    entries_visited: int
    is_complete: bool


class EvidenceScan:
    """Visit a bounded number of directory entries without constructing an inventory.

    Each scan belongs to one supervisor thread. EOF closes the directory; the next
    call starts a new sweep, discovering missed notifications and concurrent writes.
    Restart discards the cursor. Directory replacement restarts discovery, but
    additions never restart an in-progress sweep and therefore cannot starve its tail.
    Paths are hints: consumers must reload evidence and validate write authority.
    """

    def __init__(self, directory: Path, *, directories: bool = False) -> None:
        self.directory = directory
        self._is_directory_scan = directories
        self._entries = None
        self._identity: tuple[int, int] | None = None
        self._slice_token: object | None = None

    def close(self) -> None:
        """Discard advisory discovery state and release the directory handle."""
        if self._entries is not None:
            self._entries.close()
        self._entries = None
        self._identity = None
        self._slice_token = None

    def take(self, limit: int, *, slice_token: object | None = None) -> EvidencePage:
        """Count every entry; an optional slice token shares one identity check.

        Use a fresh token for every bounded caller slice. Missing directories and
        EOF remain empty until the next token; records are always advisory hints.
        Without a token, every call checks identity and can start another sweep.
        """
        if type(limit) is not int or limit <= 0:
            raise ValueError("evidence discovery limit must be a positive integer")
        paths: list[Path] = []
        visited = 0
        try:
            if slice_token is None or slice_token is not self._slice_token:
                try:
                    metadata = self.directory.stat(follow_symlinks=False)
                except FileNotFoundError:
                    self.close()
                    self._slice_token = slice_token
                    return EvidencePage((), 0, True)
                if not stat.S_ISDIR(metadata.st_mode):
                    self.close()
                    raise NotADirectoryError(str(self.directory))
                identity = (metadata.st_dev, metadata.st_ino)
                if identity != self._identity:
                    self.close()
                    self._entries = os.scandir(self.directory)
                    self._identity = identity
                self._slice_token = slice_token
            if self._entries is None:
                return EvidencePage((), 0, True)
            for _ in range(limit):
                try:
                    entry = next(self._entries)
                except StopIteration:
                    self.close()
                    self._slice_token = slice_token
                    return EvidencePage(tuple(paths), visited, True)
                visited += 1
                if self._is_directory_scan:
                    is_selected = entry.is_dir(follow_symlinks=False)
                else:
                    is_selected = entry.name.endswith(".json") and entry.is_file(follow_symlinks=False)
                if is_selected:
                    paths.append(self.directory / entry.name)
            return EvidencePage(tuple(paths), visited, False)
        except OSError:
            self.close()
            raise
