"""Bounded replay and empty-directory verification for pending claim archives."""

from __future__ import annotations

import errno
from pathlib import Path

from ..config_types import RootConfig
from .authority_scan import EvidenceScan
from .paths import shared_paths
from .store import CASConflict, create_if_absent, read_json


def is_pending_archive_clear(cfg: RootConfig, task_id: str) -> bool:
    """Verify pending emptiness atomically; scan completion alone is insufficient."""
    try:
        (shared_paths(cfg.shared_root)["claim_pending"] / task_id).rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            return False
        raise
    return True


class ClaimArchiveScan:
    """Replay a page, then verify emptiness with atomic directory removal."""

    def __init__(self, cfg: RootConfig, task_id: str) -> None:
        self.cfg = cfg
        self.task_id = task_id
        self.paths = shared_paths(cfg.shared_root)
        self.scan = EvidenceScan(self.paths["claim_pending"] / task_id)
        self.should_rotate = False

    def close(self) -> None:
        self.scan.close()

    def step(self, limit: int = 8) -> bool:
        """Replay at most limit entries; retained entries prevent cleanup completion."""
        self.should_rotate = False
        try:
            page = self.scan.take(limit)
        except OSError:
            # A failed directory must relinquish project discovery to other Tasks.
            self.should_rotate = True
            raise
        self.should_rotate = page.is_complete
        failure = None
        for path in page.paths:
            try:
                self._replay(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
        if not page.is_complete:
            return False
        return is_pending_archive_clear(self.cfg, self.task_id)

    def _replay(self, path: Path) -> None:
        record = read_json(path)
        archive = record.get("claim_archive")
        if not isinstance(archive, dict):
            raise ValueError("pending claim archive identity is unreadable")
        token = archive.get("fencing_token")
        if archive.get("task_id") != self.task_id or type(token) is not int or path.name != f"{token}.json":
            raise ValueError("pending claim archive identity does not match its path")
        destination = self.paths["claim_archive"] / self.task_id / path.name
        try:
            create_if_absent(destination, record)
        except CASConflict:
            pass
        path.unlink(missing_ok=True)


class ClaimArchiveDiscovery:
    """Retain one outer cursor and one Task scan, without per-Task cache eviction."""

    def __init__(self, cfg: RootConfig) -> None:
        self.cfg = cfg
        self.scan = EvidenceScan(shared_paths(cfg.shared_root)["claim_pending"], directories=True)
        self.current: ClaimArchiveScan | None = None

    @property
    def cursor_count(self) -> int:
        return 1 + int(self.current is not None)

    def close(self) -> None:
        self.scan.close()
        if self.current is not None:
            self.current.close()
            self.current = None

    def step(self, limit: int = 8) -> None:
        """Visit one outer entry or at most limit records in the selected Task."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("archive discovery limit must be a positive integer")
        if self.current is None:
            page = self.scan.take(1)
            if page.paths:
                self.current = ClaimArchiveScan(self.cfg, page.paths[0].name)
            return
        try:
            self.current.step(limit)
        finally:
            if self.current.should_rotate:
                self.current.close()
                self.current = None
