"""Durable record publication includes newly introduced directory links."""

import os
import stat
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import coverage as coverage_module
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.service import publish_submission_debt
from tests.helpers.qexp_discovery import confirmed_source, isolated_group

pytestmark = pytest.mark.integration


def record_directory_syncs(monkeypatch):
    synced = set()
    original = os.fsync

    def observe(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.add(Path(os.readlink(f"/proc/self/fd/{fd}")))
        return original(fd)

    monkeypatch.setattr(os, "fsync", observe)
    return synced


def test_first_debt_durably_links_ancestors_before_return(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    synced = record_directory_syncs(monkeypatch)
    path = publish_submission_debt(cfg.shared_root, "experiment", "new-batch")
    assert {path.parent, path.parent.parent, path.parent.parent.parent, cfg.shared_root / "operations"} <= synced


def test_member_and_cursor_directory_links_precede_cursor_ack(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    synced = record_directory_syncs(monkeypatch)
    original = coverage_module.atomic_replace
    acknowledged = False

    def observe(path, record):
        nonlocal acknowledged
        if path.parent.name == "publications":
            assert {
                coverage.directory,
                coverage.directory.parent,
                coverage.directory.parent.parent,
            } <= synced
            acknowledged = True
        return original(path, record)

    monkeypatch.setattr(coverage_module, "atomic_replace", observe)
    assert coverage.publish(source).state == "complete"
    assert acknowledged
