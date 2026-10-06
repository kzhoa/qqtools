"""Directory budgets and durable home turns prevent busy-home starvation."""

import importlib
import math
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.runtime.availability import offer_deadline_scan as scan
from qqtools.plugins.qexp.runtime.paths import local_paths, shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def scanner(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "helper", runtime_root=tmp_path / "runtime")
    root = shared_paths(cfg.shared_root)["offer_deadlines_active"]
    root.mkdir(parents=True, exist_ok=True)
    return cfg, root, datetime.now(timezone.utc).strftime("%Y%m%d%H")


def _entry(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _ordered_reader(monkeypatch):
    """Stable fixture cookies make numerical fairness assertions deterministic."""
    calls = []
    directories = {}

    def read(path, offset):
        calls.append(path)
        metadata = path.stat()
        identity = (metadata.st_dev, metadata.st_ino)
        existing_identity, names = directories.get(path, (None, []))
        if existing_identity != identity:
            names = []
        # Retain tombstone positions, like directory cookies, when another
        # participant removes an entry during a scan.
        names.extend(sorted({child.name for child in path.iterdir()} - set(names)))
        directories[path] = (identity, names)
        if offset >= len(names):
            return None, offset
        return names[offset], offset + 1

    monkeypatch.setattr(scan, "read_directory_entry", read)
    return calls


def test_total_budget_includes_home_future_bucket_irrelevant_entry_and_eof(scanner, monkeypatch):
    cfg, root, bucket = scanner
    for home in range(40):
        for index in range(20):
            _entry(root / f"home-{home:03d}" / "9999010100" / f"{index}.json")
            (root / f"home-{home:03d}" / f"irrelevant-{index}").mkdir()
    calls = _ordered_reader(monkeypatch)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == []
    assert 63 <= len(calls) <= 64
    by_home = Counter(path.relative_to(root).parts[0] for path in calls if path != root)
    assert max(by_home.values()) <= 8
    assert len(by_home) > 1


@pytest.mark.parametrize("reload_module", [False, True])
def test_continued_busy_home_appends_cannot_starve_later_due_home(scanner, monkeypatch, reload_module):
    cfg, root, bucket = scanner
    home_count = 19
    busy = root / "home-000" / bucket
    for index in range(40):
        _entry(busy / f"noise-{index:05d}")
    for index in range(1, home_count - 1):
        (root / f"home-{index:03d}" / "9999010100").mkdir(parents=True)
    target = _entry(root / f"home-{home_count - 1:03d}" / bucket / "target.json")
    # Every home costs <= 1 root read + 8 home reads. Include one root EOF
    # and a fresh target turn after a possible budget-truncated selection.
    cycle_bound = 2 * math.ceil((home_count * (1 + 8) + 1) / (64 - 1)) + 2
    found = []
    for cycle in range(cycle_bound):
        for index in range(40):
            _entry(busy / f"noise-{(cycle + 1) * 40 + index:05d}")
        if reload_module:
            importlib.reload(scan)
        calls = _ordered_reader(monkeypatch)
        found.extend(scan.iter_due_deadline_paths(replace(cfg), limit=1))
        assert len(calls) <= 64
        if target in found:
            break
    assert target in found, f"due home starved beyond budget-derived {cycle_bound} calls"


def test_quota_and_candidate_rotate_homes_and_preserve_each_home_progress(scanner, monkeypatch):
    cfg, root, bucket = scanner
    for index in range(10):
        _entry(root / "a" / bucket / f"{index:02d}-noise")
    a = _entry(root / "a" / bucket / "99-due.json")
    b1 = _entry(root / "b" / bucket / "01-due.json")
    b2 = _entry(root / "b" / bucket / "02-due.json")
    calls = _ordered_reader(monkeypatch)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [b1]
    assert sum(path == root / "a" / bucket for path in calls) == 7
    # The candidate yield checkpoint already points past b even when the
    # generator was closed immediately, as advance_due_offer does.
    iterator = scan.iter_due_deadline_paths(cfg, limit=1)
    assert next(iterator) == a
    iterator.close()
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [b2]


def test_home_selected_on_final_budget_read_is_deferred_instead_of_starved(scanner, monkeypatch):
    cfg, root, bucket = scanner
    for home in range(7):
        for entry in range(24):
            (root / f"home-{home}" / f"999901{entry:04d}").mkdir(parents=True)
    target = _entry(root / "home-7" / bucket / "due.json")
    calls = _ordered_reader(monkeypatch)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == []
    assert len(calls) <= 64
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [target]


def test_missing_selected_bucket_and_replaced_home_restart_safely(scanner, monkeypatch):
    cfg, root, bucket = scanner
    first = _entry(root / "a" / bucket / "01.json")
    _ordered_reader(monkeypatch)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [first]
    # Rename keeps the old inode alive, making replacement identity distinct.
    (root / "a").rename(root.parent / "old-home")
    second = _entry(root / "a" / bucket / "02.json")
    found = []
    for _ in range(3):
        found.extend(scan.iter_due_deadline_paths(cfg, limit=1))
    assert second in found
    second.unlink()
    second.parent.rmdir()
    third = _entry(root / "a" / "2000010100" / "03.json")
    found = []
    for _ in range(3):
        found.extend(scan.iter_due_deadline_paths(cfg, limit=1))
    assert third in found


@pytest.mark.parametrize("level", ["home", "bucket", "entry"])
def test_scanner_rejects_symlinks_at_every_level(scanner, tmp_path, monkeypatch, level):
    cfg, root, bucket = scanner
    target = tmp_path / "outside"
    target.mkdir()
    if level == "home":
        (root / "a").symlink_to(target, target_is_directory=True)
    elif level == "bucket":
        (root / "a").mkdir()
        (root / "a" / bucket).symlink_to(target, target_is_directory=True)
    else:
        path = root / "a" / bucket
        path.mkdir(parents=True)
        (path / "task.json").symlink_to(target / "task.json")
    _ordered_reader(monkeypatch)
    with pytest.raises(OSError, match="symlink"):
        list(scan.iter_due_deadline_paths(cfg))


def test_vanished_entry_and_directory_do_not_prevent_wraparound_discovery(scanner, monkeypatch):
    cfg, root, bucket = scanner
    vanished = _entry(root / "a" / bucket / "01.json")
    target = _entry(root / "b" / bucket / "target.json")
    calls = _ordered_reader(monkeypatch)
    real_read = scan.read_directory_entry

    def disappear(path, offset):
        result = real_read(path, offset)
        if path == vanished.parent and result[0] == vanished.name:
            vanished.unlink()
            vanished.parent.rmdir()
            vanished.parent.parent.rmdir()
        return result

    monkeypatch.setattr(scan, "read_directory_entry", disappear)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [target]
    assert len(calls) <= 64
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [target]


@pytest.mark.parametrize("obsolete", [1, 2, 99, "corrupt"])
def test_obsolete_cursor_restarts_bounded_without_rebuilding_indexes(scanner, monkeypatch, obsolete):
    # QQTOOLS-COMPAT-0020: old lexical/home-only directory checkpoints reset.
    cfg, root, bucket = scanner
    target = _entry(root / "other-home" / bucket / "due.json")
    cursor = local_paths(cfg.runtime_root)["maintenance_cursors"] / "offer_deadlines.json"
    atomic_replace(cursor, {"offer_deadline_cursor": {"version": obsolete, "bucket_offset": 999999}})
    calls = _ordered_reader(monkeypatch)
    assert list(scan.iter_due_deadline_paths(cfg, limit=1)) == [target]
    assert len(calls) == 3
    assert read_json(cursor)["offer_deadline_cursor"]["version"] == 3
    assert target.exists()


def test_real_directory_cookies_resume_records_across_calls(scanner, monkeypatch):
    cfg, root, bucket = scanner
    expected = {_entry(root / "other-home" / bucket / f"task-{index:03d}.json") for index in range(17)}
    real_read = scan.read_directory_entry
    calls = []

    def counted(*args):
        calls.append(args[0])
        return real_read(*args)

    monkeypatch.setattr(scan, "read_directory_entry", counted)
    found = set()
    # One candidate and one root wrap per cycle, plus the final discovery call.
    for _ in range(2 * len(expected) + 1):
        calls.clear()
        found.update(scan.iter_due_deadline_paths(cfg, limit=1))
        assert len(calls) <= 64
        if found == expected:
            break
    assert found == expected


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_record_limit_rejected(scanner, limit):
    cfg, _, _ = scanner
    with pytest.raises(ValueError, match="positive"):
        list(scan.iter_due_deadline_paths(cfg, limit=limit))
