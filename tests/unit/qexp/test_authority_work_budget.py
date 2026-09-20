"""Active service reservations are independent of optional discovery readiness."""

from collections import Counter
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.authority_work import AuthorityWork
from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan


@pytest.mark.parametrize("limit", [1, 2, 63, 64, 256])
@pytest.mark.parametrize("has_responsibility_error", [False, True])
def test_ready_responsibilities_preserve_half_the_active_budget(tmp_path, monkeypatch, limit, has_responsibility_error):
    supervisor = SimpleNamespace(
        recovery_owner=None,
        cfg=SimpleNamespace(runtime_root=tmp_path / "runtime", shared_root=tmp_path / "shared"),
        metrics={},
        renewal_interval_seconds=1.0,
        _record_diagnostic=lambda *_args: None,
    )
    work = AuthorityWork(supervisor)
    counts = Counter()

    def record(name):
        counts[name] += 1

    def responsibility():
        record("responsibilities")
        if has_responsibility_error:
            raise OSError("injected unavailable membership")
        return True

    def take(_scan, _limit, **_kwargs):
        return SimpleNamespace(paths=(tmp_path / "candidate",), entries_visited=1, is_complete=False)

    monkeypatch.setattr(EvidenceScan, "take", take)
    monkeypatch.setattr(work, "_active_step", lambda: record("active"))
    monkeypatch.setattr(work, "_cleanup_step", lambda: record("cleanup"))
    monkeypatch.setattr(work, "_responsibility_step", responsibility)
    for name, lane in work._lanes.items():
        monkeypatch.setattr(lane, "operation", lambda _path, name=name: record(name))
    work._remember("cached-attempt")
    try:
        # Eight ordinary slices must retain the 256 active service opportunities,
        # even when discovery contributes no duplicate service of those members.
        for _ in range(512):
            before = sum(counts.values())
            work.tick(limit)
            assert sum(counts.values()) - before <= limit
        assert counts["active"] == 512 * limit // 2
        assert counts["responsibilities"] > 0
        assert all(counts[name] > 0 for name in (*work._lanes, "cleanup"))
        if has_responsibility_error:
            assert work.snapshot()["responsibility_discovery_failures"] == counts["responsibilities"]
    finally:
        work.close()
