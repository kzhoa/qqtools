from __future__ import annotations

import pytest

from qqtools.plugins.qexp.v2.events import query_events, write_event
from qqtools.plugins.qexp.v2.layout import init_shared_root


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1", runtime_root=tmp_path / "runtime")


class TestWriteEvent:
    def test_basic_write(self, cfg):
        eid = write_event(cfg, "submit_succeeded", task_id="t1")
        assert len(eid) == 12

    def test_global_and_machine_dirs(self, cfg):
        write_event(cfg, "task_started", task_id="t1")
        from qqtools.plugins.qexp.v2.layout import global_events_dir, machine_events_dir
        g = global_events_dir(cfg)
        m = machine_events_dir(cfg)
        assert any(g.rglob("*.json"))
        assert any(m.rglob("*.json"))

    def test_with_details(self, cfg):
        write_event(cfg, "task_failed", task_id="t1", details={"exit_code": 1})
        events = query_events(cfg, event_type="task_failed")
        assert len(events) == 1
        assert events[0]["details"]["exit_code"] == 1


class TestQueryEvents:
    def test_empty(self, cfg):
        assert query_events(cfg) == []

    def test_filter_by_type(self, cfg):
        write_event(cfg, "submit_succeeded", task_id="t1")
        write_event(cfg, "task_failed", task_id="t2")
        events = query_events(cfg, event_type="submit_succeeded")
        assert len(events) == 1
        assert events[0]["task_id"] == "t1"

    def test_filter_by_machine(self, cfg):
        write_event(cfg, "task_started", task_id="t1")
        events = query_events(cfg, machine="dev1")
        assert len(events) == 1

    def test_limit(self, cfg):
        for i in range(10):
            write_event(cfg, "submit_succeeded", task_id=f"t{i}")
        events = query_events(cfg, limit=3)
        assert len(events) == 3

    def test_nonexistent_machine(self, cfg):
        events = query_events(cfg, machine="nosuch")
        assert events == []
