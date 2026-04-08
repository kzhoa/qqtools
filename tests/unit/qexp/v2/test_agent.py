from __future__ import annotations

import pytest

from qqtools.plugins.qexp.v2.agent import (
    get_agent_status,
    is_agent_running,
    read_agent_state,
    start_agent_record,
    stop_agent_record,
    write_heartbeat,
    IDLE_TIMEOUT_DEFAULT,
)
from qqtools.plugins.qexp.v2.layout import init_shared_root
from qqtools.plugins.qexp.v2.models import AGENT_STATE_ACTIVE, AGENT_STATE_STOPPED


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1")


class TestAgentState:
    def test_no_state_file(self, cfg):
        assert read_agent_state(cfg) is None
        assert not is_agent_running(cfg)

    def test_start_writes_state(self, cfg):
        start_agent_record(cfg, persistent=False)
        state = read_agent_state(cfg)
        assert state is not None
        assert state["agent_state"] == AGENT_STATE_ACTIVE
        assert state["pid"] is not None
        assert state["idle_timeout_seconds"] == IDLE_TIMEOUT_DEFAULT

    def test_persistent_flag(self, cfg):
        start_agent_record(cfg, persistent=True)
        state = read_agent_state(cfg)
        assert state["idle_timeout_seconds"] == 0
        assert state["persistent"] is True

    def test_stop_clears_pid(self, cfg):
        start_agent_record(cfg)
        stop_agent_record(cfg, reason="test")
        state = read_agent_state(cfg)
        assert state["agent_state"] == AGENT_STATE_STOPPED
        assert state["pid"] is None
        assert state["last_exit_reason"] == "test"

    def test_heartbeat(self, cfg):
        start_agent_record(cfg)
        old_hb = read_agent_state(cfg)["last_heartbeat"]
        write_heartbeat(cfg)
        new_hb = read_agent_state(cfg)["last_heartbeat"]
        assert new_hb >= old_hb


class TestGetAgentStatus:
    def test_no_agent(self, cfg):
        status = get_agent_status(cfg)
        assert status["agent_state"] == AGENT_STATE_STOPPED
        assert not status["is_running"]

    def test_running_agent(self, cfg):
        start_agent_record(cfg)
        status = get_agent_status(cfg)
        assert status["is_running"]

    def test_stale_agent(self, cfg):
        start_agent_record(cfg)
        # Fake a dead PID
        from qqtools.plugins.qexp.v2.storage import write_atomic_json
        from qqtools.plugins.qexp.v2.layout import agent_state_path
        state = read_agent_state(cfg)
        state["pid"] = 99999999
        write_atomic_json(agent_state_path(cfg), state)
        status = get_agent_status(cfg)
        assert not status["is_running"]
        assert status["agent_state"] == "stale"
