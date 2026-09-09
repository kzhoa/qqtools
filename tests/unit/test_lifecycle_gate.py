from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import _LifecycleGate


def _gate():
    gate = _LifecycleGate(SimpleNamespace(getoption=lambda _name: "representative"))
    items = [
        SimpleNamespace(
            path=Path("test_agent_lifecycle_independence.py"),
            name=name,
            nodeid=f"lifecycle::{name}",
        )
        for name in sorted(gate.representative_names)
    ]
    session = SimpleNamespace(items=items, exitstatus=0)
    return gate, session


def test_lifecycle_gate_rejects_missing_required_case():
    gate, session = _gate()
    session.items.pop()
    with pytest.raises(pytest.UsageError, match="test_li04"):
        gate.pytest_collection_finish(session)


def test_lifecycle_gate_rejects_missing_failure_variant():
    gate, session = _gate()
    session.items = [item for item in session.items if "7-failed" not in item.name]
    with pytest.raises(pytest.UsageError, match="7-failed"):
        gate.pytest_collection_finish(session)


@pytest.mark.parametrize("failure", ["skip", "teardown", "not_run"])
def test_lifecycle_gate_rejects_incomplete_execution(failure):
    gate, session = _gate()
    gate.pytest_collection_finish(session)
    for item in session.items:
        gate.pytest_runtest_logreport(
            SimpleNamespace(nodeid=item.nodeid, when="call", passed=True, skipped=False, failed=False)
        )
    nodeid = session.items[0].nodeid
    if failure == "not_run":
        gate.passed.remove(nodeid)
    else:
        gate.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid=nodeid,
                when="teardown",
                passed=False,
                skipped=failure == "skip",
                failed=failure == "teardown",
            )
        )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_lifecycle_gate_accepts_complete_execution():
    gate, session = _gate()
    gate.pytest_collection_finish(session)
    for item in session.items:
        gate.pytest_runtest_logreport(
            SimpleNamespace(nodeid=item.nodeid, when="call", passed=True, skipped=False, failed=False)
        )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_lifecycle_gate_rejects_exceeded_budget(monkeypatch):
    gate, session = _gate()
    gate.pytest_collection_finish(session)
    gate.passed = set(gate.required)
    session.config = SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda _name: None))
    monkeypatch.setattr("tests.conftest.time.monotonic", lambda: gate.started_at + gate.budget_seconds + 1)
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
