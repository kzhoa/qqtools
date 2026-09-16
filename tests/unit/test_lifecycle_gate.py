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


def test_lifecycle_gate_requires_combined_exit_result_workload():
    gate, session = _gate()
    session.items = [item for item in session.items if "li02" not in item.name]
    with pytest.raises(pytest.UsageError, match="preserves_exit_results"):
        gate.pytest_collection_finish(session)


def test_lifecycle_gate_selects_full_matrix():
    gate = _LifecycleGate(SimpleNamespace(getoption=lambda _name: "full"))
    items = [
        SimpleNamespace(
            path=Path("test_agent_lifecycle_independence.py"),
            name=name,
            nodeid=f"lifecycle::{name}",
        )
        for name in sorted(gate.full_names)
    ]
    session = SimpleNamespace(items=items, exitstatus=0)

    gate.pytest_collection_finish(session)

    assert gate.required == {item.nodeid for item in items}


def test_lifecycle_gate_collects_required_nodes_from_xdist_worker():
    gate = _LifecycleGate(SimpleNamespace(getoption=lambda _name: "full"))
    nodeids = [
        f"tests/integration/qexp/test_agent_lifecycle_independence.py::{name}" for name in sorted(gate.full_names)
    ]

    gate.pytest_xdist_node_collection_finished(SimpleNamespace(), nodeids)

    assert gate.required == set(nodeids)


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


def test_lifecycle_gate_leaves_xdist_worker_exit_status_to_controller():
    gate, session = _gate()
    session.config = SimpleNamespace(workerinput={})
    session.exitstatus = 0

    gate.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 0
