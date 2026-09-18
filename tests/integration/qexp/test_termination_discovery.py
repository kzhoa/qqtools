"""Recovery termination discovery keeps its lock fence and streams positive results."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import termination
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("commitment", [None, "signal_committed", "committed", "unavailable"])
def test_recovery_inventory_stops_at_first_commitment_and_closes(tmp_path, monkeypatch, commitment):
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "probe", tmp_path / "runtime")
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / "attempt"
    for index in range(65):
        decision = {"state": "pending", "shared_commitment": "pending"}
        if index == 0 and commitment is not None:
            field = "state" if commitment == "signal_committed" else "shared_commitment"
            decision[field] = commitment
        atomic_replace(directory / f"{index:03}.json", {"termination_decision": decision})
    with os.scandir(directory) as real_entries:
        entries = sorted(real_entries, key=lambda entry: entry.name)

    class Inventory:
        def __init__(self):
            self.iterator = iter(entries)
            self.visited = 0
            self.is_closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.is_closed = True

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.iterator)
            self.visited += 1
            return entry

    inventory = Inventory()
    monkeypatch.setattr(termination, "os", SimpleNamespace(scandir=lambda _path: inventory))
    monkeypatch.setattr(termination, "list_decisions", lambda *_args: pytest.fail("eager recovery inventory"))
    diagnostics = RuntimeDiagnostics()
    with termination.attempt_control_lock(cfg, "attempt"), activate_diagnostics(diagnostics):
        assert termination.is_recovery_blocked(cfg, "attempt") == (commitment is not None)
    expected = 65 if commitment is None else 1
    assert inventory.visited == expected
    assert inventory.is_closed
    assert diagnostics.counters["store.inventory_entries"] == expected
    assert diagnostics.counters["store.read_json.calls"] == expected


@pytest.mark.parametrize("invalid", ["invalid-json", {"termination_decision": []}, {"termination_decision": None}])
def test_recovery_inventory_closes_on_malformed_record(tmp_path, monkeypatch, invalid):
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "probe", tmp_path / "runtime")
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / "attempt"
    directory.mkdir(parents=True)
    if invalid == "invalid-json":
        (directory / "broken.json").write_text("invalid")
    else:
        atomic_replace(directory / "broken.json", invalid)
    captured = []

    def scan(path):
        entries = os.scandir(path)
        captured.append(entries)
        return entries

    monkeypatch.setattr(termination, "os", SimpleNamespace(scandir=scan))
    with termination.attempt_control_lock(cfg, "attempt"), pytest.raises(ValueError):
        termination.is_recovery_blocked(cfg, "attempt")
    assert list(captured[0]) == []


def test_missing_recovery_inventory_is_empty(tmp_path):
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "probe", tmp_path / "runtime")
    with termination.attempt_control_lock(cfg, "attempt"):
        assert not termination.is_recovery_blocked(cfg, "attempt")


def test_recovery_inventory_ignores_nonregular_entries(tmp_path):
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "probe", tmp_path / "runtime")
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / "attempt"
    directory.mkdir(parents=True)
    target = tmp_path / "other.json"
    atomic_replace(target, {"termination_decision": {"state": "confirmed"}})
    (directory / "linked.json").symlink_to(target)
    (directory / "nested.json").mkdir()
    (directory / "unrelated.txt").write_text("invalid")
    with termination.attempt_control_lock(cfg, "attempt"):
        assert not termination.is_recovery_blocked(cfg, "attempt")
