"""Machine discovery time-slices Groups without discarding ongoing source state."""

from dataclasses import replace

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.group_discovery import service as module
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


class IterationStop:
    def __init__(self, limit=350, callback=None):
        self.turn = 0
        self.limit = limit
        self.callback = callback

    def is_set(self):
        return self.turn >= self.limit

    def wait(self, seconds):
        self.turn += 1
        if self.callback:
            self.callback(self.turn)
        return self.is_set()


def setup_worker(tmp_path, monkeypatch, *, changing_registry=False, callback=None):
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance

    cfg = isolated_group(tmp_path, tail=0)
    create_group(cfg, "second")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    _, bindings = runtime.load_registry()
    binding = bindings[0]
    stop = IterationStop(callback=callback)
    monkeypatch.setattr(runtime, "load_registry", lambda: (stop.turn if changing_registry else 1, [binding]))
    monkeypatch.setattr(runtime, "binding_state", lambda _: "enabled")
    monkeypatch.setattr(module.time, "monotonic", lambda: float(stop.turn))
    instances = []
    calls = []

    class Service:
        def __init__(self, root, group):
            self.root = root
            self.group = group
            self.is_closed = False
            self.steps = 0
            instances.append(self)

        def advance(self):
            self.steps += 1
            calls.append((stop.turn, self.group, "discovery", self.steps))
            # The first Group never finishes; the other must still receive slices.
            return {"state": "progressed", "reason": None}

        def request_close(self):
            self.is_closed = True

    class Maintenance:
        def __init__(self, root, group):
            self.group = group
            self.is_closed = False

        def advance(self):
            calls.append((stop.turn, self.group, "maintenance", 0))
            return {"state": "complete", "reason": None}

        def request_close(self):
            self.is_closed = True

    monkeypatch.setattr(module, "GroupDiscoveryService", Service)
    monkeypatch.setattr(maintenance, "GroupMaintenance", Maintenance)
    worker = module.MachineGroupDiscoveryWorker(runtime)
    worker._stop = stop
    return cfg, worker, instances, calls, runtime, binding


@pytest.mark.parametrize("changing_registry", [False, True])
def test_groups_and_maintenance_progress_while_first_group_never_finishes(tmp_path, monkeypatch, changing_registry):
    _, worker, instances, calls, _, _ = setup_worker(tmp_path, monkeypatch, changing_registry=changing_registry)
    worker._run()
    for group in ("experiment", "second"):
        members = [item for item in instances if item.group == group]
        assert len(members) == 1, "registry refresh/rotation must preserve in-progress confirmation"
        assert members[0].steps > 5
        assert members[0].is_closed
        visits = [kind for _, name, kind, _ in calls if name == group]
        assert "maintenance" in visits
        assert all(left != right for left, right in zip(visits, visits[1:])), visits
    busy_turns = [turn for turn, _, _, _ in calls]
    assert len(busy_turns) == len(set(busy_turns)), "each machine turn has one Group work quantum"


def test_new_group_is_discovered_during_an_unfinished_group_bootstrap(tmp_path, monkeypatch):
    cfg, worker, instances, calls, _, _ = setup_worker(tmp_path, monkeypatch)
    worker._stop.callback = lambda turn: create_group(cfg, "later") if turn == 100 else None
    worker._run()
    assert any(item.group == "later" and item.steps > 2 for item in instances)
    assert any(group == "later" and kind == "maintenance" for _, group, kind, _ in calls)


def test_disabled_binding_stops_service_and_closes_existing_owners(tmp_path, monkeypatch):
    _, worker, instances, calls, runtime, binding = setup_worker(tmp_path, monkeypatch)
    stop = worker._stop
    monkeypatch.setattr(
        runtime, "load_registry", lambda: (2, [replace(binding, enabled=False)]) if stop.turn >= 100 else (1, [binding])
    )
    worker._run()
    assert instances
    assert all(item.is_closed for item in instances)
    assert all(turn < 100 for turn, _, _, _ in calls)


def test_failed_close_is_retried_without_stopping_other_groups(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg, worker, _, calls, _, _ = setup_worker(tmp_path, monkeypatch)
    base = module.GroupDiscoveryService
    stop = worker._stop
    close_attempts = []

    class SlowClose(base):
        is_closing = False

        def request_close(self):
            if self.group == "experiment":
                self.is_closing = True
            else:
                super().request_close()

        def advance(self):
            if self.is_closing:
                close_attempts.append(stop.turn)
                if stop.turn < 200:
                    raise OSError("temporarily failed to release source")
                self.is_closed = True
                return {"state": "closed", "reason": None}
            return super().advance()

    monkeypatch.setattr(module, "GroupDiscoveryService", SlowClose)
    stop.callback = lambda turn: group_path(cfg.shared_root, "experiment").unlink() if turn == 70 else None
    worker._run()
    assert close_attempts
    assert any(70 < turn < 200 and group == "second" and kind == "discovery" for turn, group, kind, _ in calls)
    assert max(close_attempts) >= 200


def test_binding_reenabled_before_first_group_admission_does_not_stay_blocked(tmp_path, monkeypatch):
    _, worker, instances, _, runtime, binding = setup_worker(tmp_path, monkeypatch)
    stop = worker._stop

    def registry():
        if stop.turn == 0:
            return 1, [binding]
        if stop.turn < 20:
            return 2, [replace(binding, enabled=False)]
        return 3, [binding]

    monkeypatch.setattr(runtime, "load_registry", registry)
    worker._run()
    assert any(item.steps > 3 for item in instances)
