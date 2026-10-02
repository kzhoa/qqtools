from contextlib import contextmanager
from threading import Event, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.inventory import ProjectInventoryEntry, load_inventory, save_inventory_locked
from qqtools.plugins.qexp.agent.project_admin import enable_project
from qqtools.plugins.qexp.agent.setup import SetupOperationalError, initialize_machine, set_project_enablement

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _enable_cli(runtime, project_id):
    return set_project_enablement(runtime, project_id, True)


def _registered_pair(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    for name in ("subject", "peer"):
        cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
        bindings.append(runtime.add_binding(cfg.shared_root, cfg.machine_name))
    bindings[0] = runtime.set_enabled(bindings[0].project_id, False)
    with runtime.inventory_guard():
        save_inventory_locked(
            runtime,
            1,
            [ProjectInventoryEntry(item.project_id, item.shared_root, item.enabled) for item in bindings],
        )
    return runtime, bindings[0], bindings[1]


@pytest.mark.parametrize("is_eligible", [True, False])
@pytest.mark.parametrize("enable", [_enable_cli, enable_project], ids=["cli", "python_api"])
def test_enable_checks_shared_registration_outside_all_machine_guards(tmp_path, monkeypatch, is_eligible, enable):
    runtime, subject, _peer = _registered_pair(tmp_path)
    held = set()

    def guarded(name, original):
        @contextmanager
        def guard(*args, **kwargs):
            with original(*args, **kwargs) as value:
                held.add(name)
                try:
                    yield value
                finally:
                    held.remove(name)

        return guard

    for name in ("agent_lifecycle_guard", "inventory_guard", "registry_guard"):
        monkeypatch.setattr(runtime, name, guarded(name, getattr(runtime, name)))
    original = runtime.registration.binding_write_eligible
    calls = []

    def eligible(binding, **kwargs):
        assert not held, f"shared registration check held machine guards: {held}"
        calls.append(binding)
        return original(binding, **kwargs) if is_eligible else False

    def status(binding):
        assert not held, f"shared registration status held machine guards: {held}"
        assert binding == subject
        return {"state": "expired"}

    monkeypatch.setattr(runtime.registration, "binding_write_eligible", eligible)
    monkeypatch.setattr(runtime.registration, "registration_status", status)
    if not is_eligible:
        with pytest.raises((SetupOperationalError, RuntimeError), match="registration is expired"):
            enable(runtime, subject.project_id)
        assert calls == [subject]
        assert not next(item for item in runtime.load_registry()[1] if item.project_id == subject.project_id).enabled
        return
    result = enable(runtime, subject.project_id)
    if enable is _enable_cli:
        assert result["status"] == "committed"
        assert result["effective_enabled"] is True
        assert next(item for item in load_inventory(runtime)[1] if item.project_id == subject.project_id).enabled
    else:
        assert result.enabled is True
        assert result in runtime.load_registry_uncached()[1]
    assert calls == [subject]


@pytest.mark.parametrize("enable", [_enable_cli, enable_project], ids=["cli", "python_api"])
@pytest.mark.parametrize("is_eligible", [True, False])
def test_blocked_enable_does_not_lock_peer_and_cannot_overwrite_new_registry(
    tmp_path, monkeypatch, enable, is_eligible
):
    runtime, subject, peer = _registered_pair(tmp_path)
    entered = Event()
    release = Event()
    peer_finished = Event()
    results = []
    errors = []
    original = runtime.registration.binding_write_eligible

    def blocked_eligibility(binding, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release shared registration barrier"
        return original(binding, **kwargs) if is_eligible else False

    monkeypatch.setattr(runtime.registration, "binding_write_eligible", blocked_eligibility)

    def enable_subject():
        try:
            results.append(enable(runtime, subject.project_id))
        except Exception as exc:
            errors.append(exc)

    def disable_peer():
        try:
            results.append(set_project_enablement(runtime, peer.project_id, False))
        except Exception as exc:
            errors.append(exc)
        finally:
            peer_finished.set()

    enabling = Thread(target=enable_subject)
    disabling = Thread(target=disable_peer)
    enabling.start()
    try:
        assert entered.wait(5)
        disabling.start()
        assert peer_finished.wait(2), "blocked shared validation retained machine authority locks"
        assert not release.is_set()
        assert len(results) == 1
        assert results[0]["action"] == "project_disabled"
        assert results[0]["status"] == "committed"
    finally:
        release.set()
        enabling.join(10)
        if disabling.ident is not None:
            disabling.join(10)
    assert not enabling.is_alive()
    assert not disabling.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], (SetupOperationalError, RuntimeError))
    assert "changed during enablement validation" in str(errors[0])
    assert all(not item.enabled for item in runtime.load_registry_uncached()[1])
    monkeypatch.setattr(runtime.registration, "binding_write_eligible", original)
    assert set_project_enablement(runtime, subject.project_id, True)["status"] == "committed"


@pytest.mark.parametrize("enable", [_enable_cli, enable_project], ids=["cli", "python_api"])
@pytest.mark.parametrize("is_eligible", [True, False])
def test_enable_rejects_removed_binding_before_interpreting_shared_result(tmp_path, monkeypatch, enable, is_eligible):
    runtime, subject, peer = _registered_pair(tmp_path)

    def remove_during_validation(_binding, **_kwargs):
        with runtime.registry_guard():
            revision, bindings = runtime.load_registry()
            runtime.registration.save_registry_locked(
                revision + 1, [item for item in bindings if item.project_id != subject.project_id]
            )
        return is_eligible

    monkeypatch.setattr(runtime.registration, "binding_write_eligible", remove_during_validation)
    monkeypatch.setattr(runtime.registration, "registration_status", lambda _binding: {"state": "superseded"})
    with pytest.raises(RuntimeError, match="changed during enablement validation; retry"):
        enable(runtime, subject.project_id)
    assert runtime.load_registry_uncached()[1] == [peer]
