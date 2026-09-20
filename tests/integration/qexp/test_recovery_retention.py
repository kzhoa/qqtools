"""Unfinished recovery must retain the binding and prevent idle exit."""

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
from qqtools.plugins.qexp.events import flush_local_events
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.cpu_lane import attach_cpu, release_cpu, reserve_cpu, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


def test_binding_removal_cannot_overlap_cleanup_or_import(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_capture import capture_cleanup_guard

    _, runtime, binding, root = binding_fixture(tmp_path)
    with capture_cleanup_guard(root) as can_cleanup:
        assert can_cleanup
        with pytest.raises(RuntimeError, match="writer capture"):
            runtime.remove_binding(binding.project_id)
        assert root.exists()
        assert runtime.load_registry()[1] == [binding]
    assert runtime.remove_binding(binding.project_id) == binding
    assert not root.exists()


def test_binding_removal_fence_survives_unlinking_the_partition(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import context
    from qqtools.plugins.qexp.runtime.responsibility_capture import capture_cleanup_guard

    _, runtime, binding, root = binding_fixture(tmp_path)
    original = context.shutil.rmtree
    checked = []

    def remove(path, *args, **kwargs):
        assert path == root
        with capture_cleanup_guard(root) as can_cleanup:
            assert not can_cleanup
        original(path, *args, **kwargs)
        with capture_cleanup_guard(root) as can_cleanup:
            assert not can_cleanup
        assert not root.exists()
        checked.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(context.shutil, "rmtree", remove)
        assert runtime.remove_binding(binding.project_id) == binding
    assert checked == [root] and not root.exists()


@pytest.mark.parametrize("is_corrupt", [False, True])
@pytest.mark.parametrize("location", ["current", "legacy"])
def test_pending_writer_capture_retains_an_empty_binding(tmp_path, is_corrupt, location):
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE, WriterCaptureCheckpoint

    cfg, runtime, binding, root = binding_fixture(tmp_path)
    capture_root = root if location == "current" else cfg.runtime_root
    if location == "legacy":
        atomic_replace(
            runtime.migration_path(binding.project_id),
            {"migration": {"state": "active", "legacy_runtime_root": str(capture_root)}},
        )
    ledger = Ledger.open_or_create(responsibility_root(capture_root))
    with runtime.migration_guard():
        with WriterCaptureCheckpoint(ledger, capture_root).observe():
            pass
    if is_corrupt:
        (capture_root / CAPTURE_FILE).write_text("broken")
    assert not ledger.has_members()
    prefix = "legacy:" if location == "legacy" else ""
    assert f"{prefix}writer_capture_pending" in runtime.binding_blockers(binding)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match="writer capture"):
        runtime.remove_binding(binding.project_id)
    assert root.exists() and (capture_root / CAPTURE_FILE).exists()


def binding_fixture(tmp_path, *, enabled=False):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "old")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name, enabled=enabled)
    runtime.last_cycle_had_demand = False
    root = runtime.project_paths(binding.project_id)["root"]
    root.mkdir(parents=True, exist_ok=True)
    return cfg, runtime, binding, root


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("state", ["active", "maintenance", "source_only", "corrupt"])
def test_membership_without_evidence_prevents_idle_exit(tmp_path, enabled, state):
    cfg, runtime, binding, root = binding_fixture(tmp_path, enabled=enabled)
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    ledger = Ledger.open_or_create(responsibility_root(root))
    payload = {"task_id": "task", "attempt_number": 1}
    generation = ledger.publish("task-attempt-1", payload)
    if state == "maintenance":
        ledger.handoff("task-attempt-1", generation)
    elif state == "source_only":
        ledger.capture_source("task-attempt-1", payload, cfg.runtime_root)
    elif state == "corrupt":
        (ledger.root / "marker").write_text("broken")

    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    disabled = runtime.set_enabled(binding.project_id, False)
    assert runtime.binding_state(disabled) == "draining"
    with pytest.raises(RuntimeError, match="recovery_responsibilities"):
        runtime.remove_binding(binding.project_id)
    assert root.exists()


@pytest.mark.parametrize("state", ["prepared", "legacy_agent_stopped", "reservations_imported", "blocked"])
def test_incomplete_migration_is_retained_before_first_capture(tmp_path, state):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    value = {"migration": {"state": state, "legacy_runtime_root": str(cfg.runtime_root)}}
    atomic_replace(runtime.migration_path(binding.project_id), value)

    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match="legacy_migration_incomplete"):
        runtime.remove_binding(binding.project_id)
    assert root.exists()

    value["migration"]["state"] = "active"
    atomic_replace(runtime.migration_path(binding.project_id), value)
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert runtime.remove_binding(binding.project_id) == binding


@pytest.mark.parametrize(
    "lane",
    [
        "registrations",
        "observations",
        "launch_intents",
        "wrappers",
        "authority_diagnostics",
        "active",
        "cpu_active",
        "cpu_provisional",
        "events",
    ],
)
def test_unimported_source_evidence_prevents_retirement(tmp_path, lane):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    atomic_replace(
        runtime.migration_path(binding.project_id),
        {"migration": {"state": "active", "legacy_runtime_root": str(cfg.runtime_root)}},
    )
    source = local_paths(cfg.runtime_root)[lane] / "unimported.json"
    atomic_replace(source, {"fixture": "uncaptured source evidence"})

    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match=f"legacy:{lane}"):
        runtime.remove_binding(binding.project_id)
    assert source.exists() and root.exists()

    source.unlink()
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert runtime.remove_binding(binding.project_id) == binding


def test_unreadable_migration_retains_binding_and_prevents_idle_exit(tmp_path):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    runtime.migration_path(binding.project_id).write_text("broken")
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match="legacy_migration_unavailable"):
        runtime.remove_binding(binding.project_id)
    assert root.exists()


@pytest.mark.parametrize("is_active", [False, True])
def test_existing_cpu_snapshot_prevents_reservation_only_idle_exit(tmp_path, is_active):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    reservation = reserve_cpu(runtime.root, "task", 1, attempt_id="task-attempt-1", fencing_token=1)["reservation"]
    if is_active:
        attach_cpu(runtime.root, reservation["reservation_id"], "task-attempt-1", 1)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    release_cpu(runtime.root, reservation["reservation_id"])
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)


def test_idle_retention_stops_at_first_evidence_file(tmp_path, monkeypatch):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    directory = local_paths(root)["processes"]
    first = directory / "first.json"
    atomic_replace(first, {"process": {"attempt_id": "first"}})
    atomic_replace(directory / "second.json", {"process": {"attempt_id": "second"}})
    original_scandir = os.scandir
    closed = []

    @contextmanager
    def guarded_scandir(path):
        with original_scandir(path) as entries:
            if path != directory:
                yield entries
                return

            def first_only():
                yield next(entries)
                raise AssertionError("idle detection enumerated unnecessary evidence")

            try:
                yield first_only()
            finally:
                closed.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", guarded_scandir)
        assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert closed == [directory]


@pytest.mark.parametrize("is_legacy", [False, True])
@pytest.mark.parametrize("is_nested", [False, True])
def test_unreadable_evidence_directory_retains_recovery(tmp_path, monkeypatch, is_legacy, is_nested):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    if is_legacy:
        atomic_replace(
            runtime.migration_path(binding.project_id),
            {"migration": {"state": "active", "legacy_runtime_root": str(cfg.runtime_root)}},
        )
    paths = local_paths(cfg.runtime_root if is_legacy else root)
    directory = paths["termination_decisions"] / "attempt" if is_nested else paths["observations"]
    evidence = directory / "retained.json"
    atomic_replace(evidence, {"fixture": "unreadable recovery"})
    original = os.scandir

    def unavailable(path):
        if Path(path) == directory:
            raise PermissionError("evidence inaccessible")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", unavailable)
        assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
        with pytest.raises(RuntimeError, match="unavailable"):
            runtime.remove_binding(binding.project_id)
    assert evidence.exists() and root.exists()


def test_evidence_lane_replaced_by_file_is_not_empty(tmp_path):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    directory = local_paths(root)["observations"]
    directory.write_text("damaged lane")
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.remove_binding(binding.project_id)
    assert directory.read_text() == "damaged lane"


@pytest.mark.parametrize("value", [{}, {"reservation": {}}, {"reservation": {"project_id": None}}])
def test_unknown_reservation_ownership_prevents_binding_removal(tmp_path, value):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    path = runtime.paths["active"] / "unknown.json"
    atomic_replace(path, value)
    with pytest.raises(RuntimeError, match="reservation_unavailable"):
        runtime.remove_binding(binding.project_id)
    assert root.exists() and read_json(path) == value


@pytest.mark.parametrize("lane", ["migration", "membership"])
def test_inaccessible_recovery_metadata_never_proves_idle(tmp_path, monkeypatch, lane):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    path = runtime.migration_path(binding.project_id) if lane == "migration" else responsibility_root(root)
    original = Path.stat

    def unavailable(candidate, **kwargs):
        if candidate == path:
            raise PermissionError("metadata inaccessible")
        return original(candidate, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", unavailable)
        assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
        with pytest.raises(PermissionError, match="metadata inaccessible"):
            runtime.remove_binding(binding.project_id)
    assert root.exists()


def test_pending_local_event_retains_binding_until_shared_publication(tmp_path):
    cfg, runtime, binding, root = binding_fixture(tmp_path)
    event = {"event_id": "diagnostic", "timestamp": "2026-09-18T00:00:00Z", "event_type": "retained"}
    source = local_paths(root)["events"] / "machine" / "diagnostic.json"
    atomic_replace(source, event)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    with pytest.raises(RuntimeError, match="events:"):
        runtime.remove_binding(binding.project_id)
    assert flush_local_events(replace(cfg, runtime_root=root)) == 1
    assert read_json(cfg.shared_root / "events" / "2026-09-18" / source.name) == event
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    runtime.remove_binding(binding.project_id)
