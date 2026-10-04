from __future__ import annotations

import multiprocessing
from copy import deepcopy
from dataclasses import replace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime import upgrade
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.records import utc_now
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import framework
from qqtools.plugins.qexp.runtime.upgrade.machine import advance_registered_upgrades

pytestmark = pytest.mark.integration


class _ProgressPlugin(upgrade.MigrationPlugin):
    spec = upgrade.MigrationSpec(
        name="progress-test",
        source_protocol="schema-6",
        target_protocol="metadata:progress-test",
        compatible_readers=("schema-6",),
        compatible_writers=("schema-6",),
        phases=("backfill",),
    )

    def is_applicable(self, cfg):
        return True

    def backfill(self, context):
        path = context.cfg.shared_root / "operations/upgrades/test-count.json"
        try:
            count = context.storage.read_json(path)["count"]
        except FileNotFoundError:
            count = 0
        count += 1
        context.storage.atomic_replace(path, {"count": count})
        return upgrade.PhaseResult(
            "progressed",
            metadata_ops=4,
            io_bytes=1024,
            progress={
                "version": 1,
                "scope": "shared_project",
                "stage": "sources",
                "stage_label": "Sources",
                "stage_state": "processing",
                "scan_epoch": 1,
                "completed_units": count,
                "inventoried_units": 100,
                "total_units": 100,
                "total_kind": "snapshot_exact",
                "unit": "sources",
                "remaining_stages": [],
                "last_progress_at": utc_now(),
            },
            progress_evidence={
                "version": 1,
                "identity": "a" * 64,
                "activation_revision": count,
                "activation_cursor": count,
                "source_checkpoint": None,
            },
        )


def _coordinator(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    coordinator = upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()]))
    coordinator.discover()
    return coordinator


def _lock_holder(path, acquired, release):
    with exclusive(path) as locked:
        if not locked:
            raise RuntimeError("test lock holder could not acquire lock")
        acquired.set()
        if not release.wait(15):
            raise RuntimeError("test lock holder was not released")


def _advance_holder(cfg, entered, proceed, committed, release):
    original_save = framework._save_journal

    def save_after_pause(root, journal):
        item = journal["upgrade"]["migrations"]["progress-test"]
        if item.get("in_flight"):
            entered.set()
            if not proceed.wait(15):
                raise RuntimeError("winner was not released to commit")
        original_save(root, journal)
        if not item.get("in_flight"):
            committed.set()
            if not release.wait(15):
                raise RuntimeError("winner lock was not released")

    framework._save_journal = save_after_pause
    coordinator = upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()]))
    coordinator.advance()


def _initial_advance_holder(cfg, committed, release):
    original_save = framework._save_journal

    def hold_committed_slice(root, journal):
        original_save(root, journal)
        item = journal["upgrade"]["migrations"]["progress-test"]
        if item.get("progress") is not None and not item.get("in_flight"):
            committed.set()
            if not release.wait(15):
                raise RuntimeError("initial winner was not released")

    framework._save_journal = hold_committed_slice
    upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()])).advance()


def _discovery_before_first_save(cfg, entered, proceed):
    original_save = framework._save_journal

    def pause_initial_save(root, journal):
        if not upgrade.upgrade_journal_path(root).exists():
            entered.set()
            if not proceed.wait(15):
                raise RuntimeError("initial discovery was not released")
        original_save(root, journal)

    framework._save_journal = pause_initial_save
    upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()])).advance()


def test_pause_before_first_journal_save_is_retained_and_stops_the_winner(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    coordinator = upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()]))
    process_context = multiprocessing.get_context("fork")
    entered, proceed = process_context.Event(), process_context.Event()
    child = process_context.Process(target=_discovery_before_first_save, args=(cfg, entered, proceed))
    child.start()
    try:
        assert entered.wait(10)
        assert not upgrade.upgrade_journal_path(cfg).exists()
        status = coordinator.request_pause("inspect before first slice")
        assert status["state"] == "pause_pending"
        assert status["pending"] and status["migration_blocked"]
        assert not status["can_run"]
        assert framework._load_pause_intent(cfg)["reason"] == "inspect before first slice"
    finally:
        proceed.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0
    assert coordinator.status()["state"] == "paused"
    assert not (cfg.shared_root / "operations/upgrades/test-count.json").exists()


@pytest.mark.parametrize("entry_point", ["discover", "advance"])
@pytest.mark.parametrize("hold_lock", [False, True])
def test_first_discovery_rechecks_state_without_overwriting_a_concurrent_winner(
    tmp_path, monkeypatch, entry_point, hold_lock
):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    coordinator = upgrade.UpgradeCoordinator(cfg, registry=upgrade.MigrationRegistry([_ProgressPlugin()]))
    assert not upgrade.upgrade_journal_path(cfg).exists()
    process_context = multiprocessing.get_context("fork")
    committed, release = process_context.Event(), process_context.Event()
    child = process_context.Process(target=_initial_advance_holder, args=(cfg, committed, release))
    original_exclusive = framework.exclusive
    original_load = framework._load_journal
    before = None
    reads_after_attempt = []
    attempted = False

    def race_before_attempt(path, **kwargs):
        nonlocal attempted, before
        assert kwargs.get("blocking") is False
        # The caller has observed an absent journal. Let a real coordinator
        # publish the first slice before the caller attempts exclusion.
        with monkeypatch.context() as child_patches:
            child_patches.setattr(framework, "exclusive", original_exclusive)
            child_patches.setattr(framework, "_load_journal", original_load)
            child.start()
        assert committed.wait(10)
        before = upgrade.upgrade_journal_path(cfg).read_bytes()
        attempted = True
        if not hold_lock:
            release.set()
            child.join(10)
            assert child.exitcode == 0
        return original_exclusive(path, **kwargs)

    def record_read(root):
        if attempted:
            reads_after_attempt.append(True)
        return original_load(root)

    monkeypatch.setattr(framework, "exclusive", race_before_attempt)
    monkeypatch.setattr(framework, "_load_journal", record_read)
    try:
        result = getattr(coordinator, entry_point)()
        assert result["progress"]["completed_units"] == (2 if entry_point == "advance" and not hold_lock else 1)
        if hold_lock:
            assert len(reads_after_attempt) == 1
            assert result["invocation"]["contended_lock"] == "upgrade"
            assert not result["invocation"]["slice_committed"]
            assert upgrade.upgrade_journal_path(cfg).read_bytes() == before
        elif entry_point == "discover":
            assert upgrade.upgrade_journal_path(cfg).read_bytes() == before
    finally:
        release.set()
        if child.pid is not None:
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(5)
    assert child.exitcode == 0


@pytest.mark.parametrize("initial", [True, False])
def test_machine_discovery_contention_remains_pending_without_writing(tmp_path, monkeypatch, initial):
    coordinator = _coordinator(tmp_path)
    cfg = coordinator.cfg
    if initial:
        upgrade.upgrade_journal_path(cfg).unlink()
    else:
        coordinator.advance()
    before = upgrade.upgrade_journal_path(cfg).read_bytes() if not initial else None
    extra = _ProgressPlugin()
    extra.spec = replace(extra.spec, name="new-progress-test")
    monkeypatch.setattr(framework, "DEFAULT_MIGRATIONS", upgrade.MigrationRegistry([_ProgressPlugin(), extra]))
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    process_context = multiprocessing.get_context("fork")
    acquired, release = process_context.Event(), process_context.Event()
    child = process_context.Process(
        target=_lock_holder, args=(cfg.shared_root / "locks/upgrade.lock", acquired, release)
    )
    child.start()
    original_exclusive = framework.exclusive

    def nonblocking_only(path, **kwargs):
        assert kwargs.get("blocking") is False
        return original_exclusive(path, **kwargs)

    monkeypatch.setattr(framework, "exclusive", nonblocking_only)
    try:
        assert acquired.wait(10)
        result = advance_registered_upgrades(runtime, force_discovery=True)
        assert result["pending_project_ids"]
        assert not result["all_roots_complete"]
        assert result["projects"][0]["invocation"]["contended_lock"] == "upgrade"
        assert not result["projects"][0]["invocation"]["slice_committed"]
        assert (upgrade.upgrade_journal_path(cfg).read_bytes() if not initial else None) == before
        if initial:
            assert not upgrade.upgrade_journal_path(cfg).exists()
    finally:
        release.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0


@pytest.mark.parametrize("advance_winner", [False, True])
def test_real_upgrade_lock_loser_rereads_once_and_never_writes(tmp_path, monkeypatch, advance_winner):
    coordinator = _coordinator(tmp_path)
    first = coordinator.advance()
    assert first["progress"]["completed_units"] == 1
    process_context = multiprocessing.get_context("fork")
    entered, proceed, committed, release = (process_context.Event() for _ in range(4))
    child = process_context.Process(
        target=_advance_holder, args=(coordinator.cfg, entered, proceed, committed, release)
    )
    child.start()
    try:
        assert entered.wait(10)
        original_exclusive = framework.exclusive
        original_load = framework._load_journal
        rereads = []
        attempted = False

        def before_attempt(path, **kwargs):
            nonlocal attempted
            if kwargs.get("blocking") is False:
                attempted = True
                if advance_winner:
                    proceed.set()
                    assert committed.wait(10)
            return original_exclusive(path, **kwargs)

        def record_read(cfg):
            if attempted:
                rereads.append(True)
            return original_load(cfg)

        def forbidden_write(*_args, **_kwargs):
            raise AssertionError("upgrade-lock loser wrote the journal")

        monkeypatch.setattr(framework, "exclusive", before_attempt)
        monkeypatch.setattr(framework, "_load_journal", record_read)
        monkeypatch.setattr(framework, "_save_journal", forbidden_write)
        result = coordinator.advance()
        assert len(rereads) == 1
        assert result["invocation"]["contended_lock"] == "upgrade"
        assert result["invocation"]["slice_committed"] is False
        assert result["invocation"]["observed_progress"] is advance_winner
        assert result["invocation"]["delta"] is None
        assert result["invocation"]["next_probe_at"]
    finally:
        proceed.set()
        release.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0


def test_schema_lock_contention_is_durable_waiting_without_coordinator_claim(tmp_path):
    coordinator = _coordinator(tmp_path)
    coordinator.advance()
    plugin = _ProgressPlugin()
    plugin.spec = upgrade.MigrationSpec(
        name="progress-test",
        source_protocol="schema-6",
        target_protocol="metadata:progress-test",
        compatible_readers=("schema-6",),
        compatible_writers=("schema-6",),
        phases=("activation",),
    )
    coordinator.registry = upgrade.MigrationRegistry([plugin])
    journal = read_json(upgrade.upgrade_journal_path(coordinator.cfg))
    journal["upgrade"]["migrations"]["progress-test"].update(phase="activation", audit_passed=True)
    framework._save_journal(coordinator.cfg, journal)
    process_context = multiprocessing.get_context("fork")
    acquired, release = process_context.Event(), process_context.Event()
    child = process_context.Process(
        target=_lock_holder, args=(coordinator.cfg.shared_root / "locks/schema.lock", acquired, release)
    )
    child.start()
    try:
        assert acquired.wait(10)
        result = coordinator.advance()
        assert result["state"] == "waiting"
        assert "schema_lock_busy" in result["blockers"]
        assert result["invocation"]["contended_lock"] == "schema"
        assert not result["invocation"]["slice_committed"]
        assert not result["invocation"]["observed_progress"]
        assert result["progress"] is None
    finally:
        release.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0


@pytest.mark.parametrize("fields", ["preserved", "omitted", "overwritten"])
def test_released_journal_save_invalidates_unknown_progress(tmp_path, fields):
    coordinator = _coordinator(tmp_path)
    assert coordinator.advance()["progress"] is not None
    journal = read_json(upgrade.upgrade_journal_path(coordinator.cfg))
    item = journal["upgrade"]["migrations"]["progress-test"]
    if fields == "omitted":
        item.pop("progress")
    elif fields == "overwritten":
        item["progress"] = {"version": 0}
    # Exact released _save_journal body, unchanged in v1.3.22 through v1.3.25.
    # It preserves unknown fields while renewing only the containing revision.
    journal["upgrade"]["revision"] = int(journal["upgrade"].get("revision", 0)) + 1
    journal["upgrade"]["updated_at"] = utc_now()
    atomic_replace(upgrade.upgrade_journal_path(coordinator.cfg), journal)
    original = deepcopy(journal)
    status = coordinator.status()
    assert status["progress"] is None
    assert status["state"] == "runnable"
    assert not status["admission_blocked"]
    assert read_json(upgrade.upgrade_journal_path(coordinator.cfg)) == original


@pytest.mark.parametrize("cut", ["in_flight", "final"])
def test_crash_boundary_suppresses_old_snapshot_and_does_not_replay_count(tmp_path, monkeypatch, cut):
    coordinator = _coordinator(tmp_path)
    coordinator.advance()
    save = framework._save_journal

    class Crash(BaseException):
        pass

    def interrupted(cfg, journal):
        in_flight = journal["upgrade"]["migrations"]["progress-test"].get("in_flight")
        if not in_flight and cut == "final":
            raise Crash
        save(cfg, journal)
        if in_flight and cut == "in_flight":
            raise Crash

    with monkeypatch.context() as patch:
        patch.setattr(framework, "_save_journal", interrupted)
        with pytest.raises(Crash):
            coordinator.advance()
    assert coordinator.status()["progress"] is None
    result = coordinator.advance()
    assert result["progress"]["completed_units"] == (2 if cut == "in_flight" else 3)
    assert result["invocation"]["slice_committed"] is True


def test_nonprogress_pause_save_invalidates_observation_without_changing_admission(tmp_path):
    coordinator = _coordinator(tmp_path)
    coordinator.advance()
    paused = coordinator.request_pause("inspect source")
    assert paused["state"] == "paused"
    assert paused["progress"] is None
    assert not paused["admission_blocked"]


def test_bad_plugin_observation_cannot_fail_safe_migration(tmp_path, monkeypatch):
    coordinator = _coordinator(tmp_path)
    coordinator.advance()
    original = _ProgressPlugin.backfill

    def malformed(self, context):
        return replace(original(self, context), progress={"version": 1, "completed_units": "corrupt"})

    monkeypatch.setattr(_ProgressPlugin, "backfill", malformed)
    result = coordinator.advance()
    assert result["state"] == "runnable"
    assert result["progress"] is None
    assert not result["admission_blocked"]
    assert result["invocation"]["slice_committed"] is True


def test_status_reads_no_directory_or_source_history(tmp_path, monkeypatch):
    coordinator = _coordinator(tmp_path)
    coordinator.advance()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("status scanned history")

    with monkeypatch.context() as patch:
        patch.setattr("os.scandir", forbidden)
        patch.setattr("os.listdir", forbidden)
        patch.setattr("qqtools.plugins.qexp.runtime.directory_capture.read_directory_entry", forbidden)
        for _ in range(3):
            assert coordinator.status()["progress"]["completed_units"] == 1
