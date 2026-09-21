"""Additive discovery must preserve existing launch and recovery authority."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.executor import Executor
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.responsibility import ResponsibilityReader, publish_launch, responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO, Ledger, identity_key
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task


def prepared(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"])
    claimed = claim_task(cfg, task.task_id, [0])
    assert claimed is not None
    assert authorize_launch(cfg, task.task_id, claimed.attempt_id, claimed.current_fencing_token)
    attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
    return cfg, task, attempt


def finish_reader(reader):
    reader.close()
    if reader._done is not None:
        assert reader._done.wait(5), "test-owned index worker did not finish"


def test_executor_publishes_durable_membership_before_runner_creation(tmp_path):
    cfg, task, attempt = prepared(tmp_path)
    spawned = []

    def spawn(argv, **kwargs):
        entry = Ledger(responsibility_root(cfg.runtime_root)).lookup(attempt.attempt_id)
        assert entry["stage"] == "active"
        assert entry["payload"] == {"task_id": task.task_id, "attempt_number": 1}
        spawned.append(argv)
        return SimpleNamespace(pid=4321, wait=lambda: 0)

    executor = Executor(tmux_available=lambda: False, spawn_runner=spawn)
    reference, handoff = executor.initiate_attempt(cfg, task.task_id, attempt)
    assert reference == "pid:4321"
    assert handoff.attempt_id == attempt.attempt_id
    assert len(spawned) == 1
    # The projection itself never advances the Attempt's authority or launches.
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "starting"


def test_executor_reaps_direct_runner_child(tmp_path):
    cfg, task, attempt = prepared(tmp_path)
    children = []

    def spawn(_argv, **_kwargs):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        children.append(process)
        intent = local_paths(cfg.runtime_root)["launch_intents"] / f"{attempt.attempt_id}.json"
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.touch()
        return process

    Executor(tmux_available=lambda: False, spawn_runner=spawn).launch_attempt(cfg, task.task_id, attempt)

    deadline = time.monotonic() + 2
    while children[0].returncode is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert children[0].returncode == 0


@pytest.mark.parametrize("has_tmux", [False, True])
def test_unavailable_projection_prevents_executor_creation(tmp_path, has_tmux):
    cfg, task, attempt = prepared(tmp_path)
    root = responsibility_root(cfg.runtime_root)
    root.mkdir()
    (root / "marker").write_text("broken")
    assert not publish_launch(cfg, task.task_id, attempt)
    spawned = []

    def create(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("unpublished launch reached process/window creation")

    with pytest.raises(RuntimeError, match="responsibility"):
        Executor(tmux_available=lambda: has_tmux, spawn_runner=create, create_window=create).initiate_attempt(
            cfg, task.task_id, attempt
        )
    assert not spawned
    assert (root / "marker").read_text() == "broken"


def test_runner_requires_publication_without_executor(tmp_path):
    from qqtools.plugins.qexp.runner import launch_intent_path, run_attempt

    cfg, task, attempt = prepared(tmp_path)
    root = responsibility_root(cfg.runtime_root)
    root.mkdir()
    (root / "marker").write_text("broken")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unpublished runner reached workload creation")

    with pytest.raises(RuntimeError, match="responsibility"):
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            attempt.authorization["launch_id"],
            popen_factory=forbidden,
        )
    assert not launch_intent_path(cfg, attempt.attempt_id).exists()
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "starting"


def test_runner_rechecks_publication_without_writes_or_shared_lock_and_can_finish_during_damage(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runner import run_attempt
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    cfg, task, attempt = prepared(tmp_path)
    Executor(
        tmux_available=lambda: False,
        spawn_runner=lambda *_args, **_kwargs: SimpleNamespace(pid=4321, wait=lambda: 0),
    ).initiate_attempt(cfg, task.task_id, attempt)
    original = Ledger.publish
    checks = []

    def check_unlocked(self, *args, **kwargs):
        with schema_lock(cfg.shared_root, blocking=False) as acquired:
            assert acquired, "runner publication held the shared authority fence"
        checks.append(args)
        return original(self, *args, **kwargs)

    def no_ledger_writes(*_args, **_kwargs):
        pytest.fail("exact runner publication issued a durability write")

    def finish():
        # Already-started workloads must retain offline completion even if the
        # optional discovery reader can no longer open the membership store.
        (responsibility_root(cfg.runtime_root) / "marker").write_text("broken")
        return 0

    monkeypatch.setattr(Ledger, "publish", check_unlocked)
    monkeypatch.setattr(DurableIO, "replace", no_ledger_writes)
    monkeypatch.setattr(DurableIO, "sync", no_ledger_writes)
    assert (
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            attempt.authorization["launch_id"],
            popen_factory=lambda *_args, **_kwargs: SimpleNamespace(pid=99999993, wait=finish),
        )
        == 0
    )
    assert len(checks) == 1
    observation = local_paths(cfg.runtime_root)["observations"] / f"{attempt.attempt_id}.json"
    assert read_json(observation)["exit_observation"]["observed_exit_code"] == 0


@pytest.mark.parametrize("has_committed_publication", [False, True])
def test_publication_failure_reconciles_attempt_without_launch_and_preserves_explicit_retry(
    tmp_path, monkeypatch, has_committed_publication
):
    from qqtools.plugins.qexp.commands.task import retry
    from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from qqtools.plugins.qexp.scheduler import run_dispatch_cycle

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"])
    spawned = []

    def spawn(*args, **kwargs):
        spawned.append(args)
        identity = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        atomic_replace(local_paths(cfg.runtime_root)["launch_intents"] / f"{identity}.json", {})
        return SimpleNamespace(pid=4321, wait=lambda: 0)

    executor = Executor(tmux_available=lambda: False, spawn_runner=spawn)
    original = Ledger.publish

    def interrupted(self, *args, **kwargs):
        if has_committed_publication:
            original(self, *args, **kwargs)
        raise OSError("injected interrupted publication")

    with monkeypatch.context() as patch:
        patch.setattr(Ledger, "publish", interrupted)
        assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor, should_recover_starting=False) == []
    assert not spawned
    assert load_task(cfg, task.task_id).state["projection"] == "failed"
    assert not active_reservations(cfg.runtime_root)
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "failed"
    retry(cfg, task.task_id)
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor, should_recover_starting=False) == [
        task.task_id
    ]
    assert len(spawned) == 1
    assert load_task(cfg, task.task_id).attempt_control["current_attempt_number"] == 2


def test_runner_revalidates_claim_after_membership_publication(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import runner
    from qqtools.plugins.qexp.scheduler import fail_attempt

    cfg, task, attempt = prepared(tmp_path)
    original = runner.require_launch_responsibility

    def concurrent_terminal(*args):
        original(*args)
        fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "concurrent_failure")

    def forbidden(*_args, **_kwargs):
        pytest.fail("terminal Attempt created a workload after publication")

    monkeypatch.setattr(runner, "require_launch_responsibility", concurrent_terminal)
    with pytest.raises(RuntimeError, match="not authorized"):
        runner.run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            attempt.authorization["launch_id"],
            popen_factory=forbidden,
        )
    assert not runner.launch_intent_path(cfg, attempt.attempt_id).exists()


def test_reader_reopens_membership_and_retires_only_requested_identity(tmp_path):
    cfg, task, attempt = prepared(tmp_path)
    assert publish_launch(cfg, task.task_id, attempt)
    reader = ResponsibilityReader(cfg.runtime_root)
    try:
        seen = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            seen = reader.take()
            if seen is not None:
                break
            if reader._done is not None:
                assert reader._done.wait(1)
        assert seen is not None and seen["identity"] == attempt.attempt_id
        reader.cleanup(cleanup_request(task.task_id, attempt.attempt_id))
        for _ in range(20):
            reader.take()
            if reader._done is not None:
                assert reader._done.wait(1)
            if not reader._cleanups:
                break
        ledger = Ledger(responsibility_root(cfg.runtime_root))
        assert all(ledger.service_page(bucket) == ([], None) for bucket in range(16))
    finally:
        finish_reader(reader)


def test_retained_runner_exit_reaches_terminal_truth_through_responsibility_reader(tmp_path):
    from dataclasses import replace

    from qqtools.plugins.qexp.runner import _publish_exit_observation, _publish_registration
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint
    from qqtools.plugins.qexp.runtime.tasks import load_task

    source_cfg, task, attempt = prepared(tmp_path)
    cfg = replace(source_cfg, runtime_root=tmp_path / "managed-runtime")
    cfg.runtime_root.mkdir()
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    ledger.capture_source(
        attempt.attempt_id,
        {"task_id": task.task_id, "attempt_number": 1},
        source_cfg.runtime_root,
    )
    with WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=source_cfg.runtime_root).observe():
        pass
    registration = _publish_registration(source_cfg, attempt, task, SimpleNamespace(pid=99999992))
    reader = ResponsibilityReader(cfg.runtime_root)
    try:
        deadline = time.monotonic() + 5
        destination = local_paths(cfg.runtime_root)["registrations"] / registration.name
        while not destination.exists() and time.monotonic() < deadline:
            reader.take()
            if reader._done is not None:
                assert reader._done.wait(1)
        assert read_json(destination) == read_json(registration)
    finally:
        finish_reader(reader)

    # The old runner finishes after the first read and while destructive import
    # remains prohibited. A new supervisor must consume its original outcome.
    _publish_exit_observation(source_cfg, attempt.attempt_id, 0, task_id=task.task_id)
    source_exit = local_paths(source_cfg.runtime_root)["observations"] / registration.name
    retained = {path: path.read_bytes() for path in (registration, source_exit)}
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    try:
        deadline = time.monotonic() + 5
        while load_task(cfg, task.task_id).state["projection"] != "succeeded" and time.monotonic() < deadline:
            # Drive the actual membership consumer without the old directory
            # discovery lanes, so those lanes cannot conceal a broken handoff.
            supervisor._work._responsibility_step()
            pending = supervisor._work._responsibilities._done
            if pending is not None:
                assert pending.wait(1)
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        result = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert result["attempt_id"] == attempt.attempt_id
        assert load_task(cfg, task.task_id).attempt_control["current_attempt_number"] == 1
        assert {path: path.read_bytes() for path in retained} == retained
        assert ledger.lookup(attempt.attempt_id)["stage"] == "active"
    finally:
        finish_reader(supervisor._work._responsibilities)
        supervisor.close()


def test_slow_index_io_does_not_block_existing_supervision_or_close(tmp_path, monkeypatch):
    cfg, task, attempt = prepared(tmp_path)
    assert publish_launch(cfg, task.task_id, attempt)
    entered, release = threading.Event(), threading.Event()
    original = Ledger.service_page

    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Ledger, "service_page", delayed)
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    work = supervisor._work
    try:
        # Real durable evidence predating index publication stays discoverable.
        path = local_paths(cfg.runtime_root)["registrations"] / "old-writer.json"
        atomic_replace(path, {"process_registration": {"protocol_version": 0}})
        work._responsibilities.take()
        assert entered.wait(2)
        for _ in range(80):
            supervisor.tick()
        assert not release.is_set()
        assert supervisor.work_snapshot["lanes"]["registrations"]["processed"] >= 1
        assert supervisor.work_snapshot["startup_complete"]
        supervisor.close()  # Must return while the storage worker is still blocked.
        assert not release.is_set()
    finally:
        release.set()
        finish_reader(work._responsibilities)
        supervisor.close()


def test_incomplete_evidence_cleanup_keeps_index_membership(tmp_path, monkeypatch):
    cfg, task, attempt = prepared(tmp_path)
    assert publish_launch(cfg, task.task_id, attempt)
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    request = cleanup_request(task.task_id, attempt.attempt_id)
    monkeypatch.setattr(supervisor, "_cleanup_request", lambda *_args, **_kwargs: request)
    decisions = local_paths(cfg.runtime_root)["termination_decisions"] / attempt.attempt_id
    decisions.mkdir(parents=True)
    for index in range(9):
        atomic_replace(decisions / f"{index}.json", {})
    try:
        supervisor._remove_terminal_attempt_evidence(attempt.attempt_id)
        reader = supervisor._work._responsibilities
        assert len(list(decisions.iterdir())) == 9  # Queuing alone is not durable handoff.
        reader.take()
        assert reader._done.wait(2)
        assert len(list(decisions.iterdir())) == 1
        ledger = Ledger(responsibility_root(cfg.runtime_root))
        entry = ledger.lookup(attempt.attempt_id)
        assert entry["stage"] == "maintenance"
        assert entry["cleanup_receipt"] == request.receipt
        finish_reader(reader)
        reader = ResponsibilityReader(cfg.runtime_root)
        supervisor._work._responsibilities = reader
        for _ in range(40):
            supervisor._work._responsibility_step()
            if reader._done is not None:
                assert reader._done.wait(2)
            if not decisions.exists():
                break
        assert not decisions.exists()
        assert all(ledger.service_page(bucket) == ([], None) for bucket in range(16))
    finally:
        finish_reader(supervisor._work._responsibilities)
        supervisor.close()


@pytest.mark.parametrize("has_index", [False, True])
def test_membership_never_shortcuts_startup_completeness(tmp_path, has_index):
    cfg, task, attempt = prepared(tmp_path)
    if has_index:
        assert publish_launch(cfg, task.task_id, attempt)
    for index in range(90):
        path = local_paths(cfg.runtime_root)["registrations"] / f"old-{index}.json"
        atomic_replace(path, {"process_registration": {"protocol_version": 0}})
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    try:
        supervisor.recover_startup()
        for _ in range(20):
            supervisor.tick()
        assert not supervisor.work_snapshot["startup_complete"]
    finally:
        finish_reader(supervisor._work._responsibilities)
        supervisor.close()


def terminal_attempt_with_retained_evidence(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runner import _publish_registration

    cfg, task, attempt = prepared(tmp_path)
    _publish_registration(cfg, attempt, task, SimpleNamespace(pid=99999992))
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor._materialize_registrations()
    # Model cleanup interrupted after the real terminal transition/accounting.
    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "_remove_terminal_attempt_evidence", lambda *_args: False)
        supervisor._finalize(task.task_id, attempt.attempt_id, attempt.current_fencing_token, 1, was_terminated=False)
    assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "failed"
    return cfg, task, attempt, supervisor


@pytest.mark.parametrize("has_successor", [False, True])
def test_historical_terminal_cleanup_survives_retry_without_touching_successor(tmp_path, monkeypatch, has_successor):
    from qqtools.plugins.qexp.commands.task import retry
    from qqtools.plugins.qexp.runtime.resources.reservations import has_reservation

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        retry(cfg, task.task_id)
        successor = claim_task(cfg, task.task_id, [0]) if has_successor else None
        if has_successor:
            assert successor is not None
            assert successor.attempt_number == 2
            assert has_reservation(cfg.runtime_root, successor.reservation_id)
        # The historical writer has exited; a live writer must retain evidence.
        mark_writer_exited(cfg, old.attempt_id)
        assert supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
        assert supervisor._reconcile_terminal_accounting(task.task_id, old.attempt_id)
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        assert not (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
        if successor is not None:
            assert has_reservation(cfg.runtime_root, successor.reservation_id)
            assert read_json(attempt_path(cfg.shared_root, task.task_id, 2))["attempt"]["phase"] == "claimed"
            assert not supervisor._has_terminal_attempt(task.task_id, successor.attempt_id)
    finally:
        supervisor.close()


@pytest.mark.parametrize(
    "damage", ["nonterminal", "wrong_identity", "wrong_number", "wrong_machine", "missing_task", "missing_attempt"]
)
def test_historical_cleanup_requires_matching_durable_terminal_truth(tmp_path, monkeypatch, damage):
    from qqtools.plugins.qexp.commands.task import retry

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        retry(cfg, task.task_id)
        path = attempt_path(cfg.shared_root, task.task_id, 1)
        if damage == "missing_task":
            (cfg.shared_root / "tasks" / f"{task.task_id}.json").unlink()
        elif damage == "missing_attempt":
            path.unlink()
        else:
            value = read_json(path)
            if damage == "nonterminal":
                value["attempt"]["phase"] = "orphaned"
            elif damage == "wrong_identity":
                value["attempt"]["attempt_id"] = "another-attempt"
            elif damage == "wrong_machine":
                value["attempt"]["machine_name"] = "another-machine"
            else:
                value["attempt"]["attempt_number"] = 2
            atomic_replace(path, value)
        assert not supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        assert (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
    finally:
        supervisor.close()


@pytest.mark.parametrize("phase", ["blocked", "running"])
def test_attempt_terminal_before_task_commit_is_not_cleanup_proof(tmp_path, monkeypatch, phase):
    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        task_file = cfg.shared_root / "tasks" / f"{task.task_id}.json"
        value = read_json(task_file)
        value["task"]["state"]["projection"] = phase
        atomic_replace(task_file, value)
        assert not supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
        assert not supervisor._reconcile_terminal_accounting(task.task_id, old.attempt_id)
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        assert (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
    finally:
        supervisor.close()


def test_terminal_current_number_supports_noncanonical_persisted_identity(tmp_path, monkeypatch):
    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        path = attempt_path(cfg.shared_root, task.task_id, 1)
        value = read_json(path)
        value["attempt"]["attempt_id"] = "persisted-custom-identity"
        atomic_replace(path, value)
        # Terminal transitions clear current_attempt_id, retaining the number.
        assert (
            read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")["task"]["attempt_control"][
                "current_attempt_id"
            ]
            is None
        )
        assert supervisor._has_terminal_attempt(task.task_id, "persisted-custom-identity")
        assert not supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
    finally:
        supervisor.close()


@pytest.mark.parametrize("identity_kind", ["opaque", "prefixed"])
@pytest.mark.parametrize("has_bounded_supervision", [False, True])
@pytest.mark.parametrize("has_evidence", [False, True])
@pytest.mark.parametrize("locator_number", [1, 2, 3])
def test_noncanonical_history_uses_recorded_locator_without_scanning_attempts(
    tmp_path, monkeypatch, has_evidence, locator_number, has_bounded_supervision, identity_kind
):
    from qqtools.plugins.qexp.commands.task import retry

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    identity = "historical-imported-execution" if identity_kind == "opaque" else f"{task.task_id}-attempt-imported"
    try:
        path = attempt_path(cfg.shared_root, task.task_id, 1)
        value = read_json(path)
        value["attempt"]["attempt_id"] = identity
        atomic_replace(path, value)
        archive = cfg.shared_root / "claims" / "archive" / task.task_id / f"{old.current_fencing_token}.json"
        value = read_json(archive)
        value["claim_archive"]["attempt_id"] = identity
        atomic_replace(archive, value)
        mark_writer_exited(cfg, old.attempt_id)
        for name, key in (("processes", "process"), ("registrations", "process_registration")):
            source = local_paths(cfg.runtime_root)[name] / f"{old.attempt_id}.json"
            if has_evidence:
                content = read_json(source)
                content[key]["attempt_id"] = identity
                atomic_replace(source.with_name(f"{identity}.json"), content)
            source.unlink()
        ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
        ledger.capture_local(identity, {"task_id": task.task_id, "attempt_number": locator_number})
        retry(cfg, task.task_id)
        successor = claim_task(cfg, task.task_id, [0])
        assert successor is not None and successor.attempt_number == 2
        original = os.scandir

        def no_attempt_inventory(path):
            if Path(path) == cfg.shared_root / "attempts" / task.task_id:
                raise AssertionError("historical cleanup enumerated the Attempt directory")
            return original(path)

        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", no_attempt_inventory)
            if has_bounded_supervision:
                supervisor.recover_startup()
                reader = supervisor._work._responsibilities
                supervisor_thread = threading.get_ident()
                original_read = DurableIO.read
                index_reads = []

                def background_read(self, *args, **kwargs):
                    assert threading.get_ident() != supervisor_thread, "index I/O ran on supervisor thread"
                    index_reads.append(args[0])
                    return original_read(self, *args, **kwargs)

                patch.setattr(DurableIO, "read", background_read)
                for _ in range(80):
                    supervisor._work._responsibility_step()
                    if reader._done is not None:
                        assert reader._done.wait(2)
                finish_reader(reader)
                assert index_reads
            else:
                assert supervisor._has_terminal_attempt(task.task_id, identity) == (locator_number == 1)
                assert supervisor._remove_terminal_attempt_evidence(identity) == (locator_number == 1)
        assert (ledger.find(identity) is None) == (locator_number == 1)
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 2))["attempt"]["phase"] == "claimed"
    finally:
        supervisor.close()


def mark_writer_exited(cfg, attempt_id):
    for name, key in (("registrations", "process_registration"), ("processes", "process")):
        path = local_paths(cfg.runtime_root)[name] / f"{attempt_id}.json"
        value = read_json(path)
        value[key].update(wrapper_pid=99999991, wrapper_start_time_ticks=1)
        atomic_replace(path, value)


@pytest.mark.parametrize("locator_number", [1, 2])
def test_late_evidence_completes_partial_prefetched_locator(tmp_path, monkeypatch, locator_number):
    from qqtools.plugins.qexp.commands.task import retry
    from qqtools.plugins.qexp.runtime.responsibility_backfill import ResponsibilityBackfill

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    identity = "partial-imported-history"
    try:
        mark_writer_exited(cfg, old.attempt_id)
        process_path = local_paths(cfg.runtime_root)["processes"] / f"{old.attempt_id}.json"
        process = read_json(process_path)
        process["process"]["attempt_id"] = identity
        process_path.unlink()
        (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").unlink()
        for path, key in (
            (attempt_path(cfg.shared_root, task.task_id, 1), "attempt"),
            (
                cfg.shared_root / "claims" / "archive" / task.task_id / f"{old.current_fencing_token}.json",
                "claim_archive",
            ),
        ):
            value = read_json(path)
            value[key]["attempt_id"] = identity
            atomic_replace(path, value)
        ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
        ledger.capture_local(identity, {"task_id": None, "attempt_number": locator_number})
        backfill = ResponsibilityBackfill(cfg.runtime_root)
        try:
            for _ in range(32):
                progress = backfill.take()
                if progress is not None and progress.is_sweep_complete:
                    break
            assert progress.is_sweep_complete
        finally:
            backfill.close()
        retry(cfg, task.task_id)
        successor = claim_task(cfg, task.task_id, [0])
        assert successor is not None and successor.attempt_number == 2
        # Completed lanes will not rescan this late write to fill the Task ID.
        late_path = process_path.with_name(f"{identity}.json")
        atomic_replace(late_path, process)
        supervisor.recover_startup()
        supervisor._failures[identity] = 1
        reader = supervisor._work._responsibilities
        for _ in range(80):
            supervisor._work._responsibility_step()
            if reader._done is not None:
                assert reader._done.wait(2)
        finish_reader(reader)
        assert (ledger.find(identity) is None) == (locator_number == 1)
        assert (not late_path.exists()) == (locator_number == 1)
        assert (identity not in supervisor._failures) == (locator_number == 1)
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 2))["attempt"]["phase"] == "claimed"
    finally:
        supervisor.close()


def cleanup_request(task_id, attempt_id):
    return CleanupRequest(
        attempt_id,
        {"task_id": task_id, "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": task_id, "attempt_id": attempt_id, "basis": "terminal_attempt"},
    )


def test_live_wrapper_prevents_cleanup_handoff(tmp_path, monkeypatch):
    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        supervisor.recover_startup()
        # Fixture registration records this still-live test process as wrapper.
        assert supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
        assert supervisor._cleanup_request(task.task_id, old.attempt_id) is None
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        assert not supervisor._work._responsibilities._cleanups
        assert (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
    finally:
        finish_reader(supervisor._work._responsibilities)
        supervisor.close()


def test_missing_task_cleanup_requires_completed_matching_tombstone(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands.cleanup import clean

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    try:
        # This case exercises an old tombstone after its writer has exited. Live
        # wrappers now retain evidence and delay the cleanup acknowledgement.
        mark_writer_exited(cfg, old.attempt_id)
        result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)
        assert result["operations"][task.task_id]["state"] == "completed"
        assert not (cfg.shared_root / "tasks" / f"{task.task_id}.json").exists()
        supervisor.recover_startup()
        assert supervisor._cleanup_request(task.task_id, old.attempt_id) is None
        observation = local_paths(cfg.runtime_root)["observations"] / f"{old.attempt_id}.json"
        atomic_replace(
            observation,
            {"exit_observation": {"task_id": task.task_id, "attempt_id": old.attempt_id, "observed_exit_code": 1}},
        )
        request = supervisor._cleanup_request(task.task_id, old.attempt_id)
        assert request is not None and request.receipt["basis"] == "completed_task_cleanup"
        tombstone = cfg.shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
        original = read_json(tombstone)
        for change in (
            {"state": "waiting_ack"},
            {"task_id": "other"},
            {"operation_id": None},
            {"acknowledgements": {}},
            {"required_machines": []},
        ):
            value = {**original, "cleanup": {**original["cleanup"], **change}}
            atomic_replace(tombstone, value)
            assert supervisor._cleanup_request(task.task_id, old.attempt_id) is None
        atomic_replace(tombstone, original)
        # Late evidence carries its original Task identity after Task deletion.
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        reader = supervisor._work._responsibilities
        assert reader._cleanups
        reader.take()
        assert reader._done.wait(3)
        assert not observation.exists()
        ledger = Ledger(responsibility_root(cfg.runtime_root))
        assert all(ledger.service_page(bucket) == ([], None) for bucket in range(16))
    finally:
        if supervisor._work is not None:
            finish_reader(supervisor._work._responsibilities)
        supervisor.close()


@pytest.mark.parametrize("has_bounded_supervision", [False, True])
@pytest.mark.parametrize("has_local_task_identity", [False, True])
def test_deleted_task_cleanup_handles_late_opaque_identity_without_inventing_number(
    tmp_path, monkeypatch, has_local_task_identity, has_bounded_supervision
):
    from qqtools.plugins.qexp.commands.cleanup import clean
    from qqtools.plugins.qexp.runner import _publish_exit_observation

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    identity = "late-opaque-legacy-attempt"
    try:
        mark_writer_exited(cfg, old.attempt_id)
        assert clean(cfg, task_id=task.task_id)["operations"][task.task_id]["state"] == "completed"
        assert not (cfg.shared_root / "tasks" / f"{task.task_id}.json").exists()
        ledger = Ledger(responsibility_root(cfg.runtime_root))
        ledger.capture_local(identity, {"task_id": task.task_id, "attempt_number": None})
        _publish_exit_observation(cfg, identity, 1, task_id=task.task_id if has_local_task_identity else None)
        path = local_paths(cfg.runtime_root)["observations"] / f"{identity}.json"

        if has_bounded_supervision:
            supervisor.recover_startup()
            request = supervisor._cleanup_request_for_membership(ledger.lookup(identity))
        else:
            request = supervisor._cleanup_request(task.task_id, identity)

        if has_local_task_identity:
            assert request is not None
            assert request.payload["attempt_number"] is None
            assert request.receipt["basis"] == "task_cleanup"
            assert request.receipt["operation_id"]
            assert supervisor._remove_terminal_attempt_evidence(identity)
            if has_bounded_supervision:
                reader = supervisor._work._responsibilities
                for _ in range(80):
                    supervisor._work._responsibility_step()
                    if reader._done is not None:
                        assert reader._done.wait(2)
                    if not path.exists():
                        break
                finish_reader(reader)
            assert not path.exists() and ledger.find(identity) is None
        else:
            assert request is None
            assert not supervisor._remove_terminal_attempt_evidence(identity)
            assert path.exists() and ledger.lookup(identity)["stage"] == "active"
    finally:
        supervisor.close()


def test_blocked_cleanup_cannot_block_supervision_or_recreate_manifest(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_cleanup

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    paths = local_paths(cfg.runtime_root)
    manifest = paths["processes"] / f"{old.attempt_id}.json"
    registration = paths["registrations"] / manifest.name
    for path, key in ((manifest, "process"), (registration, "process_registration")):
        value = read_json(path)
        value[key].update(wrapper_pid=99999991, wrapper_start_time_ticks=1)
        atomic_replace(path, value)
    entered, release = threading.Event(), threading.Event()
    original = responsibility_cleanup.remove_evidence_slice

    def blocked(root, attempt_id, io):
        entered.set()
        assert release.wait(5)
        return original(root, attempt_id, io)

    monkeypatch.setattr(responsibility_cleanup, "remove_evidence_slice", blocked)
    supervisor.recover_startup()
    reader = supervisor._work._responsibilities
    try:
        supervisor._remove_terminal_attempt_evidence(old.attempt_id)
        assert reader._cleanups
        reader.take()
        assert entered.wait(2)
        manifest.unlink()  # Model an already completed deletion in the blocked batch.
        supervisor._materialize_registration(registration)
        assert not manifest.exists(), "registration replay recreated evidence during cleanup"
        supervisor.tick()
        assert not release.is_set(), "supervision waited for the blocked cleanup worker"
        entry = Ledger(responsibility_root(cfg.runtime_root)).lookup(old.attempt_id)
        assert entry["cleanup_receipt"]["format"] == CLEANUP_FORMAT
    finally:
        release.set()
        finish_reader(reader)
        supervisor.close()
    assert not registration.exists()


def test_unknown_process_state_is_not_writer_quiescence(tmp_path, monkeypatch):
    from pathlib import Path

    from qqtools.plugins.qexp.runtime.responsibility_cleanup import writers_are_quiescent

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)

    def inaccessible(*_args, **_kwargs):
        raise PermissionError("process inventory unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", inaccessible)
            assert not writers_are_quiescent(cfg.runtime_root, task.task_id, old.attempt_id)
            assert supervisor._cleanup_request(task.task_id, old.attempt_id) is None
    finally:
        supervisor.close()


def test_failed_cleanup_keeps_unrelated_discovery_fair(tmp_path, monkeypatch):
    cfg, task, old = prepared(tmp_path)
    assert publish_launch(cfg, task.task_id, old)
    ledger = Ledger(responsibility_root(cfg.runtime_root))
    ledger.publish("other-attempt", {"task_id": "other", "attempt_number": 1})

    def unavailable(*_args):
        raise OSError("maintenance storage unavailable")

    monkeypatch.setattr("qqtools.plugins.qexp.runtime.responsibility.complete_cleanup", unavailable)
    reader = ResponsibilityReader(cfg.runtime_root)
    seen = set()
    try:
        for _ in range(40):
            reader.cleanup(cleanup_request(task.task_id, old.attempt_id))
            entry = reader.take()
            if entry is not None:
                seen.add(entry["identity"])
            if reader._done is not None:
                assert reader._done.wait(2)
            if "other-attempt" in seen:
                break
        assert "other-attempt" in seen
        assert reader.failures > 0
        assert ledger.lookup(old.attempt_id)["stage"] == "active"
    finally:
        finish_reader(reader)


@pytest.mark.parametrize("has_broken_maintenance", [False, True])
def test_active_initial_sweep_is_independent_of_maintenance_backlog(tmp_path, has_broken_maintenance):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ledger = Ledger.create(responsibility_root(runtime))
    names = [f"attempt-{number}" for number in range(3000) if identity_key(f"attempt-{number}")[0] == "0"][:131]
    assert len(names) == 131
    for identity in names:
        generation = ledger.publish(identity, {"task_id": None, "attempt_number": None})
        if identity != names[-1]:
            ledger.handoff(identity, generation)
    if has_broken_maintenance:
        (ledger.root / "0/m2").write_text("unreadable maintenance page")
    reader = ResponsibilityReader(runtime)
    seen = set()
    try:
        for _ in range(64):
            entry = reader.take()
            if entry is not None:
                seen.add(entry["identity"])
            if reader._done is not None:
                assert reader._done.wait(2)
            if reader._traversal is not None and reader._traversal.has_completed_initial_sweeps:
                break
        else:
            pytest.fail("active sweep waited for maintenance history")
        assert names[-1] in seen
        assert not reader._maintenance_traversal.has_completed_initial_sweeps
        if has_broken_maintenance:
            assert reader.failures > 0
    finally:
        finish_reader(reader)


def test_unbounded_cleanup_retains_live_writer_evidence(tmp_path, monkeypatch):
    cfg, task, old, original = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    original.close()
    supervisor = AuthoritySupervisor(cfg)
    try:
        supervisor.recover_startup()
        supervisor.tick()
        assert supervisor._work is None
        assert supervisor._has_terminal_attempt(task.task_id, old.attempt_id)
        assert (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
        assert not responsibility_root(cfg.runtime_root).exists()
        mark_writer_exited(cfg, old.attempt_id)

        supervisor.tick()

        assert not (local_paths(cfg.runtime_root)["registrations"] / f"{old.attempt_id}.json").exists()
        assert Ledger(responsibility_root(cfg.runtime_root)).find(old.attempt_id) is None
    finally:
        supervisor.close()


def test_unbounded_restart_replays_receipt_without_evidence_or_shared_truth(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO

    cfg, task, old, supervisor = terminal_attempt_with_retained_evidence(tmp_path, monkeypatch)
    mark_writer_exited(cfg, old.attempt_id)
    original_sync = DurableIO.sync_directory

    def fail_final_barrier(io, path, label):
        if label == "cleanup_evidence_directories":
            raise OSError("final barrier interrupted")
        return original_sync(io, path, label)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "sync_directory", fail_final_barrier)
        assert supervisor._remove_terminal_attempt_evidence(old.attempt_id)
    supervisor.close()
    ledger = Ledger(responsibility_root(cfg.runtime_root))
    assert ledger.lookup(old.attempt_id)["cleanup_receipt"]["basis"] == "terminal_attempt"
    # The diagnostic is the only remaining local file, not a recovery locator.
    paths = local_paths(cfg.runtime_root)
    diagnostic = paths["authority_diagnostics"] / f"{old.attempt_id}.json"
    assert read_json(diagnostic)["authority_diagnostic"]["reason"] == "terminal_cleanup_unavailable"
    diagnostic.unlink()
    assert not (paths["registrations"] / f"{old.attempt_id}.json").exists()
    assert not (paths["processes"] / f"{old.attempt_id}.json").exists()
    (cfg.shared_root / "tasks" / f"{task.task_id}.json").unlink()
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()

    # An old maintenance stage beside the receipt cannot authorize deletion.
    bare = ledger.publish("unproved", {"task_id": "missing-task", "attempt_number": 1})
    ledger.handoff("unproved", bare)
    resumed = AuthoritySupervisor(cfg)
    try:

        def unexpected(*_args, **_kwargs):
            pytest.fail("receipt replay consulted deleted Task truth")

        monkeypatch.setattr("qqtools.plugins.qexp.authority.load_task", unexpected)
        resumed.recover_startup()
        for _ in range(32):
            resumed.tick()
            if ledger.find(old.attempt_id) is None:
                break
        assert ledger.find(old.attempt_id) is None
        assert ledger.lookup("unproved")["stage"] == "maintenance"
    finally:
        resumed.close()


def test_unbounded_supervision_continues_with_an_unavailable_membership(tmp_path):
    from qqtools.plugins.qexp.runner import _publish_exit_observation, _publish_registration
    from qqtools.plugins.qexp.runtime.tasks import load_task

    cfg, task, attempt = prepared(tmp_path)
    _publish_registration(cfg, attempt, task, SimpleNamespace(pid=99999992))
    _publish_exit_observation(cfg, attempt.attempt_id, 0, task_id=task.task_id)
    root = responsibility_root(cfg.runtime_root)
    root.mkdir()
    (root / "marker").write_text("broken")
    supervisor = AuthoritySupervisor(cfg)
    try:
        supervisor.recover_startup()
        supervisor.tick()
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["attempt_id"] == attempt.attempt_id
        assert (root / "marker").read_text() == "broken"
        assert (local_paths(cfg.runtime_root)["registrations"] / f"{attempt.attempt_id}.json").exists()
    finally:
        supervisor.close()


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "no_exit",
        "legacy_exit",
        "wrong_attempt",
        "wrong_task",
        "wrong_protocol",
        "missing_code",
        "live_group",
        "partial_wrapper",
        "unknown_manifest",
    ],
)
def test_group_only_manifest_requires_final_matching_exit_and_absent_group(tmp_path, damage):
    import os

    from qqtools.plugins.qexp.runtime.responsibility_cleanup import writers_are_quiescent

    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    process = {
        "protocol_version": 1,
        "task_id": "task",
        "attempt_id": "attempt",
        "process_group_id": os.getpgrp() if damage == "live_group" else 99999991,
        "observed_state": "exited",
    }
    observation = {"protocol_version": 1, "task_id": "task", "attempt_id": "attempt", "observed_exit_code": 0}
    if damage == "legacy_exit":
        observation.pop("task_id")
    elif damage == "wrong_attempt":
        observation["attempt_id"] = "another-attempt"
    elif damage == "wrong_task":
        observation["task_id"] = "another-task"
    elif damage == "wrong_protocol":
        observation["protocol_version"] = 999
    elif damage == "missing_code":
        observation.pop("observed_exit_code")
    elif damage == "partial_wrapper":
        process["wrapper_pid"] = os.getpid()
    elif damage == "unknown_manifest":
        process["protocol_version"] = 999
    paths = local_paths(cfg.runtime_root)
    atomic_replace(paths["processes"] / "attempt.json", {"process": process})
    if damage != "no_exit":
        atomic_replace(paths["observations"] / "attempt.json", {"exit_observation": observation})

    assert writers_are_quiescent(cfg.runtime_root, "task", "attempt") is (damage in {None, "legacy_exit"})
