"""Progress keeps local cadence while actual worker processes own shared I/O."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import PROJECT_IO_STOP_GRACE_SECONDS
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.progress import prepare_progress_channel, shared_progress_path
from qqtools.plugins.qexp.runtime.progress_v2 import prepare_progress_v2_channel
from qqtools.plugins.qexp.runtime.progress_v3 import prepare_progress_v3_channel
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt
from qqtools.qexp._progress_protocol import read_advisory_snapshot, replace_advisory_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _case(tmp_path, runtime, name, version):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    cfg = replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])
    task = submit(cfg, ["true"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt = AttemptRecord.from_dict(read_json(path))
    attempt.phase = "running"
    attempt.process.update({"wrapper_pid": os.getpid(), "wrapper_start_time_ticks": 777})
    atomic_replace(path, attempt.to_dict())
    task = load_task(cfg, task.task_id)
    task.state.update({"projection": "running", "reason": "running"})
    task.claim_control["active_claim"]["launch_state"] = "running"
    task.meta["revision"] += 1
    save_task(cfg, task)
    if version == 1:
        channel = prepare_progress_channel(cfg, task, attempt, wrapper_start_time_ticks=777, interval_seconds=1)
        directory = "progress-contexts"
        shared = shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    else:
        prepare = prepare_progress_v2_channel if version == 2 else prepare_progress_v3_channel
        channel = prepare(cfg, task, attempt, wrapper_start_time_ticks=777, interval_seconds=1)
        directory = f"progress-v{version}-contexts"
        shared = cfg.shared_root / f"progress-v{version}" / task.task_id / f"{attempt.attempt_id}.json"
    assert channel is not None
    context_path = cfg.runtime_root / directory / f"{attempt.attempt_id}.json"
    context = read_advisory_snapshot(context_path)
    payload = {
        "protocol_version": version,
        "update_id": "update-1",
        "stage": "train",
        "current": 1,
        "total": 10,
        "unit": "step",
        "message": "first",
    }
    if version in {2, 3}:
        payload.update(
            {"metrics": {"loss": 0.5}, "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []}}
        )
    if version == 3:
        activity = {key: payload.pop(key) for key in ("stage", "current", "total", "unit", "message")}
        payload.update(activity=activity, overall={"current": 8, "total": 10, "unit": "step", "label": "Training"})
    replace_advisory_snapshot(Path(channel), payload)
    return cfg, binding, task, attempt, context, context_path, Path(channel), shared


def _consume(executor, request):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("isolated progress transaction did not finish")


@contextmanager
def _local_only(roots):
    prefixes = tuple(str(root) for root in roots)

    def check(path):
        if isinstance(path, int):
            return
        absolute = os.path.abspath(os.fsdecode(path))
        assert not any(absolute == root or absolute.startswith(root + os.sep) for root in prefixes), (
            f"local progress coordinator accessed shared path: {absolute}"
        )

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in [(Path, "open"), (os, "open"), (os, "stat"), (os, "lstat"), (os, "scandir")]:
            original = getattr(owner, name)

            def guarded(path, *args, _original=original, **kwargs):
                check(path)
                return _original(path, *args, **kwargs)

            patch.setattr(owner, name, guarded)
        yield


def _pump(controller, runtime, predicate, *, timeout=8, shared_roots=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        revision, bindings = runtime.load_registry_snapshot()
        runtime.working_set.reconcile(bindings, revision=revision)
        with _local_only(shared_roots):
            controller.advance_progress_work(bindings, revision)
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(
        "local sampling and shared progress transaction did not converge: "
        f"requests={[(item.operation_kind, item.project_id) for item in controller.executor.unresolved_requests()]}, "
        f"backoff={controller._service_backoff}, "
        f"local={[(owner.parameters, tuple(owner.projectors)) for owner in controller.progress._entries.values()]}"
    )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_real_worker_observes_without_machine_guards_or_local_writes(tmp_path, monkeypatch, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, _attempt, context, _context_path, _mailbox, _shared = _case(
        tmp_path, runtime, "project", version
    )
    revision, _bindings = runtime.load_registry_snapshot()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    local_before = {
        path.relative_to(cfg.runtime_root): path.read_bytes() for path in cfg.runtime_root.rglob("*") if path.is_file()
    }
    try:
        request = executor.prepare_progress_projection(binding, revision, context=context, projection=None)
        with runtime.registry_guard(), runtime.binding_commit_guard(binding):
            monkeypatch.setattr(
                MachineRuntime, "__init__", lambda *_args, **_kwargs: pytest.fail("worker constructed local owner")
            )
            evidence = worker._progress_projection(request, runtime.root, executor.paths, [False])
        assert evidence["state"] == "observed"
        assert evidence["binding"]["fencing_token"] > 0
        assert local_before == {
            path.relative_to(cfg.runtime_root): path.read_bytes()
            for path in cfg.runtime_root.rglob("*")
            if path.is_file()
        }
        assert executor.start(request.request_id) is not None
        result = _consume(executor, request)
        assert result.status == "completed"
        assert result.evidence["state"] == "observed"
    finally:
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_coordinator_preserves_cadence_and_consumes_exact_publication(tmp_path, monkeypatch, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, attempt, _context, _context_path, mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    clock = [0.0]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: clock[0])
    try:
        # The parent is forbidden to resolve any shared config or progress binding.
        from qqtools.plugins.qexp import layout
        from qqtools.plugins.qexp.runtime import progress, progress_projection

        def forbidden(*_args, **_kwargs):
            pytest.fail("local coordinator accessed shared Project I/O inline")

        monkeypatch.setattr(layout, "load_root_config", forbidden)
        monkeypatch.setattr(progress, "resolve_progress_binding", forbidden)
        monkeypatch.setattr(progress_projection, "resolve_progress_binding", forbidden)
        _pump(controller, runtime, shared.exists, shared_roots=(cfg.shared_root,))
        key = controller.progress._key(binding)
        owner = controller.progress._entries[key]
        projector = owner.projectors[version]
        _pump(
            controller,
            runtime,
            lambda: owner.parameters is None and projector._entries[attempt.attempt_id]["published"] is not None,
        )
        first = read_advisory_snapshot(shared)
        deadline = projector._entries[attempt.attempt_id]["cache_next_due"]
        assert deadline >= 1
        payload = read_advisory_snapshot(mailbox)
        payload["update_id"] = "update-2"
        (payload["activity"] if version == 3 else payload)["current"] = 2
        replace_advisory_snapshot(mailbox, payload)
        clock[0] = 0.5
        for _ in range(30):
            revision, bindings = runtime.load_registry_snapshot()
            with _local_only((cfg.shared_root,)):
                controller.advance_progress_work(bindings, revision)
            time.sleep(0.01)
        assert read_advisory_snapshot(shared) == first
        assert projector._entries[attempt.attempt_id]["cache_next_due"] == deadline
        clock[0] = deadline
        _pump(
            controller,
            runtime,
            lambda: read_advisory_snapshot(shared)["source_update_id"] == "update-2",
            shared_roots=(cfg.shared_root,),
        )
        second = read_advisory_snapshot(shared)
        assert second["sequence"] == first["sequence"] + 1
        assert (second["progress"]["activity"] if version == 3 else second["progress"])["current"] == 2
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_two_blocked_progress_workers_do_not_delay_peer_or_bounded_stop(tmp_path, monkeypatch, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, _attempt, context, _context_path, mailbox, shared = _case(
        tmp_path, runtime, "healthy", version
    )
    held = []
    for index in range(2):
        blocked_cfg = init_shared_root(tmp_path / f"blocked-{index}" / ".qexp", "gpu-1")
        blocked_binding = runtime.add_binding(blocked_cfg.shared_root, "gpu-1")
        root = runtime.project_paths(blocked_binding.project_id)["root"]
        directory = "progress-contexts" if version == 1 else f"progress-v{version}-contexts"
        (root / directory).mkdir(parents=True, exist_ok=True)
        replace_advisory_snapshot(root / directory / f"{context['attempt_id']}.json", context)
        local_mailbox = root / "progress" / context["attempt_id"] / mailbox.name
        local_mailbox.parent.mkdir(parents=True, exist_ok=True)
        replace_advisory_snapshot(local_mailbox, read_advisory_snapshot(mailbox))
        schema = blocked_cfg.shared_root / "schema/version.json"
        backup = schema.with_suffix(".held")
        schema.rename(backup)
        os.mkfifo(schema)
        held.append((blocked_binding, schema, backup))
    revision, bindings = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(bindings, revision=revision)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with _local_only([cfg.shared_root, *(item[0].shared_root for item in held)]):
                controller.advance_progress_work([item[0] for item in held], revision)
            if executor.status_view()["active_worker_count"] == 2:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("did not establish two real blocked progress workers")
        pending = executor.unresolved_requests()
        assert {item.project_id for item in pending} == {item[0].project_id for item in held}
        assert all(item.operation_kind == "progress_projection" for item in pending)
        started = time.monotonic()
        _pump(
            controller, runtime, shared.exists, shared_roots=[cfg.shared_root, *(item[0].shared_root for item in held)]
        )
        assert time.monotonic() - started < 8
        assert (
            read_advisory_snapshot(shared)["progress"]["activity"]
            if version == 3
            else read_advisory_snapshot(shared)["progress"]
        )["current"] == 1
        assert executor.status_view()["active_worker_count"] <= 4
        assert sum(item.project_id != binding.project_id for item in executor.unresolved_requests()) == 2
        started = time.monotonic()
        controller.progress.close()
        assert time.monotonic() - started < 0.5
        started = time.monotonic()
        executor.shutdown()
        # Public shutdown deliberately grants two separate two-second windows.
        assert time.monotonic() - started < 2 * PROJECT_IO_STOP_GRACE_SECONDS + 1
    finally:
        controller.progress.close()
        executor.shutdown()
        for _binding, schema, backup in held:
            schema.unlink()
            backup.rename(schema)


def _desired_projection(controller, runtime, binding):
    def ready():
        entry = controller.progress._entries.get(controller.progress._key(binding))
        return entry is not None and entry.parameters is not None and entry.parameters["projection"] is not None

    _pump(controller, runtime, ready)
    return dict(controller.progress._entries[controller.progress._key(binding)].parameters["projection"])


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_publication_replay_is_idempotent_and_worker_cannot_write_local_cache(tmp_path, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, _attempt, context, _context_path, _mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        desired = _desired_projection(controller, runtime, binding)
        local_before = {
            path.relative_to(cfg.runtime_root): path.read_bytes()
            for path in cfg.runtime_root.rglob("*")
            if path.is_file()
        }
        revision, _bindings = runtime.load_registry_snapshot()
        first = None
        for _ in range(2):
            request = executor.prepare_progress_projection(binding, revision, context=context, projection=desired)
            assert executor.start(request.request_id) is not None
            result = _consume(executor, request)
            assert result.status == "completed"
            assert result.evidence["state"] == "published"
            observed = shared.read_bytes(), shared.stat().st_mtime_ns
            if first is None:
                first = observed
            else:
                assert observed == first
            assert local_before == {
                path.relative_to(cfg.runtime_root): path.read_bytes()
                for path in cfg.runtime_root.rglob("*")
                if path.is_file()
            }
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_raw_replace_rechecks_epoch_without_acknowledging_local_cadence(tmp_path, monkeypatch, version):
    from qqtools.plugins.qexp.runtime import progress_projection

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, attempt, context, _context_path, _mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        desired = _desired_projection(controller, runtime, binding)
        owner = controller.progress._entries[controller.progress._key(binding)]
        cadence = owner.projectors[version]._entries[attempt.attempt_id]
        before = cadence["shared_next_due"]
        original = progress_projection.publish_progress_snapshot

        def revoked_publish(*args, **kwargs):
            def writer(path, value, **write_kwargs):
                callback = write_kwargs["before_replace"]

                def revoke():
                    executor.fence_epoch()
                    callback()

                write_kwargs["before_replace"] = revoke
                return replace_advisory_snapshot(path, value, **write_kwargs)

            return original(*args, **kwargs, writer=writer)

        monkeypatch.setattr(progress_projection, "publish_progress_snapshot", revoked_publish)
        revision, _bindings = runtime.load_registry_snapshot()
        request = executor.prepare_progress_projection(binding, revision, context=context, projection=desired)
        with pytest.raises(worker._ExecutorEpochFenced):
            worker._progress_projection(request, runtime.root, executor.paths, [False])
        assert not shared.exists()
        assert cadence["published"] is None
        assert cadence["shared_next_due"] == before
        assert list(shared.parent.glob(".*.json.*")) == []
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_disabled_owned_attempt_finishes_terminal_progress_and_retires_only_after_consumption(tmp_path, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, task, attempt, _context, context_path, _mailbox, shared = _case(tmp_path, runtime, "project", version)
    runtime.set_enabled(binding.project_id, False)
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _pump(controller, runtime, lambda: shared.exists() and not context_path.exists())
        observed = read_advisory_snapshot(shared)
        assert observed["attempt_id"] == attempt.attempt_id
        assert (observed["progress"]["activity"] if version == 3 else observed["progress"])["current"] == 1
        assert load_task(cfg, task.task_id).state["projection"] == "failed"
        assert load_task(cfg, task.task_id).attempt_control["current_attempt_number"] == attempt.attempt_number
        assert not (cfg.runtime_root / "progress-coordinator" / f"v{version}" / f"{attempt.attempt_id}.json").exists()
    finally:
        controller.progress.close()
        executor.shutdown()


def _add_idle_peers(tmp_path, runtime, count):
    for index in range(count):
        cfg = init_shared_root(tmp_path / f"idle-{index}" / ".qexp", "gpu-1")
        runtime.add_binding(cfg.shared_root, cfg.machine_name)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_scan_reaches_directory_tail_across_65_binding_evictions(tmp_path, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, attempt, context, context_path, _mailbox, shared = _case(tmp_path, runtime, "project", version)
    # Create an actual directory tail; every irrelevant name counts against the
    # bounded slice. Eviction must not restart this finite census.
    from qqtools.plugins.qexp.runtime.directory_capture import read_directory_entry

    context_path.parent.rename(context_path.parent.with_name("prepared-contexts"))
    context_path.parent.mkdir(exist_ok=True)
    replace_advisory_snapshot(context_path, context)

    def directory_names():
        names = []
        offset = 0
        while True:
            name, offset = read_directory_entry(context_path.parent, offset)
            if name is None:
                return names
            names.append(name)

    # Linux directory order need not be insertion or filename order. Populate
    # until the real channel is beyond four bounded visits, then keep that prefix.
    for batch in range(32):
        for index in range(batch * 40, (batch + 1) * 40):
            replace_advisory_snapshot(context_path.parent / f"irrelevant-{index}.json", {})
        names = directory_names()
        if names.index(context_path.name) >= 16:
            break
    else:
        raise AssertionError("could not construct a real directory tail")
    prefix = set(names[:16]) | {context_path.name}
    for name in names:
        if name not in prefix:
            (context_path.parent / name).unlink()
    names = directory_names()
    assert names.index(context_path.name) >= 16, names
    _add_idle_peers(tmp_path, runtime, 64)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        # This is finite 65-binding census evidence, not the single-peer latency
        # qualification. The real publication must still complete under a bound.
        _pump(controller, runtime, shared.exists, timeout=16, shared_roots=(cfg.shared_root,))
        assert read_advisory_snapshot(shared)["attempt_id"] == attempt.attempt_id
        assert len(controller.progress._entries) <= 64
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_sampling_deadline_survives_65_binding_eviction(tmp_path, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, attempt, _context, _context_path, mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    clock = [0.0]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: clock[0])
    try:
        _pump(controller, runtime, shared.exists)
        key = controller.progress._key(binding)
        owner = controller.progress._entries[key]
        _pump(controller, runtime, lambda: owner.parameters is None)
        cadence = owner.projectors[version]._entries[attempt.attempt_id]
        due = cadence["cache_next_due"]
        shared_due = cadence["shared_next_due"]
        first = read_advisory_snapshot(shared)
        _add_idle_peers(tmp_path, runtime, 64)
        clock[0] = 0.5
        _pump(controller, runtime, lambda: controller.progress._entries.get(key) is not owner)
        payload = read_advisory_snapshot(mailbox)
        payload["update_id"] = "update-2"
        (payload["activity"] if version == 3 else payload)["current"] = 2
        replace_advisory_snapshot(mailbox, payload)

        def restored():
            current = controller.progress._entries.get(key)
            if current is None or version not in current.projectors:
                return False
            state = current.projectors[version]._entries.get(attempt.attempt_id)
            return state is not None

        _pump(controller, runtime, restored)
        state = controller.progress._entries[key].projectors[version]._entries[attempt.attempt_id]
        assert state["cache_next_due"] == due
        assert state["shared_next_due"] == shared_due
        assert read_advisory_snapshot(shared) == first
        clock[0] = max(due, shared_due)
        _pump(controller, runtime, lambda: read_advisory_snapshot(shared)["source_update_id"] == "update-2")
        assert read_advisory_snapshot(shared)["sequence"] == first["sequence"] + 1
        assert len(controller.progress._entries) <= 64
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("invalid_checkpoint", ["clock_scope", "generation", "context", "deadline", "flag"])
def test_progress_eviction_rejects_foreign_or_malformed_cadence(tmp_path, version, invalid_checkpoint):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _task, attempt, context, _context_path, _mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    clock = [0.0]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: clock[0])
    try:
        _pump(controller, runtime, shared.exists)
        key = controller.progress._key(binding)
        owner = controller.progress._entries[key]
        _pump(controller, runtime, lambda: owner.parameters is None)
        path = controller.progress._cadence_path(owner, context)
        saved = read_advisory_snapshot(path)
        if invalid_checkpoint == "deadline":
            saved["cadence"]["cache_next_due"] = True
        elif invalid_checkpoint == "flag":
            saved["cadence"]["cache_initial_available"] = 1
        elif invalid_checkpoint == "context":
            saved["context"]["launch_id"] = "other-launch"
        else:
            saved[invalid_checkpoint] = "other-owner"
        saved["cadence"]["shared_next_due"] = 1000
        replace_advisory_snapshot(path, saved)
        assert controller.progress._park(owner)
        controller.progress._entries.pop(key)
        clock[0] = 0.5

        def restored():
            current = controller.progress._entries.get(key)
            return (
                current is not None
                and version in current.projectors
                and attempt.attempt_id in current.projectors[version]._entries
            )

        _pump(controller, runtime, restored)
        state = controller.progress._entries[key].projectors[version]._entries[attempt.attempt_id]
        assert 1.5 <= state["shared_next_due"] < 2.5
        assert state["published"] == read_advisory_snapshot(shared)
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_failed_cadence_checkpoint_retains_owner_until_storage_recovers(tmp_path, version):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _task, attempt, context, _context_path, _mailbox, shared = _case(
        tmp_path, runtime, "project", version
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _pump(controller, runtime, shared.exists)
        key = controller.progress._key(binding)
        owner = controller.progress._entries[key]
        _pump(controller, runtime, lambda: owner.parameters is None)
        state = owner.projectors[version]._entries[attempt.attempt_id]
        due = state["shared_next_due"]
        path = controller.progress._cadence_path(owner, context)
        parked_directory = path.parent.with_name(f"v{version}-retained")
        path.parent.rename(parked_directory)
        replace_advisory_snapshot(path.parent, {})
        controller.progress._save_cadence(owner, context)
        assert not owner.can_evict
        assert not controller.progress._park(owner)
        assert controller.progress._entries[key] is owner
        assert state["shared_next_due"] == due
        path.parent.unlink()
        parked_directory.rename(path.parent)
        controller.progress._save_cadence(owner, context)
        assert owner.can_evict
        assert read_advisory_snapshot(path)["cadence"]["shared_next_due"] == due
        assert controller.progress._park(owner)
    finally:
        controller.progress.close()
        executor.shutdown()


def test_progress_failed_scan_checkpoint_does_not_block_other_eviction_candidates(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _add_idle_peers(tmp_path, runtime, 65)
    revision, bindings = runtime.load_registry_snapshot()
    first_root = runtime.project_paths(bindings[0].project_id)["root"]
    directory = first_root / "progress-contexts"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(40):
        replace_advisory_snapshot(directory / f"irrelevant-{index}.json", {})
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    clock = [0.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: clock[0])
    try:
        for _ in range(4):
            controller.advance_progress_work(bindings, revision)
        assert len(controller.progress._entries) == 64
        key = controller.progress._key(bindings[0])
        owner = controller.progress._entries[key]
        assert owner.offsets[1] > 0
        checkpoint = controller.progress._checkpoint_root(owner) / "scan.json"
        checkpoint.mkdir(parents=True)
        controller.advance_progress_work(bindings, revision)
        assert controller.progress._entries[key] is owner
        assert owner.parking_retry_at == 1.0
        assert controller.progress._key(bindings[64]) in controller.progress._entries
        assert len(controller.progress._entries) == 64
        assert executor.unresolved_requests() == ()
        checkpoint.rmdir()
        clock[0] = 1.0
        assert controller.progress._park(owner)
        assert read_advisory_snapshot(checkpoint)["offsets"]["1"] == owner.offsets[1]
    finally:
        controller.progress.close()
        executor.shutdown()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_progress_local_cleanup_removes_parked_cadence_without_context(tmp_path, version):
    from qqtools.plugins.qexp.runtime.progress import cleanup_local_progress
    from qqtools.plugins.qexp.runtime.progress_v2 import cleanup_local_progress_v2

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, _binding, task, attempt, _context, context_path, _mailbox, _shared = _case(
        tmp_path, runtime, "project", version
    )
    context_path.unlink()
    path = cfg.runtime_root / "progress-coordinator" / f"v{version}" / f"{attempt.attempt_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_advisory_snapshot(path, {"cadence": {}})
    from qqtools.plugins.qexp.runtime.progress_cleanup_v3 import cleanup_local_progress_v3

    cleanup = {1: cleanup_local_progress, 2: cleanup_local_progress_v2, 3: cleanup_local_progress_v3}[version]
    removed = cleanup(cfg, task.task_id, {attempt.attempt_id})
    assert str(path) in removed
    assert not path.exists()
