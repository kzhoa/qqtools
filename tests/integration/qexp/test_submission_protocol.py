from pathlib import Path
from threading import Event, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import batch_submit, retry, submit
from qqtools.plugins.qexp.runtime.locks import schema_lock
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.submission import IdempotencyConflict
from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _activate_narrow_submission_protocol(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert that a newly initialized root has the real joint activation evidence."""
    del monkeypatch
    from qqtools.plugins.qexp.runtime.locks import is_schema_narrow_protocol_active

    assert is_schema_narrow_protocol_active(cfg)


def test_bulk_submission_has_one_operation_and_no_batch_identity(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, one]\n  - command: [echo, two]\n", encoding="utf-8")
    tasks = batch_submit(cfg, manifest, group="exp")
    assert len(tasks) == 2
    assert all(task.group_name == "exp" for task in tasks)
    assert tasks.operation_id
    assert tasks.idempotency_key
    assert tasks.target_group == "exp"
    assert tasks.state == "committed"
    assert tasks.to_dict()["task_ids"] == [task.task_id for task in tasks]
    assert not list((cfg.shared_root / "groups").glob("*.batch.json"))


def test_bulk_submission_announces_random_operation_before_task_staging(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, one]\n", encoding="utf-8")
    announced: list[tuple[str, str]] = []

    def observe_prepared(operation_id: str, key: str) -> None:
        announced.append((operation_id, key))
        assert not list((cfg.shared_root / "tasks").glob("*.json"))

    first = batch_submit(cfg, manifest, on_prepared=observe_prepared)
    second = batch_submit(cfg, manifest)
    assert announced[0][0] == first[0].submission_operation_id
    assert announced[0][1]
    assert first[0].submission_operation_id != second[0].submission_operation_id


def test_active_protocol_rollback_removes_only_its_operation_owned_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        "group:\n  workers: [gpu-1]\ntasks:\n  - command: [echo, one]\n",
        encoding="utf-8",
    )
    prepared: list[str] = []

    def fail_after_prepare(operation_id: str, _key: str) -> None:
        prepared.append(operation_id)
        raise OSError("task storage unavailable")

    with pytest.raises(OSError, match="task storage unavailable"):
        batch_submit(cfg, manifest, group="new-group", on_prepared=fail_after_prepare)

    assert prepared
    assert not group_path(cfg.shared_root, "new-group").exists()
    operation = read_json(submission_path(cfg.shared_root, prepared[0]))["submission"]
    assert operation["state"] == "aborted"
    assert operation["resolved_context"]["create_group"] is True
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


def test_same_idempotency_key_reuses_resolved_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    first = submit(cfg, ["echo", "one"], idempotency_key="k")
    second = submit(cfg, ["echo", "one"], idempotency_key="k")
    assert first.task_id == second.task_id
    with pytest.raises(IdempotencyConflict):
        submit(cfg, ["echo", "two"], idempotency_key="k")


# QQTOOLS-COMPAT-0009: active member-projection roots use shared schema readers.
def test_active_lock_protocol_allows_independent_submission_fences_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    both_entered = Event()
    release = Event()
    first_entered = Event()
    second_entered = Event()

    def hold_fence(digest: str, entered: Event) -> None:
        from qqtools.plugins.qexp.runtime.submission import _submission_protocol_lock

        with _submission_protocol_lock(cfg, digest):
            entered.set()
            if first_entered.is_set() and second_entered.is_set():
                both_entered.set()
            assert release.wait(timeout=2)

    first = Thread(target=hold_fence, args=("first", first_entered))
    second = Thread(target=hold_fence, args=("second", second_entered))
    first.start()
    second.start()
    assert both_entered.wait(timeout=1)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()


# QQTOOLS-COMPAT-0009: the keyed fence makes concurrent same-key submissions converge.
def test_active_lock_protocol_serializes_same_idempotency_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    entered = Event()
    release_first = Event()
    results = []
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("tasks:\n  - command: [echo, one]\n", encoding="utf-8")

    def first_prepared(_operation_id: str, _key: str) -> None:
        entered.set()
        assert release_first.wait(timeout=2)

    def run_first() -> None:
        results.append(batch_submit(cfg, manifest, idempotency_key="same", on_prepared=first_prepared))

    def run_second() -> None:
        results.append(batch_submit(cfg, manifest, idempotency_key="same"))

    first = Thread(target=run_first)
    second = Thread(target=run_second)
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert results[0][0].task_id == results[1][0].task_id


# QQTOOLS-COMPAT-0009: same-key conflicting requests preserve the winner's mapping.
def test_active_lock_protocol_rejects_conflicting_same_key_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    first_ready = Event()
    release_first = Event()
    failures: list[BaseException] = []

    def first_prepared(_operation_id: str, _key: str) -> None:
        first_ready.set()
        assert release_first.wait(timeout=5)

    first = Thread(
        target=lambda: batch_submit(
            cfg,
            _write_manifest(tmp_path, "first", "one"),
            idempotency_key="conflict",
            on_prepared=first_prepared,
        )
    )

    def submit_conflict() -> None:
        try:
            batch_submit(
                cfg,
                _write_manifest(tmp_path, "second", "two"),
                idempotency_key="conflict",
            )
        except BaseException as exc:
            failures.append(exc)

    first.start()
    assert first_ready.wait(timeout=2)
    conflicting = Thread(target=submit_conflict)
    conflicting.start()
    release_first.set()
    first.join(timeout=5)
    conflicting.join(timeout=5)

    assert not first.is_alive()
    assert not conflicting.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], IdempotencyConflict)


# QQTOOLS-COMPAT-0009: cleanup tombstones win a Task-identity creation race.
def test_active_protocol_cleanup_tombstone_blocks_concurrent_task_id_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    task = submit(cfg, ["echo", "old"], task_id="tombstoned-task")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    cleanup_started = Event()
    release_cleanup = Event()
    reuse_rejected = Event()
    failures: list[BaseException] = []
    from qqtools.plugins.qexp.commands import cleanup as cleanup_runtime

    original_start = cleanup_runtime._start_cleanup_operation

    def pause_after_tombstone(*args, **kwargs):
        result = original_start(*args, **kwargs)
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        return result

    monkeypatch.setattr(cleanup_runtime, "_start_cleanup_operation", pause_after_tombstone)
    cleaner = Thread(target=lambda: clean(cfg, task_id=task.task_id))
    cleaner.start()
    assert cleanup_started.wait(timeout=2)

    def reuse_task_id() -> None:
        try:
            submit(cfg, ["echo", "new"], task_id=task.task_id)
        except BaseException as exc:
            failures.append(exc)
            reuse_rejected.set()

    submitter = Thread(target=reuse_task_id)
    submitter.start()
    assert reuse_rejected.wait(timeout=2)
    release_cleanup.set()
    cleaner.join(timeout=5)
    submitter.join(timeout=5)

    assert not cleaner.is_alive()
    assert not submitter.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "cleaned" in str(failures[0])


# QQTOOLS-COMPAT-0009: Group addition and retry retain Group -> Task serialization.
def test_active_protocol_serializes_group_worker_addition_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "retry"], task_id="retry-task", group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    failures: list[BaseException] = []

    def run(callable_) -> None:
        try:
            callable_()
        except BaseException as exc:
            failures.append(exc)

    worker_change = Thread(
        target=run,
        args=(lambda: change_worker(cfg, "exp", "gpu-2", "add"),),
    )
    retry_task = Thread(target=run, args=(lambda: retry(cfg, task.task_id),))
    worker_change.start()
    retry_task.start()
    worker_change.join(timeout=5)
    retry_task.join(timeout=5)

    assert not worker_change.is_alive()
    assert not retry_task.is_alive()
    assert not failures
    group = read_json(group_path(cfg.shared_root, "exp"))
    retried = read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")
    assert group["group"]["worker_set"]["gpu-2"]["state"] == "active"
    assert retried["task"]["state"]["projection"] == "queued"


# QQTOOLS-COMPAT-0009: an exclusive activation or migration cannot overtake a narrow writer.
def test_active_protocol_schema_mutation_waits_for_submission_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    _activate_narrow_submission_protocol(cfg, monkeypatch)
    writer_entered = Event()
    release_writer = Event()
    mutation_entered = Event()

    def hold_writer() -> None:
        from qqtools.plugins.qexp.runtime.submission import _submission_protocol_lock

        with _submission_protocol_lock(cfg, "activation-race"):
            writer_entered.set()
            assert release_writer.wait(timeout=5)

    def mutate_schema() -> None:
        with schema_lock(cfg.shared_root):
            mutation_entered.set()

    writer = Thread(target=hold_writer)
    writer.start()
    assert writer_entered.wait(timeout=2)
    mutation = Thread(target=mutate_schema)
    mutation.start()
    assert not mutation_entered.wait(timeout=0.1)
    release_writer.set()
    writer.join(timeout=5)
    mutation.join(timeout=5)

    assert not writer.is_alive()
    assert not mutation.is_alive()
    assert mutation_entered.is_set()


def _write_manifest(tmp_path: Path, name: str, command: str) -> Path:
    manifest = tmp_path / f"{name}.yaml"
    manifest.write_text(f"tasks:\n  - command: [echo, {command}]\n", encoding="utf-8")
    return manifest
