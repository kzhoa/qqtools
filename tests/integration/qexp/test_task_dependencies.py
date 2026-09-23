import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import pytest

from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import cancel, edit_dependencies, submit
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime import dependencies as dependency_runtime
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime import tasks as task_runtime
from qqtools.plugins.qexp.runtime.dependencies import dependency_gate, validate_group_dependencies
from qqtools.plugins.qexp.runtime.locks import group_lock
from qqtools.plugins.qexp.runtime.paths import task_path
from qqtools.plugins.qexp.runtime.records import TaskRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _group_config(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment", ["gpu-1"])
    return cfg


def _probe_group_lock(root: Path, group_name: str, connection) -> None:
    with group_lock(root, group_name, blocking=False) as acquired:
        connection.send(acquired)
    connection.close()


def _candidate(task: TaskRecord, task_id: str, dependencies: list[str]) -> TaskRecord:
    candidate = TaskRecord.from_dict(task.to_dict())
    candidate.task_id = task_id
    candidate.depends_on_task_ids = dependencies
    return candidate


def test_dependency_validation_reads_only_reachable_task_truth(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    unrelated = task_path(cfg.shared_root, "unrelated")
    unrelated.write_text("not json", encoding="utf-8")

    without_dependencies = _candidate(parent, "independent", [])
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics):
        validate_group_dependencies(cfg, "experiment", [without_dependencies])

    assert diagnostics.counters["store.iter_json.calls"] == 0
    assert diagnostics.counters["task_json_read.records"] == 0

    with_dependency = _candidate(parent, "child", [parent.task_id])
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics):
        validate_group_dependencies(cfg, "experiment", [with_dependency])

    assert diagnostics.counters["store.iter_json.calls"] == 0
    assert diagnostics.counters["task_json_read.records"] == 1


def test_dependency_validation_detects_reachable_multi_hop_cycle(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    tail = submit(cfg, ["echo", "tail"], task_id="tail", group="experiment", working_dir=tmp_path)
    middle = submit(
        cfg,
        ["echo", "middle"],
        task_id="middle",
        group="experiment",
        working_dir=tmp_path,
        depends_on_task_ids=[tail.task_id],
    )
    tail.depends_on_task_ids = ["candidate"]
    atomic_replace(task_path(cfg.shared_root, tail.task_id), tail.to_dict())
    candidate = _candidate(middle, "candidate", [middle.task_id])

    with pytest.raises(ValueError, match="cycle"):
        validate_group_dependencies(cfg, "experiment", [candidate])


def test_dependency_validation_rejects_mismatched_task_identity(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    atomic_replace(task_path(cfg.shared_root, "alias"), parent.to_dict())
    candidate = _candidate(parent, "candidate", ["alias"])

    with pytest.raises(ValueError, match="dependency Task 'alias' does not exist"):
        validate_group_dependencies(cfg, "experiment", [candidate])


def test_dependency_validation_handles_large_relevant_closure_iteratively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _group_config(tmp_path)
    seed = submit(cfg, ["echo", "seed"], task_id="seed", group="experiment", working_dir=tmp_path)
    records = {
        f"node-{index:04d}": _candidate(
            seed,
            f"node-{index:04d}",
            [] if index == 1_099 else [f"node-{index + 1:04d}"],
        )
        for index in range(1_100)
    }
    reads: list[str] = []

    def read_reachable(_cfg, task_id: str) -> TaskRecord:
        reads.append(task_id)
        return records[task_id]

    monkeypatch.setattr(task_runtime, "load_task", read_reachable)
    monkeypatch.setattr(dependency_runtime, "is_committed_submission_task", lambda *_args: True)
    monkeypatch.setattr(dependency_runtime, "operation_exists", lambda *_args: False)
    candidate = _candidate(seed, "candidate", ["node-0000"])

    validate_group_dependencies(cfg, "experiment", [candidate])

    assert len(reads) == 1_100
    assert len(set(reads)) == 1_100


def test_dependency_gate_blocks_claim_and_reports_cancelled_parent(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="experiment",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    assert claim_task(cfg, child.task_id, [0]) is None
    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "waiting"

    cancel(cfg, parent.task_id)
    gate = dependency_gate(cfg, load_task(cfg, child.task_id))
    assert gate.state == "blocked"
    assert gate.reasons == ({"task_id": parent.task_id, "reason": "cancelled"},)
    assert claim_task(cfg, child.task_id, [0]) is None


def test_batch_dependencies_are_checked_as_one_graph_and_edits_are_guarded(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    child = submit(cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path)

    updated = edit_dependencies(cfg, child.task_id, [parent.task_id], action="add")
    assert updated.depends_on_task_ids == [parent.task_id]
    with pytest.raises(ValueError, match="cycle"):
        edit_dependencies(cfg, parent.task_id, [child.task_id])

    assert claim_task(cfg, child.task_id, [0]) is None


def test_cleanup_refuses_task_retained_by_a_downstream_dependency(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="experiment",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    cancel(cfg, parent.task_id)

    with pytest.raises(ValueError, match="referenced by: child"):
        clean(cfg, task_id=parent.task_id)


def test_dependency_submission_holds_group_lock_through_task_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    original_save = submission_runtime.save_task
    observed_locks: list[bool] = []

    def observe_save(cfg_value, task):
        if task.task_id == "child":
            context = get_context("fork")
            receiving, sending = context.Pipe(duplex=False)
            process = context.Process(
                target=_probe_group_lock,
                args=(cfg_value.shared_root, "experiment", sending),
            )
            process.start()
            sending.close()
            assert receiving.poll(5)
            observed_locks.append(receiving.recv())
            process.join(5)
            assert process.exitcode == 0
        return original_save(cfg_value, task)

    monkeypatch.setattr(submission_runtime, "save_task", observe_save)
    submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="experiment",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    assert observed_locks == [False]


def test_submission_callback_runs_outside_group_lock(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    observed_locks: list[bool] = []

    def observe_callback(_operation_id: str, _key: str) -> None:
        context = get_context("fork")
        receiving, sending = context.Pipe(duplex=False)
        process = context.Process(target=_probe_group_lock, args=(cfg.shared_root, "experiment", sending))
        process.start()
        sending.close()
        assert receiving.poll(5)
        observed_locks.append(receiving.recv())
        process.join(5)
        assert process.exitcode == 0

    submission_runtime.submit_specs(
        cfg,
        [{"task_id": "child", "command": ["echo", "child"], "depends_on_task_ids": []}],
        group_name="experiment",
        on_prepared=observe_callback,
    )

    assert observed_locks == [True]


def test_dependency_rejection_precedes_callback(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    callback_calls = 0

    def observe_callback(_operation_id: str, _key: str) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(ValueError, match="dependency"):
        submission_runtime.submit_specs(
            cfg,
            [
                {
                    "task_id": "child",
                    "command": ["echo", "child"],
                    "depends_on_task_ids": ["missing-parent"],
                }
            ],
            group_name="experiment",
            on_prepared=observe_callback,
        )

    assert callback_calls == 0


def test_dependencies_are_rechecked_after_callback_mutation(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    callback_calls = 0

    def remove_parent(_operation_id: str, _key: str) -> None:
        nonlocal callback_calls
        callback_calls += 1
        cancel(cfg, parent.task_id)
        clean(cfg, task_id=parent.task_id)

    with pytest.raises(ValueError, match="dependency"):
        submission_runtime.submit_specs(
            cfg,
            [
                {
                    "task_id": "child",
                    "command": ["echo", "child"],
                    "depends_on_task_ids": [parent.task_id],
                }
            ],
            group_name="experiment",
            on_prepared=remove_parent,
        )

    assert callback_calls == 1
    with pytest.raises(FileNotFoundError):
        load_task(cfg, "child")


def test_dependency_edits_require_activated_dependency_capability(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    child = submit(cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path)
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-dependencies-v1")
    atomic_replace(schema_path, schema)

    with pytest.raises(ValueError, match="activated task-dependencies-v1"):
        edit_dependencies(cfg, child.task_id, [parent.task_id])


def test_uncommitted_task_cannot_be_referenced_or_edited(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path)
    operation_path = cfg.shared_root / "operations" / "submissions" / (f"{parent.submission_operation_id}.json")
    operation = read_json(operation_path)
    operation["submission"]["state"] = "committing"
    atomic_replace(operation_path, operation)

    with pytest.raises(ValueError, match="has not been committed"):
        submit(
            cfg,
            ["echo", "child"],
            task_id="child",
            group="experiment",
            working_dir=tmp_path,
            depends_on_task_ids=[parent.task_id],
        )
    with pytest.raises(ValueError, match="submission is not committed"):
        edit_dependencies(cfg, parent.task_id, [])


def test_legacy_dependency_submission_replay_is_rejected(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    specs = [{"task_id": "legacy", "command": ["echo", "legacy"], "depends_on_task_ids": []}]
    first = submission_runtime.submit_specs(cfg, specs, group_name="experiment", idempotency_key="legacy-key")
    operation_path = cfg.shared_root / "operations" / "submissions" / f"{first.operation_id}.json"
    operation = read_json(operation_path)
    del operation["submission"]["resolved_context"]["task_specs"][0]["depends_on_task_ids"]
    context = operation["submission"]["resolved_context"]
    operation["submission"]["resolved_context_digest"] = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()
    atomic_replace(operation_path, operation)

    with pytest.raises(RuntimeError, match="predates the canonical task-dependencies-v1 protocol"):
        submission_runtime.submit_specs(cfg, specs, group_name="experiment", idempotency_key="legacy-key")


def test_legacy_dependency_idempotency_digest_is_rejected(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    specs = [{"task_id": "legacy", "command": ["echo", "legacy"], "depends_on_task_ids": []}]
    first = submission_runtime.submit_specs(cfg, specs, group_name="experiment", idempotency_key="legacy-key")
    operation_path = cfg.shared_root / "operations" / "submissions" / f"{first.operation_id}.json"
    operation = read_json(operation_path)
    operation["submission"]["raw_request_digest"] = submission_runtime.semantic_digest(
        {
            "group": "experiment",
            "tasks": [{"task_id": "legacy", "command": ["echo", "legacy"], "home_machine": "current"}],
            "worker_set": {},
        }
    )
    atomic_replace(operation_path, operation)

    with pytest.raises(submission_runtime.IdempotencyConflict, match="different semantic input"):
        submission_runtime.submit_specs(cfg, specs, group_name="experiment", idempotency_key="legacy-key")
