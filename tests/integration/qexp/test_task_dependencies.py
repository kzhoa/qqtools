from pathlib import Path
from multiprocessing import get_context

import pytest

from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import cancel, edit_dependencies, submit
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime.dependencies import dependency_gate
from qqtools.plugins.qexp.runtime.locks import group_lock
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import claim_task
from qqtools.plugins.qexp.runtime import submission as submission_runtime


pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _group_config(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment", ["gpu-1"])
    return cfg


def _probe_group_lock(root: Path, group_name: str, connection) -> None:
    with group_lock(root, group_name, blocking=False) as acquired:
        connection.send(acquired)
    connection.close()


def test_dependency_gate_blocks_claim_and_reports_cancelled_parent(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
    child = submit(
        cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path,
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
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
    child = submit(
        cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path
    )

    updated = edit_dependencies(cfg, child.task_id, [parent.task_id], action="add")
    assert updated.depends_on_task_ids == [parent.task_id]
    with pytest.raises(ValueError, match="cycle"):
        edit_dependencies(cfg, parent.task_id, [child.task_id])

    assert claim_task(cfg, child.task_id, [0]) is None


def test_cleanup_refuses_task_retained_by_a_downstream_dependency(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
    submit(
        cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    cancel(cfg, parent.task_id)

    with pytest.raises(ValueError, match="referenced by: child"):
        clean(cfg, task_id=parent.task_id)


def test_dependency_submission_holds_group_lock_through_task_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
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
        cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    assert observed_locks == [False]


def test_dependency_edits_require_activated_dependency_capability(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
    child = submit(
        cfg, ["echo", "child"], task_id="child", group="experiment", working_dir=tmp_path
    )
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-dependencies-v1")
    atomic_replace(schema_path, schema)

    with pytest.raises(ValueError, match="activated task-dependencies-v1"):
        edit_dependencies(cfg, child.task_id, [parent.task_id])


def test_uncommitted_task_cannot_be_referenced_or_edited(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    parent = submit(
        cfg, ["echo", "parent"], task_id="parent", group="experiment", working_dir=tmp_path
    )
    operation_path = cfg.shared_root / "operations" / "submissions" / (
        f"{parent.submission_operation_id}.json"
    )
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


def test_empty_dependency_field_replays_a_legacy_idempotency_digest(tmp_path: Path) -> None:
    cfg = _group_config(tmp_path)
    specs = [{"task_id": "legacy", "command": ["echo", "legacy"], "depends_on_task_ids": []}]
    first = submission_runtime.submit_specs(
        cfg, specs, group_name="experiment", idempotency_key="legacy-key"
    )
    operation_path = cfg.shared_root / "operations" / "submissions" / f"{first.operation_id}.json"
    operation = read_json(operation_path)
    normalized = {
        "group": "experiment",
        "tasks": submission_runtime._canonical_specs(specs),
        "worker_set": {},
    }
    operation["submission"]["raw_request_digest"] = (
        submission_runtime._legacy_empty_dependencies_digest(normalized)
    )
    del operation["submission"]["resolved_context"]["task_specs"][0]["depends_on_task_ids"]
    atomic_replace(operation_path, operation)

    replay = submission_runtime.submit_specs(
        cfg, specs, group_name="experiment", idempotency_key="legacy-key"
    )

    assert replay.operation_id == first.operation_id
    assert read_json(operation_path)["submission"]["resolved_context"]["task_specs"][0][
        "depends_on_task_ids"
    ] == []
