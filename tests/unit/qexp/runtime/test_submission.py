from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.task import batch_submit, submit
from qqtools.plugins.qexp.runtime.submission import IdempotencyConflict


def test_bulk_submission_has_one_operation_and_no_batch_identity(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
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


def test_same_idempotency_key_reuses_resolved_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    first = submit(cfg, ["echo", "one"], idempotency_key="k")
    second = submit(cfg, ["echo", "one"], idempotency_key="k")
    assert first.task_id == second.task_id
    with pytest.raises(IdempotencyConflict):
        submit(cfg, ["echo", "two"], idempotency_key="k")
