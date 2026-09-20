from pathlib import Path

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path, task_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_group_submission_stage_failure_retains_aborted_membership_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    first_manifest = tmp_path / "first.yaml"
    first_manifest.write_text(
        "tasks:\n  - task_id: a\n    command: [echo, a]\n  - task_id: b\n    command: [echo, b]\n",
        encoding="utf-8",
    )
    prepared: list[str] = []

    def capture_prepared(operation_id: str, _key: str) -> None:
        prepared.append(operation_id)

    original_save_task = submission_runtime.save_task
    save_calls = 0

    def fail_on_second_stage(cfg_arg: object, task: object) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise OSError("injected stage failure")
        original_save_task(cfg_arg, task)

    monkeypatch.setattr(submission_runtime, "save_task", fail_on_second_stage)
    with pytest.raises(OSError, match="injected stage failure"):
        batch_submit(cfg, first_manifest, group="exp", on_prepared=capture_prepared)

    assert prepared
    operation_id = prepared[0]
    operation = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    assert operation["state"] == "aborted"
    assert operation["commit_plan"]["group_membership_sequences"] == [1, 2]
    assert not task_path(cfg.shared_root, "a").exists()
    assert not task_path(cfg.shared_root, "b").exists()
    group = read_json(group_path(cfg.shared_root, "exp"))["group"]
    assert group["next_membership_sequence"] == 1
    assert group["pending_submission_commit"] is None

    monkeypatch.setattr(submission_runtime, "save_task", original_save_task)
    second_manifest = tmp_path / "second.yaml"
    second_manifest.write_text(
        "tasks:\n  - task_id: c\n    command: [echo, c]\n  - task_id: d\n    command: [echo, d]\n",
        encoding="utf-8",
    )
    second = batch_submit(cfg, second_manifest, group="exp")

    assert second.state == "committed"
    assert [task.task_id for task in second] == ["c", "d"]
    assert [task.group_membership_sequence for task in second] == [1, 2]
    assert second.operation_id != operation_id
    second_operation = read_json(submission_path(cfg.shared_root, second.operation_id))["submission"]
    assert second_operation["state"] == "committed"
    assert second_operation["commit_plan"]["group_membership_sequences"] == [1, 2]
    retained = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    assert retained["state"] == "aborted"
    group = read_json(group_path(cfg.shared_root, "exp"))["group"]
    assert group["next_membership_sequence"] == 3
    assert group["pending_submission_commit"] is None


def test_group_submission_finalizer_failure_reuses_committed_tasks_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "tasks:\n  - task_id: a\n    command: [echo, a]\n  - task_id: b\n    command: [echo, b]\n",
        encoding="utf-8",
    )
    prepared: list[str] = []

    def capture_prepared(operation_id: str, _key: str) -> None:
        prepared.append(operation_id)

    original_finalize = submission_runtime.finalize_submission_group

    def fail_finalizer(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected finalizer failure")

    monkeypatch.setattr(submission_runtime, "finalize_submission_group", fail_finalizer)
    with pytest.raises(OSError, match="injected finalizer failure"):
        batch_submit(
            cfg,
            manifest,
            group="exp",
            idempotency_key="retained-committed-batch",
            on_prepared=capture_prepared,
        )

    assert prepared
    operation_id = prepared[0]
    operation = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    assert operation["state"] == "committed"
    assert operation["resolved_context"]["task_ids"] == ["a", "b"]
    assert operation["commit_plan"]["group_membership_sequences"] == [1, 2]
    for task_id, sequence in zip(("a", "b"), (1, 2)):
        task = load_task(cfg, task_id)
        assert task.submission_operation_id == operation_id
        assert task.group_membership_sequence == sequence
    group = read_json(group_path(cfg.shared_root, "exp"))["group"]
    assert group["next_membership_sequence"] == 1
    assert group["pending_submission_commit"]["operation_id"] == operation_id

    monkeypatch.setattr(submission_runtime, "finalize_submission_group", original_finalize)
    reissued = batch_submit(
        cfg,
        manifest,
        group="exp",
        idempotency_key="retained-committed-batch",
    )
    assert reissued.operation_id == operation_id
    assert [task.task_id for task in reissued] == ["a", "b"]
    assert [task.group_membership_sequence for task in reissued] == [1, 2]
    operation = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    assert operation["resolved_context"]["task_ids"] == [task.task_id for task in reissued]
    assert operation["commit_plan"]["group_membership_sequences"] == [
        task.group_membership_sequence for task in reissued
    ]
    group = read_json(group_path(cfg.shared_root, "exp"))["group"]
    assert group["next_membership_sequence"] == 3
    assert group["pending_submission_commit"] is None

    task_paths = sorted((cfg.shared_root / "tasks").glob("*.json"))
    repeated = batch_submit(
        cfg,
        manifest,
        group="exp",
        idempotency_key="retained-committed-batch",
    )
    assert repeated.operation_id == operation_id
    assert [task.task_id for task in repeated] == ["a", "b"]
    assert [task.group_membership_sequence for task in repeated] == [1, 2]
    assert sorted((cfg.shared_root / "tasks").glob("*.json")) == task_paths
    group = read_json(group_path(cfg.shared_root, "exp"))["group"]
    assert group["next_membership_sequence"] == 3
    assert group["pending_submission_commit"] is None
