from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.paths import group_path, idempotency_path, submission_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_cross_machine_idempotency_reuses_original_home_and_workers(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    create_group(cfg, "exp")
    first = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover", idempotency_key="same")
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    second = submit(other, ["echo", "ok"], group="exp", sharing_mode="spillover", idempotency_key="same")
    assert second.task_id == first.task_id
    assert second.placement_policy["home_machine"] == "g1"


class AbruptSubmissionInterruption(BaseException):
    """Simulate process loss without entering the normal rollback path."""


def _mapped_operation(cfg: RootConfig, key: str) -> dict:
    digest = submission_runtime.semantic_digest({"project": str(cfg.shared_root), "key": key})
    mapping = read_json(idempotency_path(cfg.shared_root, digest))
    return read_json(submission_path(cfg.shared_root, mapping["operation_id"]))


def test_callback_mutation_cannot_change_the_frozen_submission_decision(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    command = ["echo", "before"]
    dependencies: list[str] = []
    specs = [{"command": command, "depends_on_task_ids": dependencies}]

    def mutate_request(_operation_id: str, _key: str) -> None:
        command[1] = "after"
        dependencies.append("late-parent")
        specs[0]["home_machine"] = "other"

    result = submission_runtime.submit_specs(cfg, specs, on_prepared=mutate_request)

    assert result[0].spec.command == ["echo", "before"]
    assert result[0].depends_on_task_ids == []
    assert result[0].placement_policy["home_machine"] == "g1"


def test_unfinished_replay_does_not_reenter_planning_or_retry_machine_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_directory = tmp_path / "first-cwd"
    second_directory = tmp_path / "second-cwd"
    first_directory.mkdir()
    second_directory.mkdir()
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    specs = [{"command": ["echo", "ok"], "depends_on_task_ids": []}]
    key = "unfinished-cross-machine"
    monkeypatch.chdir(first_directory)

    def interrupt(_operation_id: str, _key: str) -> None:
        raise AbruptSubmissionInterruption

    with pytest.raises(AbruptSubmissionInterruption):
        submission_runtime.submit_specs(cfg, specs, idempotency_key=key, on_prepared=interrupt)

    operation = _mapped_operation(cfg, key)["submission"]
    planned_id = operation["resolved_context"]["task_ids"][0]
    planned_spec = operation["resolved_context"]["task_specs"][0]
    assert planned_spec["working_directory"] == str(first_directory)

    def planning_is_forbidden(*_args, **_kwargs):
        raise AssertionError("replay re-entered initial submission planning")

    monkeypatch.setattr(submission_runtime, "prepare_submission_plan", planning_is_forbidden)
    monkeypatch.chdir(second_directory)
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    replayed = submission_runtime.submit_specs(other, specs, idempotency_key=key)

    assert replayed.operation_id == operation["operation_id"]
    assert replayed[0].task_id == planned_id
    assert replayed[0].spec.working_directory == str(first_directory)
    assert replayed[0].placement_policy["home_machine"] == "g1"


def test_new_group_replay_uses_the_original_submitting_machine_as_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    specs = [{"command": ["echo", "ok"], "depends_on_task_ids": []}]
    key = "new-group-owner"
    original_execute = submission_runtime._execute_submission_locked

    def interrupt_before_execution(*_args, **_kwargs):
        raise AbruptSubmissionInterruption

    monkeypatch.setattr(submission_runtime, "_execute_submission_locked", interrupt_before_execution)
    with pytest.raises(AbruptSubmissionInterruption):
        submission_runtime.submit_specs(
            cfg,
            specs,
            group_name="new-group",
            idempotency_key=key,
            kind="bulk",
            worker_set=["g1"],
        )
    assert not group_path(cfg.shared_root, "new-group").exists()

    monkeypatch.setattr(submission_runtime, "_execute_submission_locked", original_execute)
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    replayed = submission_runtime.submit_specs(
        other,
        specs,
        group_name="new-group",
        idempotency_key=key,
        kind="bulk",
        worker_set=["g1"],
    )

    group = read_json(group_path(cfg.shared_root, "new-group"))
    assert replayed[0].placement_policy["home_machine"] == "g1"
    assert group["meta"]["updated_by"]["machine_name"] == "g1"
