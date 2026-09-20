from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.paths import idempotency_path, submission_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.submission import semantic_digest

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


class AbruptSubmissionInterruption(BaseException):
    """Simulate process loss without normal submission rollback."""


def test_sealed_group_does_not_poison_idempotency_key(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "first"], group="exp")
    from qqtools.plugins.qexp.commands.group import group_control

    group_control(cfg, "exp", "seal")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "second"], group="exp", idempotency_key="sealed-key")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "second"], group="exp", idempotency_key="sealed-key")
    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": "sealed-key"}))
    assert not mapping.exists()


def test_operation_without_mapping_is_not_discovered_or_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    key = "orphaned-operation"
    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    original_create = submission_runtime.create_if_absent

    def fail_before_mapping(path: Path, value: dict):
        if path == mapping:
            raise AbruptSubmissionInterruption
        return original_create(path, value)

    monkeypatch.setattr(submission_runtime, "create_if_absent", fail_before_mapping)
    with pytest.raises(AbruptSubmissionInterruption):
        submit(cfg, ["echo", "first"], idempotency_key=key)

    orphan_paths = list((cfg.shared_root / "operations" / "submissions").glob("*.json"))
    assert len(orphan_paths) == 1
    orphan = read_json(orphan_paths[0])["submission"]
    assert not mapping.exists()

    monkeypatch.setattr(submission_runtime, "create_if_absent", original_create)
    completed = submit(cfg, ["echo", "first"], idempotency_key=key)
    operation_paths = list((cfg.shared_root / "operations" / "submissions").glob("*.json"))

    assert len(operation_paths) == 2
    assert completed.submission_operation_id != orphan["operation_id"]
    assert completed.task_id != orphan["resolved_context"]["task_ids"][0]


def test_mapping_write_that_raises_after_success_is_replayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    key = "durable-mapping"
    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    original_create = submission_runtime.create_if_absent

    def fail_after_mapping(path: Path, value: dict):
        result = original_create(path, value)
        if path == mapping:
            raise AbruptSubmissionInterruption
        return result

    monkeypatch.setattr(submission_runtime, "create_if_absent", fail_after_mapping)
    with pytest.raises(AbruptSubmissionInterruption):
        submit(cfg, ["echo", "first"], idempotency_key=key)

    operation_id = read_json(mapping)["operation_id"]
    monkeypatch.setattr(submission_runtime, "create_if_absent", original_create)
    completed = submit(cfg, ["echo", "first"], idempotency_key=key)

    assert completed.submission_operation_id == operation_id
    assert len(list((cfg.shared_root / "operations" / "submissions").glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("operation_id", "different-operation"), ("idempotency_key", "different-key")],
)
def test_mapping_and_operation_identity_mismatch_fails_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    key = f"identity-{field}"
    original_execute = submission_runtime._execute_submission_locked

    def interrupt_before_execution(*_args, **_kwargs):
        raise AbruptSubmissionInterruption

    monkeypatch.setattr(submission_runtime, "_execute_submission_locked", interrupt_before_execution)
    with pytest.raises(AbruptSubmissionInterruption):
        submit(cfg, ["echo", "first"], idempotency_key=key)

    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    operation_id = read_json(mapping)["operation_id"]
    operation_file = submission_path(cfg.shared_root, operation_id)
    operation = read_json(operation_file)
    operation["submission"][field] = replacement
    atomic_replace(operation_file, operation)

    monkeypatch.setattr(submission_runtime, "_execute_submission_locked", original_execute)
    with pytest.raises(RuntimeError, match="does not match"):
        submit(cfg, ["echo", "first"], idempotency_key=key)

    assert not list((cfg.shared_root / "tasks").glob("*.json"))
