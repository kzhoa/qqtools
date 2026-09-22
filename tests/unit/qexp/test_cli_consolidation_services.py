from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp.commands.configuration import reset_config, show_config
from qqtools.plugins.qexp.commands.operation import (
    OperationReferenceError,
    create_operation_reference,
    inspect_operation,
)
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.launch_policy import (
    launch_handoff_policy_path,
    reset_launch_handoff_policy,
    set_launch_handoff_policy,
)
from qqtools.plugins.qexp.progress_policy import progress_policy_path, reset_progress_policy, set_progress_policy
from qqtools.plugins.qexp.runtime.operation_store import active_operation_path, archive_operation
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.tmux_policy import reset_tmux_policy, set_tmux_policy, tmux_policy_path


def _cfg(tmp_path: Path) -> RootConfig:
    root = tmp_path / ".qexp"
    (root / "locks").mkdir(parents=True)
    return RootConfig(root, tmp_path, "gpu-1", tmp_path / "runtime")


@pytest.mark.parametrize(
    ("setter", "resetter", "path_getter", "value", "default_key", "default_value"),
    [
        (set_progress_policy, reset_progress_policy, progress_policy_path, 7, "interval_seconds", 30),
        (set_tmux_policy, reset_tmux_policy, tmux_policy_path, True, "enabled", False),
        (
            set_launch_handoff_policy,
            reset_launch_handoff_policy,
            launch_handoff_policy_path,
            22,
            "timeout_seconds",
            10,
        ),
    ],
)
def test_project_policy_reset_removes_override_and_is_idempotent(
    tmp_path: Path, setter, resetter, path_getter, value, default_key, default_value
) -> None:
    cfg = _cfg(tmp_path)
    setter(cfg, value)
    assert path_getter(cfg).exists()

    first = resetter(cfg)
    second = resetter(cfg)

    assert not path_getter(cfg).exists()
    assert first["source"] == "default"
    assert first[default_key] == default_value
    assert second == first


def test_reset_refuses_to_discard_corrupt_policy(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    atomic_replace(progress_policy_path(cfg), {"version": 1, "interval_seconds": 0})

    with pytest.raises(ValueError, match="malformed progress policy"):
        reset_progress_policy(cfg)

    assert progress_policy_path(cfg).exists()


def test_aggregate_config_show_reports_each_malformed_section(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    atomic_replace(progress_policy_path(cfg), {"version": 1, "interval_seconds": 0})

    result = show_config(None, cfg=cfg, runtime=object())

    assert result["complete"] is False
    assert result["sections"]["progress"]["status"] == "error"
    assert result["sections"]["tmux"]["status"] == "ok"
    assert "agent" not in result["sections"]


def test_agent_config_reset_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be reset"):
        reset_config("agent", cfg=None, runtime=object())


def _group_cancel_record(operation_id: str, state: str = "blocked") -> dict:
    return {
        "meta": {"schema_version": 6, "revision": 1},
        "group_control": {
            "operation_id": operation_id,
            "operation_type": "cancel",
            "group_name": "experiment",
            "state": state,
            "progress": {"target_tasks": 3},
            "pending_machine_acknowledgements": {"gpu-2": ["task-1"]},
            "blocked_reason": "waiting_for_machine",
        },
    }


def test_operation_reference_survives_archival_and_blocked_is_successful_read(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    operation_id = "operation-1"
    reference = create_operation_reference(cfg, "group_cancel", operation_id, operation_id)
    path = active_operation_path(cfg, "group_control", operation_id)
    atomic_replace(path, _group_cancel_record(operation_id))

    active, active_exit = inspect_operation(cfg, reference)
    archive_operation(cfg, "group_control", operation_id, _group_cancel_record(operation_id))
    archived, archived_exit = inspect_operation(cfg, reference)

    assert active_exit == archived_exit == 0
    assert active["state"] == archived["state"] == "blocked"
    assert active["pending_machines"] == ["gpu-2"]
    assert archived["operation_id"] == operation_id


def test_operation_inspection_preserves_timestamps_and_lifecycle_outcome(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    operation_id = "operation-timestamps"
    reference = create_operation_reference(cfg, "group_cancel", operation_id, operation_id)
    record = _group_cancel_record(operation_id)
    record["meta"].update(
        {
            "created_at": "2026-09-22T10:00:00Z",
            "updated_at": "2026-09-22T10:01:00Z",
        }
    )
    atomic_replace(active_operation_path(cfg, "group_control", operation_id), record)

    result, exit_code = inspect_operation(cfg, reference)

    assert exit_code == 0
    assert result["outcome"] == "ok"
    assert result["lifecycle_outcome"] == "blocked"
    assert result["created_at"] == "2026-09-22T10:00:00Z"
    assert result["updated_at"] == "2026-09-22T10:01:00Z"
    assert result["next_action"] == f"qexp admin operation show {reference} --project {cfg.project_root}"


def test_operation_reference_rejects_wrong_project_before_record_lookup(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    reference = create_operation_reference(cfg, "group_cancel", "operation-1", "operation-1")
    payload = json.loads(base64.urlsafe_b64decode(reference + "=" * (-len(reference) % 4)))
    payload["project"] = "other-project"
    changed = (
        base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.commands.operation.active_operation_path",
        lambda *_args, **_kwargs: pytest.fail("record I/O occurred before project validation"),
    )

    with pytest.raises(OperationReferenceError) as error:
        inspect_operation(cfg, changed)

    assert error.value.exit_code == 2
    assert error.value.code == "invalid_reference"


def test_missing_operation_is_not_reported_as_expired(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    reference = create_operation_reference(cfg, "cleanup", "task-1", "operation-1")

    result, exit_code = inspect_operation(cfg, reference)

    assert exit_code == 1
    assert result["state"] is None
    assert result["error"]["code"] == "not_found"
    assert "expired" not in result["error"]["message"].lower()


@pytest.mark.parametrize(
    "reference",
    [
        "x" * 4097,
        "not+urlsafe",
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "v": 1,
                    "project": "project",
                    "kind": "cleanup",
                    "key": "../task",
                    "operation_id": "operation-1",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        .decode()
        .rstrip("="),
    ],
)
def test_operation_reference_rejects_oversized_noncanonical_and_traversal_before_io(
    tmp_path: Path, monkeypatch, reference: str
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.commands.operation.active_operation_path",
        lambda *_args, **_kwargs: pytest.fail("record I/O occurred before reference validation"),
    )

    with pytest.raises(OperationReferenceError) as error:
        inspect_operation(cfg, reference)

    assert error.value.exit_code == 2
    assert error.value.code == "invalid_reference"
