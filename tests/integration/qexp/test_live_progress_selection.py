from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.group_observation_policy import set_group_policy
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.task_live_progress import read_task_live_progress

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _batch():
    return [
        {"command": ["echo", "inherited"]},
        {"command": ["echo", "explicit"], "live_progress": False},
    ]


def test_group_default_is_frozen_once_and_same_key_replay_never_resamples(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    create_group(cfg, "training")
    set_group_policy(cfg.shared_root, "training", True)

    first = submission_runtime.submit_specs(cfg, _batch(), group_name="training", idempotency_key="same", kind="bulk")
    operation = read_json(submission_path(cfg.shared_root, first.operation_id))
    selection = operation["live_progress_selection"]["tasks"]
    assert [item["enabled"] for item in selection] == [True, False]
    assert [item["source"] for item in selection] == ["group", "explicit"]
    assert selection[0]["group_policy"]["revision"] == 1
    assert all(read_task_live_progress(cfg, task.task_id) == expected for task, expected in zip(first, [True, False]))

    set_group_policy(cfg.shared_root, "training", False)
    replay = submission_runtime.submit_specs(cfg, _batch(), group_name="training", idempotency_key="same", kind="bulk")
    assert replay.operation_id == first.operation_id
    assert (
        read_json(submission_path(cfg.shared_root, first.operation_id))["live_progress_selection"]
        == operation["live_progress_selection"]
    )

    later = submission_runtime.submit_specs(cfg, _batch(), group_name="training", idempotency_key="later", kind="bulk")
    later_selection = read_json(submission_path(cfg.shared_root, later.operation_id))["live_progress_selection"][
        "tasks"
    ]
    assert later_selection[0]["enabled"] is False
    assert later_selection[0]["group_policy"]["revision"] == 2


def test_invalid_advisory_selection_disables_only_observation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    result = submission_runtime.submit_specs(cfg, [{"command": ["echo", "ok"], "live_progress": True}])
    assert read_task_live_progress(cfg, result[0].task_id)
    path = submission_path(cfg.shared_root, result.operation_id)
    operation = read_json(path)
    operation["live_progress_selection"]["selection_digest"] = "0" * 64
    atomic_replace(path, operation)
    assert read_task_live_progress(cfg, result[0].task_id) is False


def test_explicit_selection_skips_policy_read_and_survives_unavailable_default(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    create_group(cfg, "training")

    def unavailable(root, name, timeout=0.1):
        return {"status": "unavailable", "reason": "read timed out"}

    monkeypatch.setattr(submission_runtime, "read_policy_snapshot", unavailable)
    result = submission_runtime.submit_specs(
        cfg,
        [{"command": ["echo", "inherited"]}, {"command": ["echo", "explicit"], "live_progress": True}],
        group_name="training",
        kind="bulk",
    )
    selection = read_json(submission_path(cfg.shared_root, result.operation_id))["live_progress_selection"]["tasks"]
    assert [entry["enabled"] for entry in selection] == [False, True]
    assert [entry["source"] for entry in selection] == ["unavailable", "explicit"]

    def forbidden(root, name, timeout=0.1):
        raise AssertionError("an explicit batch must not read Group policy")

    monkeypatch.setattr(submission_runtime, "read_policy_snapshot", forbidden)
    explicit = submission_runtime.submit_specs(
        cfg,
        [{"command": ["echo", "only"], "live_progress": True}],
        group_name="training",
    )
    assert read_task_live_progress(cfg, explicit[0].task_id)


def test_same_key_replay_does_not_read_policy_and_explicit_change_conflicts(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    create_group(cfg, "training")
    first = submission_runtime.submit_specs(
        cfg, [{"command": ["echo", "ok"], "live_progress": True}], group_name="training", idempotency_key="same"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("same-key replay must not sample Group policy")

    monkeypatch.setattr(submission_runtime, "read_policy_snapshot", forbidden)
    replay = submission_runtime.submit_specs(
        cfg, [{"command": ["echo", "ok"], "live_progress": True}], group_name="training", idempotency_key="same"
    )
    assert replay.operation_id == first.operation_id
    with pytest.raises(ValueError, match="[Ii]dempotency|[Cc]onflict|different"):
        submission_runtime.submit_specs(
            cfg,
            [{"command": ["echo", "ok"], "live_progress": False}],
            group_name="training",
            idempotency_key="same",
        )


class _InterruptedAfterPrepare(BaseException):
    pass


def test_preparing_recovery_on_other_machine_keeps_frozen_policy(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    create_group(cfg, "training")
    set_group_policy(cfg.shared_root, "training", True)

    def interrupt(_operation_id, _key):
        raise _InterruptedAfterPrepare

    with pytest.raises(_InterruptedAfterPrepare):
        submission_runtime.submit_specs(
            cfg,
            [{"command": ["echo", "ok"]}],
            group_name="training",
            idempotency_key="recovery",
            on_prepared=interrupt,
        )
    set_group_policy(cfg.shared_root, "training", False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepared Operation recovery must not read Group policy")

    monkeypatch.setattr(submission_runtime, "read_policy_snapshot", forbidden)
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    recovered = submission_runtime.submit_specs(
        other,
        [{"command": ["echo", "ok"]}],
        group_name="training",
        idempotency_key="recovery",
    )
    assert read_task_live_progress(other, recovered[0].task_id) is True


def test_missing_historical_selection_is_off_without_resampling(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    result = submission_runtime.submit_specs(cfg, [{"command": ["echo", "ok"], "live_progress": True}])
    path = submission_path(cfg.shared_root, result.operation_id)
    operation = read_json(path)
    del operation["live_progress_selection"]
    atomic_replace(path, operation)
    with pytest.warns(RuntimeWarning, match="historical"):
        assert read_task_live_progress(cfg, result[0].task_id) is False
