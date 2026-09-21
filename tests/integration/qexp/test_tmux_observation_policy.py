from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.runtime.paths import submission_path, task_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.submission import IdempotencyConflict
from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt
from qqtools.plugins.qexp.task_observation import resolve_task_tmux_observation
from qqtools.plugins.qexp.tmux_policy import set_tmux_policy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _metadata(cfg, task):
    operation = read_json(submission_path(cfg.shared_root, task.submission_operation_id))
    return operation["task_observation"]


def test_single_override_is_durable_idempotent_and_outside_task_spec(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    first = submit(cfg, ["echo", "ok"], task_id="visible", tmux_override=True, idempotency_key="same")
    second = submit(cfg, ["echo", "ok"], task_id="visible", tmux_override=True, idempotency_key="same")

    assert second.task_id == first.task_id
    assert _metadata(cfg, first) == {
        "version": 1,
        "tasks": [{"task_id": "visible", "tmux_override": True}],
    }
    stored = read_json(task_path(cfg.shared_root, first.task_id))
    assert "tmux_override" not in stored["task"]["spec"]
    assert inspect_task(cfg, first.task_id)["observation"] == {"tmux_override": "enabled"}

    before = stored
    with pytest.raises(IdempotencyConflict):
        submit(cfg, ["echo", "ok"], task_id="visible", tmux_override=False, idempotency_key="same")
    assert read_json(task_path(cfg.shared_root, first.task_id)) == before


def test_omitted_and_null_override_have_the_same_identity(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    first = submit(cfg, ["echo", "ok"], task_id="inherited", idempotency_key="same")
    second = submit(
        cfg,
        ["echo", "ok"],
        task_id="inherited",
        tmux_override=None,
        idempotency_key="same",
    )

    assert second.task_id == first.task_id
    assert inspect_task(cfg, first.task_id)["observation"] == {"tmux_override": "inherit"}


def test_batch_precedence_preserves_false_and_does_not_change_group(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "study")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(
        """
defaults:
  tmux: false
tasks:
  - task_id: task-explicit-false
    tmux: false
    command: [echo, one]
  - task_id: task-cli-default
    tmux: null
    command: [echo, two]
  - task_id: task-explicit-true
    tmux: true
    command: [echo, three]
""",
        encoding="utf-8",
    )

    tasks = batch_submit(cfg, manifest, group="study", tmux_override=True)
    assert _metadata(cfg, tasks[0])["tasks"] == [
        {"task_id": "task-explicit-false", "tmux_override": False},
        {"task_id": "task-cli-default", "tmux_override": True},
        {"task_id": "task-explicit-true", "tmux_override": True},
    ]
    group = read_json(cfg.shared_root / "groups" / "study.json")["group"]
    assert "tmux" not in group
    assert "tmux_override" not in group


@pytest.mark.parametrize("value", ["true", 1, 0, [], {}])
def test_manifest_rejects_non_boolean_tmux_before_mutation(tmp_path: Path, value):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    manifest = tmp_path / "runs.yaml"
    manifest.write_text(f"tasks:\n  - tmux: {value!r}\n    command: [echo, ok]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tmux"):
        batch_submit(cfg, manifest)
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


def test_explicit_override_bypasses_policy_and_inheritance_resamples_it(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    explicit = submit(cfg, ["echo", "explicit"], task_id="explicit", tmux_override=False)
    inherited = submit(cfg, ["echo", "inherited"], task_id="inherited")
    set_tmux_policy(cfg, True)

    assert resolve_task_tmux_observation(cfg, explicit.task_id)["enabled"] is False
    assert resolve_task_tmux_observation(cfg, explicit.task_id)["source"] == "task_override"
    assert resolve_task_tmux_observation(cfg, inherited.task_id)["enabled"] is True

    set_tmux_policy(cfg, False)
    assert resolve_task_tmux_observation(cfg, inherited.task_id)["enabled"] is False


def test_retry_retains_explicit_override_and_inherited_task_uses_current_policy(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    explicit = submit(cfg, ["echo", "explicit"], task_id="explicit-retry", tmux_override=True)
    inherited = submit(cfg, ["echo", "inherited"], task_id="inherited-retry")

    for task in (explicit, inherited):
        attempt = claim_task(cfg, task.task_id, [0])
        assert attempt is not None
        assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test failure")
        retry(cfg, task.task_id)

    set_tmux_policy(cfg, False)
    assert resolve_task_tmux_observation(cfg, explicit.task_id)["enabled"] is True
    assert resolve_task_tmux_observation(cfg, inherited.task_id)["enabled"] is False


def test_malformed_override_disables_without_falling_through_to_enabled_policy(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["echo", "ok"], task_id="damaged", tmux_override=True)
    set_tmux_policy(cfg, True)
    path = submission_path(cfg.shared_root, task.submission_operation_id)
    operation = read_json(path)
    operation["task_observation"]["tasks"][0]["tmux_override"] = "true"
    atomic_replace(path, operation)

    decision = resolve_task_tmux_observation(cfg, task.task_id)
    assert decision["enabled"] is False
    assert decision["diagnostic_reason"]


def test_historical_operation_without_metadata_inherits(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["echo", "ok"], task_id="historical")
    path = submission_path(cfg.shared_root, task.submission_operation_id)
    operation = read_json(path)
    del operation["task_observation"]
    atomic_replace(path, operation)
    set_tmux_policy(cfg, True)

    assert resolve_task_tmux_observation(cfg, task.task_id)["enabled"] is True
    assert inspect_task(cfg, task.task_id)["observation"] == {"tmux_override": "inherit"}


def test_historical_idempotency_digest_replays_only_as_inherit(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["echo", "ok"], task_id="legacy-replay", idempotency_key="same")
    path = submission_path(cfg.shared_root, task.submission_operation_id)
    operation = read_json(path)
    del operation["task_observation"]
    legacy_request = {
        "group": None,
        "tasks": [
            {
                "task_id": "legacy-replay",
                "name": None,
                "command": ["echo", "ok"],
                "requested_gpus": 1,
                "requested_cpus": None,
                "working_directory": str(Path.cwd()),
                "home_machine": "current",
                "sharing_mode": "private",
                "fallback_machines": "group",
                "offer_after_seconds": None,
                "depends_on_task_ids": [],
            }
        ],
        "worker_set": {},
    }
    operation["submission"]["raw_request_digest"] = hashlib.sha256(
        json.dumps(legacy_request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_replace(path, operation)

    replayed = submit(cfg, ["echo", "ok"], task_id="legacy-replay", idempotency_key="same")
    assert replayed.task_id == task.task_id
    with pytest.raises(IdempotencyConflict):
        submit(cfg, ["echo", "ok"], task_id="legacy-replay", tmux_override=True, idempotency_key="same")
