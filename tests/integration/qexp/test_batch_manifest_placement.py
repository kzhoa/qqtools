from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root
from qqtools.plugins.qexp.commands.group import change_worker
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "runs.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_nested_task_private_overrides_spillover_defaults(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
defaults:
  placement:
    sharing:
      mode: spillover
      fallback_machines: group
      offer:
        after_seconds: 600
tasks:
  - name: private-control
    placement:
      sharing:
        mode: private
    command: [echo, ok]
""",
    )

    task = batch_submit(cfg, manifest, group="exp")[0]

    assert task.placement_policy["sharing_mode"] == "private"
    assert task.placement_policy["fallback_constraint"] == "group"
    assert task.placement_policy["offer_after_seconds"] is None
    assert task.placement_runtime["offer_eligible_at"] is None


def test_tasks_can_use_different_nested_placement_in_one_group(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
group:
  workers: [g2, g3]
defaults:
  placement:
    home_machine: g2
    sharing:
      mode: spillover
      fallback_machines: group
tasks:
  - name: shared
    command: [echo, shared]
  - name: private
    placement:
      home_machine: current
      sharing:
        mode: private
    command: [echo, private]
  - name: constrained
    placement:
      sharing:
        fallback_machines: [g3]
    command: [echo, constrained]
""",
    )

    shared, private, constrained = batch_submit(cfg, manifest, group="exp")

    assert shared.placement_policy["home_machine"] == "g2"
    assert shared.placement_policy["sharing_mode"] == "spillover"
    assert private.placement_policy["home_machine"] == "g1"
    assert private.placement_policy["sharing_mode"] == "private"
    assert constrained.placement_policy["home_machine"] == "g2"
    assert constrained.placement_policy["fallback_constraint"] == ["g3"]
    group = read_json(group_path(cfg.shared_root, "exp"))
    assert set(group["group"]["worker_set"]) == {"g1", "g2", "g3"}


def test_current_home_and_generated_task_ids_are_frozen_for_idempotency(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    manifest = _manifest(
        tmp_path,
        """
tasks:
  - placement:
      home_machine: current
    command: [echo, ok]
""",
    )

    first = batch_submit(cfg, manifest, group="exp", idempotency_key="same")[0]
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    second = batch_submit(other, manifest, group="exp", idempotency_key="same")[0]

    assert second.task_id == first.task_id
    assert second.placement_policy["home_machine"] == "g1"


def test_group_workers_allow_new_home_and_are_recorded_in_resolved_context(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
group:
  workers: [g2]
tasks:
  - placement:
      home_machine: g2
    command: [echo, ok]
""",
    )

    task = batch_submit(cfg, manifest, group="exp", idempotency_key="workers")[0]
    operation = read_json(submission_path(cfg.shared_root, task.submission_operation_id))

    assert task.placement_policy["home_machine"] == "g2"
    assert operation["submission"]["resolved_context"]["worker_set_additions"] == ["g2"]
    assert operation["submission"]["resolved_context"]["planned_worker_set"] == ["g1", "g2"]


def test_worker_set_epoch_changes_only_when_a_worker_is_added(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    first = _manifest(tmp_path, "tasks:\n  - command: [echo, one]\n")

    batch_submit(cfg, first, group="exp")
    group = read_json(group_path(cfg.shared_root, "exp"))
    assert group["group"]["worker_set_epoch"] == 1
    assert group["group"]["worker_set"]["g1"]["state_epoch"] == 1

    second = _manifest(tmp_path, "tasks:\n  - command: [echo, two]\n")
    batch_submit(cfg, second, group="exp")
    group = read_json(group_path(cfg.shared_root, "exp"))
    assert group["group"]["worker_set_epoch"] == 1


def test_rejects_unknown_manifest_fields_with_yaml_path(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
defaults:
  placement:
    workers: [g2]
tasks:
  - command: [echo, ok]
""",
    )

    with pytest.raises(ValueError, match=r"defaults\.placement\.workers is not allowed"):
        batch_submit(cfg, manifest, group="exp")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("workers: [g2]\ntasks:\n  - command: [echo, ok]\n", r"root\.workers is not allowed"),
        ("group:\n  name: exp\ntasks:\n  - command: [echo, ok]\n", r"group\.name is not allowed"),
        ("group:\n  workers: [g2]\ntasks:\n  - command: [echo, ok]\n", r"manifest group requires --group"),
        ("tasks: {}\n", r"tasks must be a non-empty list"),
        ("tasks:\n  - placement: []\n    command: [echo, ok]\n", r"tasks\[0\]\.placement must be a mapping"),
    ],
)
def test_manifest_allow_list_and_type_errors(tmp_path: Path, body: str, message: str):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    with pytest.raises(ValueError, match=message):
        batch_submit(cfg, _manifest(tmp_path, body), group=None)


def test_flat_task_fields_work_with_deprecation_warning(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
tasks:
  - name: legacy
    sharing_mode: spillover
    fallback_machines: group
    offer_after_seconds: null
    command: [echo, ok]
""",
    )

    with pytest.warns(FutureWarning, match="deprecated flat placement fields"):
        task = batch_submit(cfg, manifest, group="exp")[0]

    assert task.placement_policy["sharing_mode"] == "spillover"


def test_rejects_flat_and_nested_duplicate_semantic_fields(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
tasks:
  - sharing_mode: spillover
    placement:
      sharing:
        mode: spillover
    command: [echo, ok]
""",
    )

    with pytest.raises(ValueError, match=r"tasks\[0\] declares placement\.sharing\.mode and sharing_mode"):
        batch_submit(cfg, manifest, group="exp")


def test_rejects_task_private_with_task_level_fallback_or_offer(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
tasks:
  - placement:
      sharing:
        mode: private
        fallback_machines: group
    command: [echo, ok]
""",
    )

    with pytest.raises(ValueError, match=r"tasks\[0\] declares private sharing with fallback or offer"):
        batch_submit(cfg, manifest, group="exp")


def test_rejects_defaults_private_with_fallback_or_offer(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
defaults:
  placement:
    sharing:
      mode: private
      fallback_machines: group
tasks:
  - command: [echo, ok]
""",
    )

    with pytest.raises(ValueError, match="defaults.placement declares private sharing with fallback or offer"):
        batch_submit(cfg, manifest, group="exp")


def test_failed_submission_removes_operation_added_origin_and_workers(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
group:
  workers: [g2]
tasks:
  - command: [echo, ok]
""",
    )

    def fail_after_prepare(operation_id: str, key: str) -> None:
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        batch_submit(cfg, manifest, group="exp", on_prepared=fail_after_prepare)

    group = read_json(group_path(cfg.shared_root, "exp"))
    assert group["group"]["worker_set"] == {}
    assert group["group"]["worker_set_epoch"] == 2


def test_failed_final_operation_commit_removes_operation_added_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    manifest = _manifest(
        tmp_path,
        """
group:
  workers: [g2]
tasks:
  - command: [echo, ok]
""",
    )
    original_atomic_replace = submission_runtime.atomic_replace

    def fail_committed_submission_write(path: Path, value: dict) -> None:
        submission = value.get("submission")
        if submission and submission.get("state") == "committed":
            raise OSError("simulated committed operation write failure")
        original_atomic_replace(path, value)

    monkeypatch.setattr(submission_runtime, "atomic_replace", fail_committed_submission_write)

    with pytest.raises(OSError, match="simulated committed operation write failure"):
        batch_submit(cfg, manifest, group="exp")

    group = read_json(group_path(cfg.shared_root, "exp"))
    assert group["group"]["pending_submission_commit"] is None
    assert group["group"]["worker_set"] == {}
    assert group["group"]["worker_set_epoch"] == 2
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


def test_rejects_home_or_fallback_outside_active_planned_workers(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    batch_submit(cfg, _manifest(tmp_path, "tasks:\n  - command: [echo, seed]\n"), group="exp")
    change_worker(cfg, "exp", "g2", "add")
    change_worker(cfg, "exp", "g2", "drain")
    manifest = _manifest(
        tmp_path,
        """
tasks:
  - placement:
      home_machine: g2
    command: [echo, ok]
""",
    )

    with pytest.raises(ValueError, match="not an active worker"):
        batch_submit(cfg, manifest, group="exp")


def test_rejects_ungrouped_spillover_and_remote_home(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    spillover = _manifest(
        tmp_path,
        """
tasks:
  - placement:
      sharing:
        mode: spillover
    command: [echo, ok]
""",
    )
    with pytest.raises(ValueError, match="ungrouped tasks must use private placement"):
        batch_submit(cfg, spillover)

    remote_home = _manifest(
        tmp_path,
        """
tasks:
  - placement:
      home_machine: g2
    command: [echo, ok]
""",
    )
    with pytest.raises(ValueError, match="ungrouped tasks must use the submitting machine as home"):
        batch_submit(cfg, remote_home)
