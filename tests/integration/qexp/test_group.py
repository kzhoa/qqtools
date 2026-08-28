from pathlib import Path

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root, submit
from qqtools.plugins.qexp.commands.group import group_control
from qqtools.plugins.qexp.observer import list_groups

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_group_membership_sequences_and_workers_are_authoritative(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    first = submit(cfg, ["echo", "one"], group="exp", sharing_mode="spillover")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("tasks:\n  - command: [echo, two]\n", encoding="utf-8")
    second = batch_submit(cfg, manifest, group="exp")[0]
    assert first.group_membership_sequence == 1
    assert second.group_membership_sequence == 2
    group = list_groups(cfg)[0]["group"]
    assert set(group["worker_set"]) == {"g1"}


def test_sealed_group_rejects_new_tasks(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "one"], group="exp")
    group_control(cfg, "exp", "seal")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "two"], group="exp")


def test_pause_and_resume_change_dispatch_epoch(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "one"], group="exp")
    group_control(cfg, "exp", "pause")
    paused = list_groups(cfg)[0]["group"]
    assert paused["dispatch_state"] == "paused"
    assert paused["dispatch_epoch"] == 1
    group_control(cfg, "exp", "resume")
    assert list_groups(cfg)[0]["group"]["dispatch_state"] == "active"
