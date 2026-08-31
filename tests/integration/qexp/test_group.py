from pathlib import Path

import pytest

from qqtools.plugins.qexp import batch_submit, init_shared_root, submit
from qqtools.plugins.qexp.commands.group import change_worker, create_group, group_control
from qqtools.plugins.qexp.observer import list_group_machines, list_groups
from qqtools.plugins.qexp.runtime.paths import group_path
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.reservations import reserve_admitted
from qqtools.plugins.qexp.runtime.worker_encoding import (
    ENCODING_CANONICAL_V2,
    read_primary_borrow_encoding,
    write_primary_borrow_encoding,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_group_membership_sequences_and_workers_are_authoritative(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
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
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], group="exp")
    group_control(cfg, "exp", "seal")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "two"], group="exp")


def test_pause_and_resume_change_dispatch_epoch(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], group="exp")
    group_control(cfg, "exp", "pause")
    paused = list_groups(cfg)[0]["group"]
    assert paused["dispatch_state"] == "paused"
    assert paused["dispatch_epoch"] == 1
    group_control(cfg, "exp", "resume")
    assert list_groups(cfg)[0]["group"]["dispatch_state"] == "active"


def test_group_worker_role_and_limit_change_is_epoch_linearized(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")

    added = change_worker(
        cfg, "exp", "g2", "add", role="borrow", gpu_limit_gpus=2, has_gpu_limit=True
    )
    worker = added["group"]["worker_set"]["g2"]
    assert worker["state"] == "borrow"
    assert worker["scheduling_role"] == "borrow"
    assert worker["gpu_limit_gpus"] == 2
    epoch = added["group"]["worker_set_epoch"]

    updated = change_worker(
        cfg, "exp", "g2", "set", gpu_limit_gpus=1, has_gpu_limit=True
    )
    assert updated["group"]["worker_set"]["g2"]["gpu_limit_gpus"] == 1
    assert updated["group"]["worker_set_epoch"] == epoch + 1

    primary = change_worker(cfg, "exp", "g2", "set", role="primary")
    assert primary["group"]["worker_set"]["g2"]["state"] == "active"
    assert primary["group"]["worker_set"]["g2"]["gpu_limit_gpus"] == 1


def test_group_worker_set_accepts_primary_gpu_limit(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")

    group = change_worker(
        cfg, "exp", "g2", "add", role="primary", gpu_limit_gpus=1, has_gpu_limit=True
    )
    assert group["group"]["worker_set"]["g2"]["gpu_limit_gpus"] == 1


def test_canonical_marker_keeps_borrow_reader_and_writer_compatible(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    change_worker(cfg, "exp", "g2", "add", role="borrow", gpu_limit_gpus=2,
                  has_gpu_limit=True)
    marker = read_primary_borrow_encoding(cfg)
    marker.update({"state": ENCODING_CANONICAL_V2, "revision": 2, "started_by_agent": "agent-1"})
    write_primary_borrow_encoding(cfg, marker)

    updated = change_worker(cfg, "exp", "g2", "set", role="borrow",
                            gpu_limit_gpus=1, has_gpu_limit=True)
    assert updated["group"]["worker_set"]["g2"]["state"] == "active"
    persisted = read_json(group_path(cfg.shared_root, "exp"))
    assert persisted["group"]["worker_set"]["g2"]["scheduling_role"] == "borrow"


def test_group_machine_usage_includes_unexpired_provisional_reservation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    change_worker(cfg, "exp", "g1", "set", role="borrow", gpu_limit_gpus=1,
                  has_gpu_limit=True)
    reserve_admitted(
        cfg.runtime_root, "task-1", [0], project_id="project", group_name="exp",
        machine_name="g1", gpu_limit_gpus=1, worker_scheduling_role="borrow",
        shared_root=str(cfg.shared_root),
        group_worker_set_epoch=1, worker_state_epoch=1,
    )

    machine = list_group_machines(cfg, "exp", reservation_runtime_root=cfg.runtime_root)

    assert machine["machines"][0]["gpu_usage"] == 1
    assert machine["machines"][0]["state"] == "full"
