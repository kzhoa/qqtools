from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import change_worker
from qqtools.plugins.qexp.commands.task import offer
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.reservations import reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    claim_task,
    resume_starting_attempt,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _offered_task_and_remote_cfg(tmp_path: Path) -> tuple[RootConfig, RootConfig, str]:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "gpu-1"
    )
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    change_worker(cfg, "exp", "gpu-2", "add")
    offer(cfg, task.task_id)
    remote_cfg = RootConfig(
        cfg.shared_root,
        cfg.project_root,
        "gpu-2",
        tmp_path / "gpu-2",
    )
    return cfg, remote_cfg, task.task_id


def test_cross_host_claim_lifecycle_does_not_require_qualification_file(tmp_path: Path) -> None:
    _cfg, remote_cfg, task_id = _offered_task_and_remote_cfg(tmp_path)

    attempt = claim_task(remote_cfg, task_id, [0])

    assert attempt is not None
    assert attempt.machine_name == "gpu-2"
    assert authorize_launch(
        remote_cfg,
        task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
    )

    resumed_attempt = resume_starting_attempt(remote_cfg, task_id)

    assert resumed_attempt is not None
    assert resumed_attempt.attempt_id == attempt.attempt_id
    assert reserved_gpu_ids(remote_cfg.runtime_root) == {0}
    assert load_task(remote_cfg, task_id).state["projection"] == "running"


def test_setup_and_registration_ignore_legacy_qualification_file(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "gpu-1"
    )
    atomic_replace(
        shared_paths(cfg.shared_root)["project"] / "filesystem-qualification.json",
        {"filesystem_qualification": {"version": 1}},
    )

    remote_cfg = init_shared_root(
        cfg.shared_root,
        "gpu-2",
        runtime_root=tmp_path / "gpu-2",
    )
    binding, is_created = MachineRuntime(tmp_path / "machine-runtime").ensure_binding(
        cfg.shared_root,
        cfg.machine_name,
    )

    assert remote_cfg.machine_name == "gpu-2"
    assert binding.machine_name == cfg.machine_name
    assert is_created is True
