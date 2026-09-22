from __future__ import annotations

import pytest

from qqtools.plugins.qexp.commands.group import change_worker, create_group
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def _worker(result, machine):
    return result["group"]["worker_set"][machine]


def test_worker_add_set_drain_resume_have_distinct_lifecycle_meanings(tmp_path) -> None:
    cfg = isolated_group(tmp_path, tail=0)
    create_group(cfg, "workers", ["gpu-1"])

    created = change_worker(
        cfg,
        "workers",
        "gpu-2",
        "add",
        role="borrow",
        gpu_limit_gpus=2,
        has_gpu_limit=True,
    )
    repeated = change_worker(
        cfg,
        "workers",
        "gpu-2",
        "add",
        role="borrow",
        gpu_limit_gpus=2,
        has_gpu_limit=True,
    )
    assert _worker(repeated, "gpu-2") == _worker(created, "gpu-2")

    with pytest.raises(ValueError, match="set"):
        change_worker(cfg, "workers", "gpu-2", "add", role="primary")

    drained = change_worker(cfg, "workers", "gpu-2", "drain")
    drained_worker = dict(_worker(drained, "gpu-2"))
    changed_while_draining = change_worker(cfg, "workers", "gpu-2", "set", role="primary")
    assert _worker(changed_while_draining, "gpu-2")["state"] == "draining"
    assert _worker(changed_while_draining, "gpu-2")["scheduling_role"] == "primary"

    with pytest.raises(ValueError, match="resume"):
        change_worker(cfg, "workers", "gpu-2", "add")

    resumed = change_worker(cfg, "workers", "gpu-2", "resume")
    assert _worker(resumed, "gpu-2")["state"] == "active"
    assert _worker(resumed, "gpu-2")["scheduling_role"] == "primary"
    assert _worker(resumed, "gpu-2")["gpu_limit_gpus"] == drained_worker["gpu_limit_gpus"]
    assert change_worker(cfg, "workers", "gpu-2", "resume")["group"]["worker_set"]["gpu-2"] == _worker(resumed, "gpu-2")


def test_worker_resume_rejects_an_in_progress_removal(tmp_path) -> None:
    cfg = isolated_group(tmp_path, tail=0)
    created = create_group(cfg, "workers", ["gpu-1", "gpu-2"])
    worker = created["group"]["worker_set"]["gpu-2"]
    worker["state"] = "draining"
    worker["removal_operation_id"] = "operation-1"
    from qqtools.plugins.qexp.runtime.paths import group_path
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    atomic_replace(group_path(cfg.shared_root, "workers"), created)

    with pytest.raises(ValueError, match="removal"):
        change_worker(cfg, "workers", "gpu-2", "resume")
