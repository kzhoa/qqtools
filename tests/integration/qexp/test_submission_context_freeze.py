from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.config_types import RootConfig

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_cross_machine_idempotency_reuses_original_home_and_workers(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "g1")
    first = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover", idempotency_key="same")
    other = RootConfig(cfg.shared_root, cfg.project_root, "g2", tmp_path / "g2")
    second = submit(other, ["echo", "ok"], group="exp", sharing_mode="spillover", idempotency_key="same")
    assert second.task_id == first.task_id
    assert second.placement_policy["home_machine"] == "g1"
