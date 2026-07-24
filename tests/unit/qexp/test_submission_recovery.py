from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.task import batch_submit
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.paths import idempotency_path, submission_path
from qqtools.plugins.qexp.runtime.submission import IdempotencyConflict
from qqtools.plugins.qexp.runtime.submission import semantic_digest


def test_sealed_group_does_not_poison_idempotency_key(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "first"], group="exp")
    from qqtools.plugins.qexp.commands.group import group_control
    group_control(cfg, "exp", "seal")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "second"], group="exp", idempotency_key="sealed-key")
    with pytest.raises(ValueError, match="sealed"):
        submit(cfg, ["echo", "second"], group="exp", idempotency_key="sealed-key")
    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": "sealed-key"}))
    assert not mapping.exists()
