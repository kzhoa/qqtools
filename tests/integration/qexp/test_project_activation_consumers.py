"""Shared per-consumer progress for Project activation replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.runtime import project_activation as activation_module
from qqtools.plugins.qexp.runtime.project_activation import (
    activation_event_path,
    activation_snapshot_path,
    compact_project_activation,
    publish_project_activation,
    read_project_activation_events,
)
from qqtools.plugins.qexp.runtime.project_activation_consumers import (
    ack_consumer,
    consumer_progress_path,
    read_consumer_progress,
    register_consumer,
    retire_consumer,
)
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _identity(cfg) -> dict[str, str]:
    project_id = read_json(cfg.shared_root / "project" / "identity.json")["project"]["project_id"]
    return {"runtime_id": "runtime-a", "project_id": project_id, "registration_generation": "generation-a"}


def test_consumer_ack_is_fenced_by_current_checkpoint_and_process(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    identity = _identity(cfg)
    first = publish_project_activation(cfg, "first")["project_activation"]
    assert (
        register_consumer(cfg.shared_root, **identity, process_fence="process-a")["project_activation_consumer"]["ack"]
        is None
    )

    ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=first["epoch"],
        sequence=first["sequence"],
    )
    second = publish_project_activation(cfg, "second")["project_activation"]
    # A repeated contiguous acknowledgement remains idempotent after publication.
    ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=first["epoch"],
        sequence=first["sequence"],
    )
    with pytest.raises(ValueError):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=first["epoch"],
            sequence=second["sequence"] + 1,
        )
    register_consumer(cfg.shared_root, **identity, process_fence="process-b")
    with pytest.raises(ValueError):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=second["epoch"],
            sequence=second["sequence"],
        )
    acked = ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-b",
        epoch=second["epoch"],
        sequence=second["sequence"],
    )
    assert acked["project_activation_consumer"]["ack"] == {
        "epoch": second["epoch"],
        "sequence": second["sequence"],
    }


def test_consumer_retirement_keeps_replay_proof_and_rejects_reuse(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    identity = _identity(cfg)
    checkpoint = publish_project_activation(cfg, "work")["project_activation"]
    register_consumer(cfg.shared_root, **identity, process_fence="process-a")
    ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=checkpoint["epoch"],
        sequence=checkpoint["sequence"],
    )

    retired = retire_consumer(cfg.shared_root, **identity, process_fence="process-a")

    assert retired["project_activation_consumer"]["state"] == "retired"
    assert read_consumer_progress(cfg.shared_root, **identity) == retired
    assert consumer_progress_path(cfg.shared_root, "runtime-a", "generation-a").exists()
    with pytest.raises(ValueError):
        register_consumer(cfg.shared_root, **identity, process_fence="process-b")


def test_shared_ack_rejects_a_gap_in_the_event_suffix(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    identity = _identity(cfg)
    first = publish_project_activation(cfg, "first")["project_activation"]
    publish_project_activation(cfg, "second")
    last = publish_project_activation(cfg, "third")["project_activation"]
    register_consumer(cfg.shared_root, **identity, process_fence="process-a")
    activation_event_path(cfg.shared_root, first["epoch"], 2).unlink()

    with pytest.raises((RuntimeError, ValueError)):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=last["epoch"],
            sequence=last["sequence"],
        )
    assert read_consumer_progress(cfg.shared_root, **identity)["project_activation_consumer"]["ack"] is None


def test_shared_ack_advances_only_one_bounded_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    monkeypatch.setattr(activation_module, "_maybe_compact_activation_locked", lambda *_args: None)
    identity = _identity(cfg)
    last = None
    for index in range(257):
        last = publish_project_activation(cfg, f"change_{index}")["project_activation"]
    register_consumer(cfg.shared_root, **identity, process_fence="process-a")

    with pytest.raises(ValueError, match="bounded replay batch"):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=last["epoch"],
            sequence=last["sequence"],
        )
    first_batch = ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=last["epoch"],
        sequence=256,
    )
    assert first_batch["project_activation_consumer"]["ack"]["sequence"] == 256
    final = ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=last["epoch"],
        sequence=257,
    )
    assert final["project_activation_consumer"]["ack"]["sequence"] == 257


def test_automatic_compaction_requires_snapshot_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    identity = _identity(cfg)
    monkeypatch.setattr(activation_module, "_COMPACTION_BATCH", 3)
    last = None
    for index in range(3):
        last = publish_project_activation(cfg, f"change_{index}")["project_activation"]
    register_consumer(cfg.shared_root, **identity, process_fence="process-a")

    snapshot = read_json(activation_snapshot_path(cfg.shared_root))["project_activation_snapshot"]
    assert snapshot["floor_sequence"] == last["sequence"]
    with pytest.raises(ValueError, match="reconstruction"):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=last["epoch"],
            sequence=last["sequence"],
        )

    reconstructed = ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=last["epoch"],
        sequence=last["sequence"],
        reconstructed_floor=last["sequence"],
    )
    assert reconstructed["project_activation_consumer"]["ack"]["sequence"] == last["sequence"]


def test_compaction_requires_checkpoint_reconstruction_before_acknowledgement(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    identity = _identity(cfg)
    first = publish_project_activation(cfg, "first")["project_activation"]
    register_consumer(cfg.shared_root, **identity, process_fence="process-a")
    ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=first["epoch"],
        sequence=first["sequence"],
    )
    publish_project_activation(cfg, "second")
    last = publish_project_activation(cfg, "third")["project_activation"]

    snapshot = compact_project_activation(cfg.shared_root)["project_activation_snapshot"]

    assert snapshot["epoch"] == last["epoch"]
    assert snapshot["floor_sequence"] == last["sequence"]
    assert activation_snapshot_path(cfg.shared_root).is_file()
    assert not activation_event_path(cfg.shared_root, last["epoch"], 1).exists()
    with pytest.raises(ValueError, match="retention floor"):
        read_project_activation_events(
            cfg.shared_root,
            epoch=last["epoch"],
            after_sequence=first["sequence"],
            limit=2,
        )
    with pytest.raises(ValueError, match="reconstruction"):
        ack_consumer(
            cfg.shared_root,
            **identity,
            process_fence="process-a",
            epoch=last["epoch"],
            sequence=last["sequence"],
        )

    reconstructed = ack_consumer(
        cfg.shared_root,
        **identity,
        process_fence="process-a",
        epoch=last["epoch"],
        sequence=last["sequence"],
        reconstructed_floor=last["sequence"],
    )
    assert reconstructed["project_activation_consumer"]["ack"] == {
        "epoch": last["epoch"],
        "sequence": last["sequence"],
    }
