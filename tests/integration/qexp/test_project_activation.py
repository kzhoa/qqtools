"""Durable wake discovery for dormant qexp Project bindings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import submit
from qqtools.plugins.qexp.runtime.availability import offer_deadlines as offer_deadline_module
from qqtools.plugins.qexp.runtime.availability.offer_deadlines import remove_deadline_index
from qqtools.plugins.qexp.runtime.group_discovery import locator
from qqtools.plugins.qexp.runtime.group_discovery import service as group_service
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from qqtools.plugins.qexp.runtime.observation import maintenance as observation_maintenance
from qqtools.plugins.qexp.runtime.observation import projection as observation_projection
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.project_activation import (
    activation_checkpoint_path,
    activation_event_path,
    project_activation_transaction,
    publish_project_activation,
    read_project_activation,
    read_project_activation_events,
    recover_project_activation,
)
from qqtools.plugins.qexp.runtime.ready import bump_primary_ready_revision
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from tests.helpers.qexp_discovery import isolated_group

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _activation(cfg) -> dict[str, object]:
    return read_project_activation(cfg.shared_root)["project_activation"]


def test_activation_checkpoint_retains_epoch_and_serializes_publishers(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda index: publish_project_activation(cfg, f"test_{index}"), range(32)))

    persisted = _activation(cfg)
    epochs = {record["project_activation"]["epoch"] for record in records}
    sequences = sorted(record["project_activation"]["sequence"] for record in records)
    assert len(epochs) == 1
    assert sequences == list(range(1, 33))
    assert persisted["epoch"] in epochs
    assert persisted["sequence"] == 32
    assert persisted["reason"].startswith("test_")

    events = read_project_activation_events(cfg.shared_root, epoch=persisted["epoch"], after_sequence=0, limit=64)
    assert [event["project_activation_event"]["sequence"] for event in events] == list(range(1, 33))


def test_activation_replay_rejects_a_missing_suffix_event(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    first = publish_project_activation(cfg, "first")["project_activation"]
    second = publish_project_activation(cfg, "second")["project_activation"]
    activation_event_path(cfg.shared_root, first["epoch"], first["sequence"]).unlink()

    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        read_project_activation_events(cfg.shared_root, epoch=first["epoch"], after_sequence=0, limit=2)
    assert _activation(cfg)["sequence"] == 2
    activation_event_path(cfg.shared_root, second["epoch"], second["sequence"]).unlink()
    with pytest.raises(RuntimeError, match="missing event"):
        recover_project_activation(cfg.shared_root)


def test_event_write_failure_never_advances_the_visible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    from qqtools.plugins.qexp.runtime import project_activation as activation_module

    original_replace = activation_module.atomic_replace

    def fail_event(path, value):
        if path.parent.parent.name == "events":
            raise OSError("event storage unavailable")
        return original_replace(path, value)

    monkeypatch.setattr(activation_module, "atomic_replace", fail_event)
    with pytest.raises(OSError, match="event storage unavailable"):
        publish_project_activation(cfg, "interrupted")

    assert read_project_activation(cfg.shared_root) is None


def test_activation_transaction_hides_prepared_event_until_truth_handoff(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    with project_activation_transaction(cfg, "covered_change") as prepared:
        activation = prepared["project_activation"]
        assert read_project_activation(cfg.shared_root) is None
        assert activation_event_path(cfg.shared_root, activation["epoch"], activation["sequence"]).is_file()

    assert _activation(cfg) == activation


def test_checkpoint_failure_is_recovered_before_the_next_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    from qqtools.plugins.qexp.runtime import project_activation as activation_module

    original_replace = activation_module.atomic_replace
    failed = False

    def fail_checkpoint_once(path, value):
        nonlocal failed
        if path == activation_checkpoint_path(cfg.shared_root) and not failed:
            failed = True
            raise OSError("checkpoint storage unavailable")
        return original_replace(path, value)

    monkeypatch.setattr(activation_module, "atomic_replace", fail_checkpoint_once)
    with pytest.raises(OSError, match="checkpoint storage unavailable"):
        publish_project_activation(cfg, "interrupted")

    assert read_project_activation(cfg.shared_root) is None
    recovered = recover_project_activation(cfg.shared_root)["project_activation"]
    assert recovered["reason"] == "interrupted"
    second = publish_project_activation(cfg, "next")["project_activation"]
    assert second["epoch"] == recovered["epoch"]
    assert second["sequence"] == recovered["sequence"] + 1
    events = read_project_activation_events(
        cfg.shared_root,
        epoch=second["epoch"],
        after_sequence=0,
        limit=second["sequence"],
    )
    assert [event["project_activation_event"]["reason"] for event in events] == ["interrupted", "next"]


def test_checkpoint_without_event_rotates_to_a_replayable_epoch(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    legacy_epoch = "12345678123456781234567812345678"
    project_id = read_json(cfg.shared_root / "project" / "identity.json")["project"]["project_id"]
    atomic_replace(
        activation_checkpoint_path(cfg.shared_root),
        {
            "project_activation": {
                "version": 1,
                "identity": {"project_id": project_id},
                "epoch": legacy_epoch,
                "sequence": 7,
                "reason": "legacy_checkpoint",
                "updated_at": "2026-09-25T00:00:00+00:00",
            }
        },
    )

    with pytest.raises(RuntimeError, match="missing event"):
        read_project_activation(cfg.shared_root)
    reconstructed = recover_project_activation(cfg.shared_root)["project_activation"]

    assert reconstructed["epoch"] != legacy_epoch
    assert reconstructed["sequence"] == 1
    assert reconstructed["reason"] == "checkpoint_reconstruction"
    assert (
        read_project_activation_events(
            cfg.shared_root,
            epoch=reconstructed["epoch"],
            after_sequence=0,
            limit=1,
        )[0]["project_activation_event"]
        == reconstructed
    )


def test_corrupt_activation_checkpoint_fails_closed_without_replacement(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    path = activation_checkpoint_path(cfg.shared_root)
    atomic_replace(
        path,
        {
            "project_activation": {
                "version": 1,
                "identity": {"project_id": "wrong-project"},
                "epoch": "0" * 32,
                "sequence": 9,
                "reason": "ready_route_update",
                "updated_at": "2026-09-25T00:00:00+00:00",
            }
        },
    )
    before = path.read_bytes()

    with pytest.raises((RuntimeError, ValueError)):
        publish_project_activation(cfg, "ready_route_update")

    assert path.read_bytes() == before


def test_ready_route_change_commits_activation_with_the_route_revision(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    bump_primary_ready_revision(cfg, "home", "gpu-1")
    first = _activation(cfg)
    bump_primary_ready_revision(cfg, "home", "gpu-1", lane="cpu")
    second = _activation(cfg)

    assert first["sequence"] == 1
    assert second["epoch"] == first["epoch"]
    assert second["sequence"] == 2
    assert second["reason"] == "ready_route_update"


def test_group_transition_wakes_through_compatibility_fallback(tmp_path: Path) -> None:
    cfg = isolated_group(tmp_path, tail=0)

    with group_writer_lock(cfg, "experiment"):
        assert (
            group_service.publish_group_locator_for_transition(
                cfg,
                "experiment",
                "control",
                "task_change",
            )
            is None
        )
    published = _activation(cfg)

    with group_writer_lock(cfg, "experiment"), pytest.raises(ValueError, match="unsupported Group service lane"):
        group_service.publish_group_locator_for_transition(cfg, "experiment", "invalid", "task_change")

    assert _activation(cfg) == published
    assert published["sequence"] == 1
    assert published["reason"] == "group_locator_control"


def test_fenced_group_locator_failure_stops_transition_after_wake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = isolated_group(tmp_path, tail=0)
    activation_path = cfg.shared_root / "schema" / "group-service.json"
    original_read = group_service.read_json_limited

    def fenced_read(path, **kwargs):
        if path == activation_path:
            return {"state": "fenced"}
        return original_read(path, **kwargs)

    def unavailable(*_args, **_kwargs):
        raise OSError("locator storage unavailable")

    monkeypatch.setattr(group_service, "read_json_limited", fenced_read)
    monkeypatch.setattr(locator, "publish_group_locator_locked", unavailable)

    with group_writer_lock(cfg, "experiment"), pytest.raises(OSError, match="locator storage unavailable"):
        group_service.publish_group_locator_for_transition(
            cfg,
            "experiment",
            "maintenance",
            "metadata_cleanup",
        )

    assert _activation(cfg)["reason"] == "group_locator_maintenance"


def test_deadline_noop_and_removal_publish_only_real_changes(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment")

    task = submit(
        cfg,
        ["echo", "ok"],
        group="experiment",
        sharing_mode="spillover",
        offer_after_seconds=60,
    )
    after_add = _activation(cfg)

    # Re-submitting the already materialized deadline projection is a no-op.
    from qqtools.plugins.qexp.runtime.availability.offer_deadlines import sync_deadline_index

    sync_deadline_index(cfg, task)
    after_noop = _activation(cfg)
    remove_deadline_index(cfg, task.task_id)
    after_remove = _activation(cfg)
    remove_deadline_index(cfg, task.task_id)

    assert after_noop["sequence"] == after_add["sequence"]
    assert after_remove["sequence"] == after_add["sequence"] + 1
    assert after_remove["reason"] == "offer_deadline_update"
    assert _activation(cfg)["sequence"] == after_remove["sequence"]


def test_deadline_index_syncs_links_before_activation_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment")
    synced: set[Path] = set()
    original_sync = offer_deadline_module._sync_directory

    def record_sync(path: Path) -> None:
        synced.add(path)
        original_sync(path)

    monkeypatch.setattr(offer_deadline_module, "_sync_directory", record_sync)
    task = submit(
        cfg,
        ["echo", "ok"],
        group="experiment",
        sharing_mode="spillover",
        offer_after_seconds=60,
    )

    paths = shared_paths(cfg.shared_root)
    assert paths["offer_deadlines"] in synced
    assert paths["offer_deadlines_active"] in synced
    synced.clear()

    remove_deadline_index(cfg, task.task_id)

    assert paths["offer_deadlines"] in synced
    assert paths["offer_deadlines_active"] in synced


def test_observation_rebuild_request_commits_wake_after_state_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = isolated_group(tmp_path, tail=0)
    original_write = observation_projection.write_state
    published = []

    def observe_write(project_cfg, state):
        published.append(read_project_activation(project_cfg.shared_root))
        return original_write(project_cfg, state)

    monkeypatch.setattr(observation_projection, "write_state", observe_write)

    result = observation_maintenance.request_rebuild(cfg)

    assert result["reason"] == "rebuild_requested"
    assert published == [None]
    assert _activation(cfg)["reason"] == "observation_rebuild_request"


def test_uninitialized_project_has_no_activation_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "missing"

    assert read_project_activation(root) is None
    assert not activation_checkpoint_path(root).exists()
