from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build, classify_ready_marker
from qqtools.plugins.qexp.runtime.records import TaskRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _ready_reference(cfg, task: TaskRecord):
    from qqtools.plugins.qexp.runtime.ready import ReadyMarkerRef

    path = (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    )
    record = read_json(path)["ready_reservation"]
    return ReadyMarkerRef(
        task.task_id,
        task.ready_generation,
        record["queue_scope"],
        record["home_machine"],
        record["partition"],
        record["catalog_page"],
        record["marker_name"],
    )


def _finish_build(cfg) -> dict:
    record = advance_ready_index_build(cfg)
    while record["state"] == "building":
        record = advance_ready_index_build(cfg)
    return record


def test_doctor_reports_and_repairs_missing_active_marker(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "repair"], task_id="repair-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    marker = (
        shared_paths(cfg.shared_root)["ready_home"]
        / reference.home_machine
        / reference.partition
        / reference.marker_name
    )
    marker.unlink()

    verification = verify_integrity(cfg, reservation_runtime_root=cfg.runtime_root)
    repaired = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root)
    current = load_task(cfg, task.task_id)

    assert any(
        issue["code"] == "ready_projection_inconsistent"
        for issue in verification["issues"]
    )
    assert repaired["ready_index"]["state"] == "active"
    assert current.ready_generation > task.ready_generation
    assert classify_ready_marker(cfg, _ready_reference(cfg, current)).classification == "claimable"


def test_doctor_rebuilds_corrupt_catalog_from_task_truth(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "repair"], task_id="catalog-task")
    _finish_build(cfg)
    reference = _ready_reference(cfg, load_task(cfg, task.task_id))
    catalog = (
        shared_paths(cfg.shared_root)["ready_catalogs"]
        / f"home.{cfg.machine_name}"
        / f"{reference.catalog_page:016d}.json"
    )
    atomic_replace(catalog, {"corrupt": {}})

    repaired = repair_metadata(cfg, reservation_runtime_root=cfg.runtime_root)
    current = load_task(cfg, task.task_id)

    assert repaired["ready_index"]["state"] == "active"
    assert current.ready_generation > task.ready_generation
    assert classify_ready_marker(cfg, _ready_reference(cfg, current)).classification == "claimable"
