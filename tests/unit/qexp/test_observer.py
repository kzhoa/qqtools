from pathlib import Path

from qqtools.plugins.qexp import observer
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.records import SCHEMA_VERSION, TaskRecord


def _task(task_id: str, *, projection: str = "queued") -> TaskRecord:
    return TaskRecord.from_dict(
        {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "revision": 1,
                "created_at": "2026-09-15T00:00:00Z",
                "updated_at": "2026-09-15T00:00:00Z",
                "updated_by": {"actor_type": "test", "machine_name": "gpu-1", "process_id": "0"},
            },
            "task": {
                "task_id": task_id,
                "group_name": None,
                "group_membership_sequence": None,
                "submission_operation_id": None,
                "name": None,
                "depends_on_task_ids": [],
                "ready_generation": 1,
                "spec": {
                    "command": ["true"],
                    "working_directory": "/tmp",
                    "requested_gpus": 1,
                    "requested_cpus": None,
                },
                "placement_policy": {
                    "home_machine": "gpu-1",
                    "sharing_mode": "private",
                    "fallback_machines": [],
                },
                "placement_runtime": {"queue_scope": "home", "offer_clock_evidence": None},
                "state": {"projection": projection, "reason": None},
                "control": {"cleanup_operation_id": None, "cleanup_state": None},
                "attempt_control": {
                    "next_attempt_number": 1,
                    "current_attempt_number": None,
                    "current_attempt_id": None,
                },
                "claim_control": {"fencing_epoch": 0, "active_claim": None},
            },
        }
    )


def test_list_tasks_stops_reading_after_limit(monkeypatch, tmp_path: Path) -> None:
    paths = [tmp_path / f"task-{index}.json" for index in range(100)]
    read_paths: list[Path] = []
    monkeypatch.setattr(observer, "iter_json", lambda _directory: paths)

    def read_task(path: Path):
        read_paths.append(path)
        return _task(path.stem).to_dict()

    monkeypatch.setattr(observer, "read_json", read_task)
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")

    result = observer.list_tasks(cfg, limit=3)

    assert [item["task_id"] for item in result] == ["task-0", "task-1", "task-2"]
    assert read_paths == paths[:3]
