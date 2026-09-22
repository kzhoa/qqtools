import math
from datetime import datetime, timezone
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


def test_machine_view_reads_nested_agent_freshness(monkeypatch, tmp_path: Path) -> None:
    machine_root = tmp_path / "machines"
    state_dir = machine_root / "gpu-1" / "state"
    state_dir.mkdir(parents=True)
    machine_path = machine_root / "gpu-1" / "machine.json"
    agent_path = state_dir / "agent.json"
    machine_path.touch()
    agent_path.touch()
    heartbeat = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(observer, "shared_paths", lambda _root: {"machines": machine_root})
    monkeypatch.setattr(
        observer,
        "read_json",
        lambda path: (
            {"machine": {"machine_name": "gpu-1"}}
            if path == machine_path
            else {
                "agent": {
                    "observed_state": "active",
                    "heartbeat_at": heartbeat,
                    "heartbeat_interval_seconds": 10,
                }
            }
        ),
    )
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")

    result = observer._machine_view(cfg, machine_path)

    assert result["state"]["freshness"] == "fresh"


def test_machine_view_distinguishes_stopped_missing_and_unavailable(monkeypatch, tmp_path: Path) -> None:
    machine_root = tmp_path / "machines"
    state_dir = machine_root / "gpu-1" / "state"
    state_dir.mkdir(parents=True)
    machine_path = machine_root / "gpu-1" / "machine.json"
    agent_path = state_dir / "agent.json"
    machine_path.touch()
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")
    monkeypatch.setattr(observer, "shared_paths", lambda _root: {"machines": machine_root})

    def view(agent: object | None) -> str:
        if agent is None:
            agent_path.unlink(missing_ok=True)
        else:
            agent_path.touch()
        monkeypatch.setattr(
            observer,
            "read_json",
            lambda path: {"machine": {"machine_name": "gpu-1"}} if path == machine_path else agent,
        )
        return observer._machine_view(cfg, machine_path)["state"]["freshness"]

    assert view({"agent": {"observed_state": "stopped"}}) == "stopped"
    assert view(None) == "missing"
    assert view({"agent": "malformed"}) == "unavailable"
    for interval in (True, 0, -1, math.inf, math.nan):
        assert (
            view(
                {
                    "agent": {
                        "observed_state": "active",
                        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                        "heartbeat_interval_seconds": interval,
                    }
                }
            )
            == "unavailable"
        )
