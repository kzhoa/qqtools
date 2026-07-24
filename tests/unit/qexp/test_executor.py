from pathlib import Path

from qqtools.plugins.qexp.executor import Executor
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.records import AttemptRecord


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid


def _attempt() -> AttemptRecord:
    return AttemptRecord.from_dict({
        "meta": {
            "schema_version": 5,
            "revision": 1,
            "created_at": "2026-07-24T00:00:00Z",
            "updated_at": "2026-07-24T00:00:00Z",
            "updated_by": {"actor_type": "test", "machine_name": "gpu-1", "process_id": "0"},
        },
        "attempt": {
            "attempt_id": "task-1-attempt-1",
            "task_id": "task-1",
            "attempt_number": 1,
            "phase": "claimed",
            "machine_name": "gpu-1",
            "assigned_gpus": [0],
            "reservation_id": "res-1",
            "current_fencing_token": 7,
            "token_history": [7],
            "lease": {
                "claimed_at": "2026-07-24T00:00:00Z",
                "renewed_at": "2026-07-24T00:00:00Z",
                "expires_at": "2026-07-24T00:01:00Z",
            },
            "authorization": {"group_name": None, "group_dispatch_epoch": None, "group_worker_set_epoch": None},
            "process": {
                "wrapper_pid": None,
                "process_group_id": None,
                "tmux_reference": None,
                "local_process_manifest": "",
                "log_references": [],
            },
            "termination": {
                "requested_by_operation_id": None,
                "requested_at": None,
                "acknowledged_at": None,
                "result": None,
            },
            "timestamps": {
                "launch_authorized_at": None,
                "process_created_at": None,
                "running_at": None,
                "orphaned_at": None,
                "recovered_at": None,
                "finished_at": None,
            },
            "result": {"exit_code": None, "signal": None, "category": None, "reason": None},
        },
    })


def _cfg(tmp_path: Path) -> RootConfig:
    return RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")


def test_executor_uses_tmux_when_available(tmp_path: Path):
    sent: list[tuple[str, str]] = []
    executor = Executor(
        create_window=lambda *args: "@7",
        send_command=lambda window_id, command: sent.append((window_id, command)),
        destroy_window=lambda window_id: None,
        check_window=lambda window_id: True,
        tmux_available=lambda: True,
    )

    result = executor.launch_attempt(_cfg(tmp_path), "task-1", _attempt())

    assert result == "@7"
    assert sent == [("@7", executor.build_runner_command(_cfg(tmp_path), "task-1", "task-1-attempt-1", 7))]


def test_executor_falls_back_to_detached_runner_without_tmux(tmp_path: Path):
    spawned: list[dict[str, object]] = []

    def fake_spawn(argv, **kwargs):
        spawned.append({"argv": argv, **kwargs})
        return _FakeProcess(4321)

    executor = Executor(
        create_window=lambda *args: (_ for _ in ()).throw(AssertionError("tmux path should not be used")),
        send_command=lambda *args: (_ for _ in ()).throw(AssertionError("tmux path should not be used")),
        destroy_window=lambda window_id: None,
        check_window=lambda window_id: False,
        tmux_available=lambda: False,
        spawn_runner=fake_spawn,
    )
    cfg = _cfg(tmp_path)

    result = executor.launch_attempt(cfg, "task-1", _attempt())

    assert result == "pid:4321"
    assert len(spawned) == 1
    assert spawned[0]["argv"] == executor.build_runner_argv(cfg, "task-1", "task-1-attempt-1", 7)
    assert spawned[0]["cwd"] == str(cfg.project_root)
    assert spawned[0]["start_new_session"] is True
