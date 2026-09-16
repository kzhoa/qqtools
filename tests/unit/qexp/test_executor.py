from pathlib import Path

import pytest

from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.executor import Executor, LaunchHandoff
from qqtools.plugins.qexp.runtime.records import SCHEMA_VERSION, AttemptRecord


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid


def _attempt() -> AttemptRecord:
    return AttemptRecord.from_dict(
        {
            "meta": {
                "schema_version": SCHEMA_VERSION,
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
                    "clock_evidence": {
                        "clock_error_bound_seconds": 0.1,
                        "clock_provider": "chrony",
                        "clock_observation_id": "test-observation",
                    },
                },
                "authority_mode": "bounded_lease",
                "authorization": {
                    "group_name": None,
                    "group_dispatch_epoch": None,
                    "group_worker_set_epoch": None,
                    "launch_id": "launch-1",
                },
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
        }
    )


def _cfg(tmp_path: Path) -> RootConfig:
    return RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "rt")


def test_executor_uses_tmux_when_available(tmp_path: Path):
    sent: list[tuple[str, str]] = []

    def send_command(window_id: str, command: str) -> None:
        sent.append((window_id, command))
        intent = tmp_path / "rt" / "launch-intents" / "task-1-attempt-1.json"
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.touch()

    executor = Executor(
        create_window=lambda *args: "@7",
        send_command=send_command,
        destroy_window=lambda window_id: None,
        check_window=lambda window_id: True,
        tmux_available=lambda: True,
    )

    result = executor.launch_attempt(_cfg(tmp_path), "task-1", _attempt())

    assert result == "@7"
    assert sent == [("@7", executor.build_runner_command(_cfg(tmp_path), "task-1", "task-1-attempt-1", 7, "launch-1"))]


def test_executor_falls_back_to_detached_runner_without_tmux(tmp_path: Path):
    spawned: list[dict[str, object]] = []

    def fake_spawn(argv, **kwargs):
        spawned.append({"argv": argv, **kwargs})
        intent = tmp_path / "rt" / "launch-intents" / "task-1-attempt-1.json"
        intent.parent.mkdir(parents=True)
        intent.touch()
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
    assert spawned[0]["argv"] == executor.build_runner_argv(cfg, "task-1", "task-1-attempt-1", 7, "launch-1")
    assert spawned[0]["cwd"] == str(cfg.project_root)
    assert spawned[0]["start_new_session"] is True


def test_executor_rejects_runner_without_launch_handoff(tmp_path: Path):
    executor = Executor(tmux_available=lambda: False, spawn_runner=lambda *_args, **_kwargs: _FakeProcess(4321))

    with pytest.raises(RuntimeError, match="did not publish launch intent"):
        executor._wait_for_launch_intent(_cfg(tmp_path), "task-1-attempt-1", timeout_seconds=0.01)


def test_executor_reports_only_failed_handoffs(tmp_path: Path):
    published = tmp_path / "published.json"
    published.touch()
    failures = Executor.wait_for_launch_handoffs(
        [
            LaunchHandoff("published", published, 0.0),
            LaunchHandoff("missing", tmp_path / "missing.json", 0.0),
        ]
    )

    assert list(failures) == ["missing"]
    assert str(failures["missing"]) == "runner did not publish launch intent for 'missing'"
