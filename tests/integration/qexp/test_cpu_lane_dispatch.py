from pathlib import Path

from qqtools.plugins.qexp.commands.task import submit
from qqtools.plugins.qexp.executor import Executor
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime.cpu_lane import cpu_reservation_snapshot, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.paths import task_path
from qqtools.plugins.qexp.project_maintenance import reconcile_project_reservations
from qqtools.plugins.qexp.scheduler import claim_task, run_dispatch_cycle
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


class _RecordingExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(tmux_available=lambda: False)
        self.launched: list[tuple[str, list[int]]] = []

    def launch_attempt(self, cfg, task_id, attempt, session_name="experiments") -> str:
        self.launched.append((task_id, attempt.assigned_gpus))
        return "recorded"


def test_canonical_root_dispatches_cpu_only_task(tmp_path: Path) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "cpu-host")
    set_cpu_lane_capacity(cfg.runtime_root, capacity=2)
    task = submit(
        cfg,
        ["echo", "ok"],
        requested_gpus=0,
        requested_cpus=1,
        working_dir=working_directory,
    )
    executor = _RecordingExecutor()

    launched = run_dispatch_cycle(cfg, available_cpus=2, executor=executor)

    assert launched == [task.task_id]
    assert executor.launched == [(task.task_id, [])]
    _policy, reservations = cpu_reservation_snapshot(cfg.runtime_root)
    assert [item["cpu_slots"] for item in reservations] == [1]


def test_legacy_root_rejects_cpu_only_submission_without_task_write(tmp_path: Path) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "cpu-host")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    del schema["schema"]["required_capabilities"]
    atomic_replace(schema_path, schema)

    try:
        submit(
            cfg, ["echo", "no"], requested_gpus=0, requested_cpus=1,
            working_dir=working_directory,
        )
    except ValueError as exc:
        assert "canonical CPU-lane root" in str(exc)
    else:
        raise AssertionError("legacy root accepted a CPU-only Task")
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


def test_cpu_claim_race_does_not_consume_the_next_task_slot(
    tmp_path: Path, monkeypatch
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "cpu-host")
    set_cpu_lane_capacity(cfg.runtime_root, capacity=1)
    submit(
        cfg,
        ["echo", "first"],
        requested_gpus=0,
        requested_cpus=1,
        task_id="first",
        working_dir=working_directory,
    )
    expected = submit(
        cfg,
        ["echo", "second"],
        requested_gpus=0,
        requested_cpus=1,
        task_id="second",
        working_dir=working_directory,
    )
    original_claim_task = claim_task
    attempts = 0

    def reject_first_claim(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("simulated concurrent claim")
        return original_claim_task(*args, **kwargs)

    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.claim_task", reject_first_claim)
    executor = _RecordingExecutor()

    assert run_dispatch_cycle(cfg, available_cpus=1, executor=executor) == [expected.task_id]
    assert executor.launched == [(expected.task_id, [])]


def test_project_reconciliation_releases_a_stale_cpu_reservation(tmp_path: Path) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "cpu-host")
    set_cpu_lane_capacity(cfg.runtime_root, capacity=1)
    task = submit(
        cfg,
        ["echo", "stale"],
        requested_gpus=0,
        requested_cpus=1,
        working_dir=working_directory,
    )
    attempt = claim_task(cfg, task.task_id, [])
    assert attempt is not None
    task_path(cfg.shared_root, task.task_id).unlink()

    reconcile_project_reservations(cfg)

    _policy, reservations = cpu_reservation_snapshot(cfg.runtime_root)
    assert reservations == ()
