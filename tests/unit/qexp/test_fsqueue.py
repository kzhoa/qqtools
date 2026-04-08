import json

import pytest

from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp.models import TASK_DONE, TASK_PENDING, qExpTask


@pytest.fixture
def qexp_root(tmp_path, monkeypatch):
    root = tmp_path / "qexp-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    return root


def test_ensure_qexp_layout_bootstraps_all_phase1_directories(qexp_root):
    fsqueue.ensure_qexp_layout()

    assert qexp_root.joinpath("jobs", "logs").is_dir()
    for state in ("pending", "running", "done", "failed", "cancelled"):
        assert qexp_root.joinpath("jobs", state).is_dir()


def test_iter_tasks_ignores_tmp_files(qexp_root):
    fsqueue.ensure_qexp_layout()
    pending_dir = qexp_root / "jobs" / "pending"
    pending_dir.joinpath("half_written.json.tmp").write_text("{}", encoding="utf-8")

    task = qExpTask(task_id="job_valid", argv=["python", "train.py"], num_gpus=1)
    fsqueue.save_task(task)

    tasks = fsqueue.iter_tasks(TASK_PENDING)

    assert [item.task_id for item in tasks] == ["job_valid"]


def test_move_task_keeps_state_and_directory_aligned(qexp_root):
    task = qExpTask(task_id="job_move", argv=["python", "train.py"], num_gpus=1)
    fsqueue.save_task(task)

    fsqueue.move_task(task.task_id, TASK_PENDING, "running")
    moved_path = fsqueue.move_task(task.task_id, "running", TASK_DONE)
    moved_task = fsqueue.load_task(moved_path)

    assert moved_path == qexp_root / "jobs" / "done" / "job_move.json"
    assert moved_task.status == TASK_DONE
    assert not (qexp_root / "jobs" / "pending" / "job_move.json").exists()


def test_load_task_rejects_directory_state_mismatch(qexp_root):
    fsqueue.ensure_qexp_layout()
    invalid_path = qexp_root / "jobs" / "done" / "job_bad.json"
    invalid_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "task_id": "job_bad",
                "argv": ["python", "train.py"],
                "num_gpus": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match directory"):
        fsqueue.load_task(invalid_path)


def test_move_task_rejects_illegal_state_transition(qexp_root):
    task = qExpTask(task_id="job_illegal", argv=["python", "train.py"], num_gpus=1)
    fsqueue.save_task(task)
    fsqueue.move_task(task.task_id, TASK_PENDING, "running")

    with pytest.raises(ValueError, match="Illegal task state transition"):
        fsqueue.move_task(task.task_id, "running", TASK_PENDING)
