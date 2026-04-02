import pytest

from qqtools.plugins.qexp.models import (
    TASK_PENDING,
    qExpTask,
    ensure_valid_task_state,
    get_state_directory_name,
)


def test_qexp_task_requires_positive_num_gpus():
    with pytest.raises(ValueError, match="positive integer"):
        qExpTask(
            task_id="job_zero_gpu",
            argv=["python", "train.py"],
            num_gpus=0,
        )


def test_qexp_task_rejects_unknown_state():
    with pytest.raises(ValueError, match="must be one of"):
        ensure_valid_task_state("todo")


def test_qexp_task_serializes_required_phase1_fields():
    task = qExpTask(
        task_id="job_demo",
        argv=["python", "train.py", "--lr", "0.01"],
        num_gpus=1,
        workdir=".",
        status=TASK_PENDING,
        env={
            "kind": "conda",
            "name": "torch",
            "activate_script": "/tmp/conda.sh",
            "extra_env": {"OMP_NUM_THREADS": 8},
        },
    )

    payload = task.to_dict()

    assert payload["task_id"] == "job_demo"
    assert payload["argv"] == ["python", "train.py", "--lr", "0.01"]
    assert payload["num_gpus"] == 1
    assert payload["status"] == TASK_PENDING
    assert payload["created_at"].endswith("Z")
    assert payload["env"]["kind"] == "conda"
    assert payload["env"]["extra_env"]["OMP_NUM_THREADS"] == "8"
    assert get_state_directory_name(payload["status"]) == "pending"
