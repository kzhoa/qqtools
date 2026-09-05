import pytest

from qqtools.plugins.qexp.runtime.records import TaskRecord, TaskSpec, normalize_group_record


def test_task_record_has_no_batch_or_retry_truth(tmp_path):
    task = TaskRecord.new(
        task_id="task-1",
        machine="gpu-1",
        spec=TaskSpec(["echo", "ok"], str(tmp_path), 1),
    )
    payload = task.to_dict()["task"]
    assert "batch_id" not in payload
    assert "lineage" not in payload
    assert payload["placement_policy"]["home_machine"] == "gpu-1"


def test_private_task_cannot_become_shared(tmp_path):
    task = TaskRecord.new(task_id="task-1", machine="gpu-1", spec=TaskSpec(["echo"], str(tmp_path), 1))
    task.placement_runtime["queue_scope"] = "shared"
    try:
        TaskRecord.from_dict(task.to_dict())
    except ValueError as exc:
        assert "private" in str(exc)
    else:
        raise AssertionError("private task was accepted in shared scope")


def test_group_worker_requires_canonical_fields():
    group = {"group": {"worker_set": {"gpu-1": {"state": "active"}}}}

    with pytest.raises(ValueError, match="missing required fields"):
        normalize_group_record(group)


def test_group_worker_rejects_legacy_limit_field():
    group = {
        "group": {
            "worker_set": {
                "gpu-1": {
                    "state": "active",
                    "scheduling_role": "borrow",
                    "gpu_limit_gpus": 2,
                    "borrow_limit_gpus": 2,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="obsolete borrow_limit_gpus"):
        normalize_group_record(group)


def test_group_worker_rejects_legacy_borrow_state():
    group = {
        "group": {
            "worker_set": {
                "gpu-1": {
                    "state": "borrow",
                    "scheduling_role": "borrow",
                    "gpu_limit_gpus": 2,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Worker state is invalid"):
        normalize_group_record(group)


def test_canonical_borrow_worker_encoding_is_readable():
    group = {
        "group": {
            "worker_set": {
                "gpu-1": {
                    "state": "active",
                    "scheduling_role": "borrow",
                    "gpu_limit_gpus": 2,
                }
            }
        }
    }

    normalize_group_record(group)

    assert group["group"]["worker_set"]["gpu-1"]["scheduling_role"] == "borrow"
