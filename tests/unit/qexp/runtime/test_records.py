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


def test_legacy_group_worker_defaults_to_primary():
    group = {"group": {"worker_set": {"gpu-1": {"state": "active"}}}}

    normalize_group_record(group)

    assert group["group"]["worker_set"]["gpu-1"]["scheduling_role"] == "primary"
    assert group["group"]["worker_set"]["gpu-1"]["borrow_limit_gpus"] is None


def test_canonical_borrow_worker_encoding_is_readable():
    group = {
        "group": {
            "worker_set": {
                "gpu-1": {
                    "state": "active",
                    "scheduling_role": "borrow",
                    "borrow_limit_gpus": 2,
                }
            }
        }
    }

    normalize_group_record(group)

    assert group["group"]["worker_set"]["gpu-1"]["scheduling_role"] == "borrow"
