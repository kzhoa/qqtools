from qqtools.plugins.qexp.runtime.records import TaskRecord, TaskSpec


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
