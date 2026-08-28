from qqtools.plugins import qexp


def test_public_surface_has_group_task_attempt_without_batch():
    assert callable(qexp.submit)
    assert callable(qexp.batch_submit)
    assert callable(qexp.list_groups)
    assert qexp.Task.__name__ == "TaskRecord"
    assert not hasattr(qexp, "Batch")
    assert not hasattr(qexp, "resubmit")
