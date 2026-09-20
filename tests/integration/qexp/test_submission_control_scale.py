"""Bulk Submission size is independent of steady classification and actual claim I/O.

The source retains declarations for cleaned Tasks; one acknowledged Task remains.
Fixture encoding and publication are outside the measured admission window.
"""

import hashlib
import json
import os
import time

import pytest

from qqtools.plugins.qexp import init_shared_root, scheduler, submit
from qqtools.plugins.qexp.runtime import submission_control as control
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.ready import classify_ready_marker, routes
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("retained", [0, 1000, 10_000, 100_000])
def test_bulk_history_has_bounded_admission_reads(tmp_path, monkeypatch, retained, record_property):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"], task_id="survivor")
    path = submission_path(cfg.shared_root, task.submission_operation_id)
    operation = read_json(path)
    submission = operation["submission"]
    context = submission["resolved_context"]
    template = context["task_specs"][0]
    for index in range(retained):
        spec = dict(template, task_id=f"cleaned-{index:06d}")
        context["task_ids"].append(spec["task_id"])
        context["task_specs"].append(spec)
    submission["staged_task_count"] = len(context["task_ids"])
    submission["resolved_context_digest"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    control.publish_submission(cfg, operation)
    size = path.stat().st_size
    ready_ref = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
    assert ready_ref is not None
    original_path_open = type(path).open
    original_os_open = os.open
    source_opens = []

    def check_source(target):
        if os.fspath(target) == os.fspath(path):
            source_opens.append(size)
            assert size <= 65536, "active admission opened an unbounded Submission source"

    def observe_open(target, *args, **kwargs):
        check_source(target)
        return original_path_open(target, *args, **kwargs)

    def observe_os_open(target, *args, **kwargs):
        check_source(target)
        return original_os_open(target, *args, **kwargs)

    begin = time.monotonic()
    with monkeypatch.context() as guard:
        guard.setattr(type(path), "open", observe_open)
        guard.setattr(os, "open", observe_os_open)
        assert classify_ready_marker(cfg, ready_ref).classification == "claimable"
        assert scheduler._eligible(cfg, task)
        attempt = scheduler.claim_task(cfg, task.task_id, [0])
        assert attempt is not None
        assert attempt.task_id == task.task_id
    record_property("retained_submission_members", retained)
    record_property("source_bytes", size)
    record_property("admission_source_bytes", sum(source_opens))
    record_property("classification_and_claim_seconds", time.monotonic() - begin)
    if retained:
        assert not source_opens
