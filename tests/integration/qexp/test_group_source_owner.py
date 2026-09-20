"""Machine source sessions survive passes and close without executor races."""

from __future__ import annotations

import json
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.session_owner import GroupSourceOwner, SourceRequest
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO

pytestmark = pytest.mark.integration


class ControlledDriver:
    def __init__(self, *, enter=None, resume=None):
        self.enter = enter
        self.resume = resume
        self.is_closed = False
        self.summary = None
        self.closing = False
        self.checkpoints = 0
        self.calls = []

    def request_checkpoint(self):
        self.checkpoints += 1

    def request_close(self):
        self.closing = True

    def advance(self, io, **kwargs):
        self.calls.append((self.closing, io, current_thread().name, kwargs))
        if self.closing:
            self.is_closed = True
            return SimpleNamespace(state="closed", reason=None)
        if self.enter is not None:
            self.enter.set()
            assert self.resume.wait(10), "test did not release the source executor"
        return SimpleNamespace(state="progressed", reason=None)


def request(tmp_path, suffix="a"):
    return SourceRequest(
        project_id=f"project-{suffix}",
        source=tmp_path / f"source-{suffix}.json",
        scratch=tmp_path / f"scratch-{suffix}",
        operation_id=f"op-{suffix}",
        group="exp",
    )


def close_owner(owner):
    owner.shutdown()
    assert owner.wait_closed(10), "source owner left cleanup pending"
    assert owner.cleanup_error is None


def test_resident_session_reuses_driver_and_caller_budget_until_release(tmp_path):
    created = []

    def factory(source_request):
        driver = ControlledDriver()
        created.append((source_request, driver))
        return driver

    owner = GroupSourceOwner(driver_factory=factory)
    first, other = request(tmp_path), request(tmp_path, "b")
    first_io, second_io = SliceIO(), SliceIO()
    try:
        assert owner.advance(first, first_io).state == "progressed"
        assert owner.advance(first, second_io).state == "progressed"
        assert len(created) == 1
        driver = created[0][1]
        assert driver.calls[0][1] is first_io
        assert driver.calls[1][1] is second_io
        assert owner.current_request == first
        blocked = owner.advance(other, SliceIO())
        assert (blocked.state, blocked.reason) == ("waiting", "source_busy")
        assert len(driver.calls) == 2
        assert owner.request_checkpoint(first)
        assert driver.checkpoints == 1
        release_io = SliceIO()
        assert owner.release(release_io).state == "closed"
        assert driver.calls[-1][1] is release_io
        assert driver.is_closed
        assert owner.current_request is None
        assert owner.advance(other, SliceIO()).state == "progressed"
        assert len(created) == 2
        assert created[1][0] == other
    finally:
        close_owner(owner)


def test_shutdown_hands_cleanup_to_blocked_executor_without_overlap(tmp_path):
    entered, resume = Event(), Event()
    driver = ControlledDriver(enter=entered, resume=resume)
    owner = GroupSourceOwner(driver_factory=lambda _: driver)
    selected = request(tmp_path)
    errors = []

    def advance():
        try:
            owner.advance(selected, SliceIO())
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=advance, name="test-source-executor", daemon=True)
    thread.start()
    try:
        assert entered.wait(10)
        second = owner.advance(selected, SliceIO())
        assert (second.state, second.reason) == ("waiting", "executor_busy")
        assert not owner.request_checkpoint(selected)
        owner.shutdown()
        owner.shutdown()
        assert not owner.wait_closed(0)
        assert not driver.closing
        assert len(driver.calls) == 1
        rejected = owner.advance(request(tmp_path, "b"), SliceIO())
        assert rejected.state == "waiting"
        assert len(driver.calls) == 1
    finally:
        resume.set()
        thread.join(10)
        close_owner(owner)
    assert not thread.is_alive()
    assert not errors
    assert [call[0] for call in driver.calls] == [False, True]
    assert {call[2] for call in driver.calls} == {"test-source-executor"}
    assert driver.is_closed


def test_idle_shutdown_uses_cleanup_only_executor_and_never_opens_another_source(tmp_path):
    driver = ControlledDriver()
    created = []

    def factory(selected):
        created.append(selected)
        return driver

    owner = GroupSourceOwner(driver_factory=factory)
    owner.advance(request(tmp_path), SliceIO())
    close_owner(owner)
    owner.shutdown()
    assert owner.is_closed
    assert [call[0] for call in driver.calls] == [False, True]
    assert driver.calls[-1][2] == "qexp-group-source-close"
    assert owner.advance(request(tmp_path, "b"), SliceIO()).state == "closed"
    assert len(created) == 1


def test_shutdown_before_first_request_does_not_construct_a_driver(tmp_path):
    def unexpected_factory(_):
        pytest.fail("shutdown owner constructed a source driver")

    owner = GroupSourceOwner(driver_factory=unexpected_factory)
    close_owner(owner)
    assert owner.advance(request(tmp_path), SliceIO()).state == "closed"


def test_machine_worker_stop_closes_its_runtime_source_owner(tmp_path):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.runtime.upgrade.machine import MachineUpgradeWorker

    runtime = MachineRuntime(tmp_path / "machine")
    owner = runtime.group_source_owner
    assert isinstance(owner, GroupSourceOwner)
    assert not owner.is_closed
    first = MachineUpgradeWorker(runtime)
    second = MachineUpgradeWorker(runtime)
    assert first.runtime.group_source_owner is second.runtime.group_source_owner is owner
    first.stop()
    assert owner.wait_closed(10)
    assert owner.cleanup_error is None


def test_runtime_owner_drives_real_source_and_reopens_durable_completion(tmp_path):
    selected = request(tmp_path)
    selected.source.write_text(
        json.dumps(
            {
                "meta": {"schema_version": 6},
                "submission": {
                    "operation_id": selected.operation_id,
                    "target_group": selected.group,
                    "state": "committed",
                    "resolved_context": {"task_ids": ["task-a", "task-b"]},
                    "commit_plan": {"group_membership_sequences": [1, 2]},
                },
            }
        )
    )
    owner = GroupSourceOwner()
    try:
        for _ in range(2_000):
            result = owner.advance(selected, SliceIO(max_io_bytes=128, max_operations=2), max_processed_bytes=31)
            if result.state == "complete":
                break
        else:
            pytest.fail("runtime source did not complete through budgeted passes")
        assert result.summary is not None
        summary = result.summary
        events = (selected.scratch / "events.jsonl").read_bytes()
        assert events
    finally:
        close_owner(owner)

    resumed = GroupSourceOwner()
    try:
        for _ in range(2_000):
            result = resumed.advance(selected, SliceIO(max_io_bytes=128, max_operations=2))
            if result.state == "complete":
                break
        else:
            pytest.fail("runtime source did not restore its durable completion")
        assert result.summary == summary
        assert (selected.scratch / "events.jsonl").read_bytes() == events
    finally:
        close_owner(resumed)
