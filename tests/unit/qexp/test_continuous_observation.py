from __future__ import annotations

from copy import deepcopy
from io import BytesIO, StringIO
from pathlib import Path

import pytest

from qqtools.plugins.qexp import observer
from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.commands import logs
from qqtools.plugins.qexp.commands.logs import follow_logs
from qqtools.plugins.qexp.commands.watch import watch_task
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.records import AttemptRecord, TaskRecord, TaskSpec


def _cfg(tmp_path: Path) -> RootConfig:
    return RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")


@pytest.mark.parametrize("watch", [False, True])
def test_task_show_details_combines_with_finite_and_watch_modes(watch: bool) -> None:
    argv = ["task", "show", "task-1", "--details"]
    if watch:
        argv.append("--watch")
    args = build_parser().parse_args(argv)
    assert args.details is True
    assert args.watch is watch


def _records(tmp_path: Path) -> tuple[TaskRecord, AttemptRecord]:
    task = TaskRecord.new(
        task_id="task-1",
        machine="gpu-1",
        name="demo",
        spec=TaskSpec(["python", "train.py"], str(tmp_path), 1),
    )
    attempt = AttemptRecord.claimed(
        task,
        "gpu-1",
        [0],
        "reservation-1",
        7,
        authority_mode="holder_bound",
        clock_evidence=None,
        attempt_id="attempt-1",
    )
    attempt.phase = "running"
    task.state = {"projection": "running", "reason": None}
    task.attempt_control.update(current_attempt_id=attempt.attempt_id, current_attempt_number=1)
    task.claim_control["active_claim"] = {
        "attempt_id": attempt.attempt_id,
        "attempt_number": 1,
        "fencing_token": attempt.current_fencing_token,
        "machine_name": attempt.machine_name,
    }
    return task, attempt


def _progress(state: str = "no_report", reason: str = "no_snapshot") -> dict[str, object]:
    return {"status": "unavailable", "observation_state": state, "reason": reason}


def _follow_payload(path: Path, *, terminal: bool, attempt: int = 1) -> dict[str, object]:
    phase = "succeeded" if terminal else "running"
    return {
        "task_id": "task-1",
        "name": "demo",
        "phase": phase,
        "reason": None,
        "revision": attempt,
        "terminal": terminal,
        "observation_state": "no_report",
        "observation_reason": "no_snapshot",
        "selected_attempt": {
            "attempt_id": f"attempt-{attempt}",
            "attempt_number": attempt,
            "phase": phase,
            "machine_name": "gpu-1",
            "log_path": str(path),
        },
        "progress": _progress(),
    }


def test_current_observation_selects_exact_running_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, attempt = _records(tmp_path)
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(observer, "read_json", lambda *_args: attempt.to_dict())
    monkeypatch.setattr(observer, "inspect_progress", lambda *_args: _progress())

    view = observer.inspect_current_task(_cfg(tmp_path), task.task_id)

    assert view["selected_attempt"] == {
        "attempt_id": "attempt-1",
        "attempt_number": 1,
        "phase": "running",
        "machine_name": "gpu-1",
        "log_path": str(tmp_path / ".qexp" / "logs" / "task-1" / "attempt-1.log"),
    }
    assert view["phase"] == "running"


def test_queued_retry_never_selects_retained_attempt_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, _attempt = _records(tmp_path)
    task.state = {"projection": "queued", "reason": "retry_requested"}
    task.attempt_control["current_attempt_id"] = None
    task.claim_control["active_claim"] = None
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(
        observer,
        "read_json",
        lambda *_args: (_ for _ in ()).throw(AssertionError("queued retry must not read historical Attempt")),
    )
    monkeypatch.setattr(observer, "inspect_progress", lambda *_args: _progress("pending", "not_started"))

    view = observer.inspect_current_task(_cfg(tmp_path), task.task_id)

    assert view["selected_attempt"] is None
    assert view["progress"]["observation_state"] == "pending"


def test_terminal_selection_uses_preserved_number_with_cleared_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, attempt = _records(tmp_path)
    task.state = {"projection": "succeeded", "reason": None}
    task.attempt_control["current_attempt_id"] = None
    task.claim_control["active_claim"] = None
    attempt.phase = "succeeded"
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(observer, "read_json", lambda *_args: attempt.to_dict())
    monkeypatch.setattr(observer, "inspect_progress", lambda *_args: _progress())

    view = observer.inspect_current_task(_cfg(tmp_path), task.task_id)

    assert view["terminal"] is True
    assert view["selected_attempt"]["attempt_id"] == attempt.attempt_id


@pytest.mark.parametrize("retain_id", [True, False])
def test_blocked_orphan_selection_requires_exact_orphan_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retain_id: bool
) -> None:
    task, attempt = _records(tmp_path)
    task.state = {"projection": "blocked", "reason": "orphaned_attempt_requires_recovery"}
    task.claim_control["active_claim"] = None
    if not retain_id:
        task.attempt_control["current_attempt_id"] = None
    attempt.phase = "orphaned"
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(observer, "read_json", lambda *_args: attempt.to_dict())
    monkeypatch.setattr(observer, "inspect_progress", lambda *_args: _progress("unavailable", "identity_mismatch"))

    view = observer.inspect_current_task(_cfg(tmp_path), task.task_id)

    assert view["selected_attempt"]["attempt_id"] == attempt.attempt_id
    assert view["selected_attempt"]["phase"] == "orphaned"


def test_selection_discards_candidate_when_task_changes_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, attempt = _records(tmp_path)
    changed = deepcopy(first)
    changed.meta["revision"] += 1
    changed.state = {"projection": "queued", "reason": "retry_requested"}
    changed.attempt_control["current_attempt_id"] = None
    changed.claim_control["active_claim"] = None
    tasks = iter((first, changed))
    monkeypatch.setattr(observer, "load_task", lambda *_args: next(tasks))
    monkeypatch.setattr(observer, "read_json", lambda *_args: attempt.to_dict())
    monkeypatch.setattr(
        observer,
        "inspect_progress",
        lambda *_args: (_ for _ in ()).throw(AssertionError("transition frame must not inspect old progress")),
    )

    view = observer.inspect_current_task(_cfg(tmp_path), first.task_id)

    assert view["selected_attempt"] is None
    assert view["observation_reason"] == "concurrent_transition"
    assert view["phase"] == "queued"


def test_stable_missing_selected_attempt_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, _attempt = _records(tmp_path)
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(observer, "read_json", lambda *_args: (_ for _ in ()).throw(FileNotFoundError()))

    with pytest.raises(observer.CurrentObservationError, match="was not found"):
        observer.inspect_current_task(_cfg(tmp_path), task.task_id)


def test_running_selection_rejects_stable_claim_attempt_number_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, attempt = _records(tmp_path)
    task.claim_control["active_claim"]["attempt_number"] = 2
    monkeypatch.setattr(observer, "load_task", lambda *_args: task)
    monkeypatch.setattr(observer, "read_json", lambda *_args: attempt.to_dict())

    with pytest.raises(observer.CurrentObservationError, match="inconsistent active claim"):
        observer.inspect_current_task(_cfg(tmp_path), task.task_id)


def test_watch_renders_queued_retry_without_old_attempt_and_exits_on_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {
        "task_id": "task-1",
        "name": "demo",
        "reason": None,
        "revision": 1,
        "observation_state": "pending",
        "observation_reason": "not_started",
        "selected_attempt": None,
        "progress": _progress("pending", "not_started"),
    }
    frames = iter(
        (
            {**base, "phase": "queued", "terminal": False},
            {
                **base,
                "phase": "succeeded",
                "terminal": True,
                "observation_state": "no_report",
                "observation_reason": "no_snapshot",
                "selected_attempt": {
                    "attempt_id": "attempt-2",
                    "attempt_number": 2,
                    "phase": "succeeded",
                    "machine_name": "gpu-1",
                    "log_path": "/tmp/attempt-2.log",
                },
            },
        )
    )
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: next(frames))
    output = StringIO()
    sleeps: list[float] = []

    assert watch_task(_cfg(tmp_path), "task-1", stdout=output, sleep=sleeps.append) == 0

    assert sleeps == [2]
    assert output.getvalue().count("Task ID: task-1") == 2
    assert "Attempt: none" in output.getvalue()
    assert "Attempt ID: attempt-2" in output.getvalue()
    assert "\x1b" not in output.getvalue()


def test_non_cursor_watch_appends_only_on_snapshot_or_task_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = {
        "task_id": "task-1",
        "name": "demo",
        "phase": "running",
        "reason": None,
        "revision": 1,
        "terminal": False,
        "observation_state": "no_report",
        "observation_reason": "no_snapshot",
        "selected_attempt": None,
        "progress": _progress(),
    }
    frames = iter((base, dict(base), {**base, "phase": "succeeded", "terminal": True}))
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: next(frames))
    output = StringIO()
    sleeps: list[float] = []

    assert watch_task(_cfg(tmp_path), "task-1", stdout=output, sleep=sleeps.append) == 0

    assert sleeps == [2, 2]
    assert output.getvalue().count("Task ID: task-1") == 2
    assert "\x1b" not in output.getvalue()


def test_watch_preserves_project_header_and_marks_only_the_first_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "task_id": "task-1",
        "name": "demo",
        "phase": "running",
        "reason": None,
        "revision": 1,
        "terminal": False,
        "observation_state": "no_report",
        "observation_reason": "no_snapshot",
        "selected_attempt": None,
        "progress": _progress(),
    }
    frames = iter((base, {**base, "phase": "succeeded", "terminal": True}))
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: next(frames))
    output = StringIO()
    presented: list[str] = []

    assert (
        watch_task(
            _cfg(tmp_path),
            "task-1",
            project_line="Project: /work/a (from parent directory)",
            on_first_frame_rendered=lambda: presented.append("first"),
            stdout=output,
            sleep=lambda _seconds: None,
        )
        == 0
    )

    assert output.getvalue().count("Project: /work/a (from parent directory)") == 2
    assert presented == ["first"]


def test_attempt_pinned_viewer_exits_before_showing_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = _follow_payload(tmp_path / "task.log", terminal=False)
    newer = _follow_payload(tmp_path / "task.log", terminal=False, attempt=2)
    frames = iter((base, newer))
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: next(frames))
    output = StringIO()
    sleeps: list[float] = []

    assert (
        watch_task(
            _cfg(tmp_path),
            "task-1",
            observer_attempt_id="attempt-1",
            stdout=output,
            sleep=sleeps.append,
        )
        == 0
    )

    assert sleeps == [2]
    assert "Attempt ID: attempt-1" in output.getvalue()
    assert "attempt-2" not in output.getvalue()


@pytest.mark.parametrize(
    ("content", "lines", "expected"),
    [
        (b"one\ntwo\nthree\n", 2, b"two\nthree\n"),
        (b"one\ntwo\nthree", 2, b"two\nthree"),
        (b"one\n", 1, b"one\n"),
        (b"one\n", 0, b""),
        (b"single-long-line", 3, b"single-long-line"),
    ],
)
def test_tail_offset_selects_logical_lines_without_terminal_empty_line(
    content: bytes, lines: int, expected: bytes
) -> None:
    stream = BytesIO(content)

    offset = logs._tail_offset(stream, len(content), lines, 3)

    assert content[offset:] == expected


def test_follow_terminal_log_tails_and_decodes_split_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_bytes("old\nkeep-α\nlast-β".encode())
    payload = _follow_payload(path, terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)
    stdout, stderr = StringIO(), StringIO()

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            tail_lines=2,
            stdout=stdout,
            stderr=stderr,
            sleep=lambda _seconds: pytest.fail("terminal drain must not sleep"),
            chunk_size=1,
        )
        == 0
    )

    assert stdout.getvalue() == "keep-α\nlast-β"
    assert stderr.getvalue() == ""


def test_follow_discards_chunk_when_attempt_changes_before_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text("must-not-leak\n")
    second.write_text("new-attempt\n")
    first_payload = _follow_payload(first, terminal=False, attempt=1)
    second_payload = _follow_payload(second, terminal=True, attempt=2)
    calls = 0

    def inspect(*_args):
        nonlocal calls
        calls += 1
        return first_payload if calls <= 2 else second_payload

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    stdout, stderr = StringIO(), StringIO()

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            tail_lines=100,
            stdout=stdout,
            stderr=stderr,
            sleep=lambda _seconds: pytest.fail("Attempt switch must not sleep"),
            chunk_size=64,
        )
        == 0
    )

    assert stdout.getvalue() == "new-attempt\n"
    assert "must-not-leak" not in stdout.getvalue()
    assert "Attempt boundary" in stderr.getvalue()


def test_follow_discards_chunk_when_post_read_observation_retries_then_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text("must-not-leak\n")
    second.write_text("new-attempt\n")
    first_payload = _follow_payload(first, terminal=False, attempt=1)
    second_payload = _follow_payload(second, terminal=True, attempt=2)
    calls = 0

    def inspect(*_args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise observer.CurrentObservationError("publication in progress")
        return first_payload if calls <= 2 else second_payload

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    stdout, stderr = StringIO(), StringIO()
    sleeps: list[float] = []

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            tail_lines=100,
            stdout=stdout,
            stderr=stderr,
            sleep=sleeps.append,
            chunk_size=64,
        )
        == 0
    )

    assert stdout.getvalue() == "new-attempt\n"
    assert "must-not-leak" not in stdout.getvalue()
    assert sleeps == [2]
    assert "temporarily inconsistent" in stderr.getvalue()
    assert "Attempt boundary" in stderr.getvalue()


def test_terminal_concurrent_transition_reselects_before_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "terminal.log"
    path.write_text("final-byte\n")
    transition = {
        **_follow_payload(path, terminal=True),
        "revision": 1,
        "selected_attempt": None,
        "observation_state": "unavailable",
        "observation_reason": "concurrent_transition",
    }
    stable = _follow_payload(path, terminal=True)
    frames = iter((transition, stable, stable, stable))
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: next(frames))
    stdout, stderr = StringIO(), StringIO()
    sleeps: list[float] = []

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=stdout,
            stderr=stderr,
            sleep=sleeps.append,
        )
        == 0
    )

    assert stdout.getvalue() == "final-byte\n"
    assert sleeps == [2]
    assert "retrying selection" in stderr.getvalue()


def test_follow_retries_transient_inconsistent_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "terminal.log"
    path.write_text("final-byte\n")
    stable = _follow_payload(path, terminal=True)
    frames = iter(
        (
            observer.CurrentObservationError("running Task has inconsistent active claim"),
            stable,
            stable,
            stable,
        )
    )

    def inspect(*_args):
        frame = next(frames)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    stdout, stderr = StringIO(), StringIO()
    sleeps: list[float] = []

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=stdout,
            stderr=stderr,
            sleep=sleeps.append,
        )
        == 0
    )

    assert stdout.getvalue() == "final-byte\n"
    assert sleeps == [2]
    assert "temporarily inconsistent; retrying selection" in stderr.getvalue()


def test_follow_rejects_persistently_inconsistent_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def inspect(*_args):
        raise observer.CurrentObservationError("stable corruption")

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    sleeps: list[float] = []

    with pytest.raises(observer.CurrentObservationError, match="stable corruption"):
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=StringIO(),
            stderr=StringIO(),
            sleep=sleeps.append,
        )

    assert sleeps == [2, 2, 2]


def test_follow_reopens_replaced_generation_with_tail_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_text("old-a\nold-b\n")
    state = {"terminal": False, "replaced": False}

    def inspect(*_args):
        return _follow_payload(path, terminal=state["terminal"])

    def replace_on_wait(_seconds: float) -> None:
        assert not state["replaced"]
        replacement = tmp_path / "replacement.log"
        replacement.write_text("new-a\nnew-b\n")
        replacement.replace(path)
        state.update(terminal=True, replaced=True)

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    stdout, stderr = StringIO(), StringIO()

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            tail_lines=1,
            stdout=stdout,
            stderr=stderr,
            sleep=replace_on_wait,
            chunk_size=4,
        )
        == 0
    )

    assert stdout.getvalue() == "old-b\nnew-b\n"
    assert "prefix may be skipped" in stderr.getvalue()
    assert "cannot be recovered" in stderr.getvalue()


def test_terminal_missing_log_retries_once_then_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _follow_payload(tmp_path / "missing.log", terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="after one terminal retry"):
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=StringIO(),
            stderr=StringIO(),
            sleep=sleeps.append,
        )

    assert sleeps == [2]


def test_closed_downstream_pipe_returns_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_text("application output\n")
    payload = _follow_payload(path, terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)

    class ClosedPipe(StringIO):
        def write(self, _text: str) -> int:
            raise BrokenPipeError

    assert follow_logs(_cfg(tmp_path), "task-1", stdout=ClosedPipe(), stderr=StringIO()) == 0


def test_follow_waits_for_late_log_creation_with_coalesced_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "late.log"
    state = {"terminal": False, "sleeps": 0}
    monkeypatch.setattr(
        observer,
        "inspect_current_task",
        lambda *_args: _follow_payload(path, terminal=state["terminal"]),
    )

    def create_after_two_waits(_seconds: float) -> None:
        state["sleeps"] += 1
        if state["sleeps"] == 2:
            path.write_text("late output\n")
            state["terminal"] = True

    stdout, stderr = StringIO(), StringIO()
    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=stdout,
            stderr=stderr,
            sleep=create_after_two_waits,
        )
        == 0
    )

    assert state["sleeps"] == 2
    assert stdout.getvalue() == "late output\n"
    assert stderr.getvalue().count("waiting for log") == 1


def test_follow_reports_transient_recovery_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_text("recovered output\n")
    state = {"terminal": False, "opens": 0}
    monkeypatch.setattr(
        observer,
        "inspect_current_task",
        lambda *_args: _follow_payload(path, terminal=state["terminal"]),
    )
    real_open = logs._open_attachment

    def fail_once(*args, **kwargs):
        state["opens"] += 1
        if state["opens"] == 1:
            raise OSError("temporary storage failure")
        return real_open(*args, **kwargs)

    def make_terminal(_seconds: float) -> None:
        state["terminal"] = True

    monkeypatch.setattr(logs, "_open_attachment", fail_once)
    stdout, stderr = StringIO(), StringIO()

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=stdout,
            stderr=stderr,
            sleep=make_terminal,
        )
        == 0
    )

    assert stdout.getvalue() == "recovered output\n"
    assert stderr.getvalue().count("temporarily unavailable") == 1
    assert stderr.getvalue().count("recovered") == 1


def test_attached_metadata_error_preserves_tail_zero_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_text("old-before-attach\n")
    state = {"terminal": False, "checks": 0}
    monkeypatch.setattr(
        observer,
        "inspect_current_task",
        lambda *_args: _follow_payload(path, terminal=state["terminal"]),
    )
    real_status = logs._generation_status

    def fail_first_attached_check(attachment):
        state["checks"] += 1
        if state["checks"] == 1:
            raise OSError("temporary metadata failure")
        return real_status(attachment)

    def append_during_retry(_seconds: float) -> None:
        with path.open("a") as stream:
            stream.write("new-after-attach\n")
        state["terminal"] = True

    monkeypatch.setattr(logs, "_generation_status", fail_first_attached_check)
    stdout, stderr = StringIO(), StringIO()

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            tail_lines=0,
            stdout=stdout,
            stderr=stderr,
            sleep=append_during_retry,
        )
        == 0
    )

    assert stdout.getvalue() == "new-after-attach\n"
    assert stderr.getvalue().count("temporarily unavailable") == 1
    assert stderr.getvalue().count("recovered") == 1


def test_post_read_metadata_error_retries_same_bytes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_text("exactly-once\n")
    payload = _follow_payload(path, terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)
    real_status = logs._generation_status
    checks = 0

    def fail_after_first_read(attachment):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise OSError("post-read metadata failure")
        return real_status(attachment)

    monkeypatch.setattr(logs, "_generation_status", fail_after_first_read)
    stdout, stderr = StringIO(), StringIO()
    sleeps: list[float] = []

    assert (
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            stdout=stdout,
            stderr=stderr,
            sleep=sleeps.append,
        )
        == 0
    )

    assert stdout.getvalue() == "exactly-once\n"
    assert sleeps == [2]
    assert stderr.getvalue().count("recovered") == 1


def test_follow_retries_crosses_terminal_queued_and_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text("first terminal\n")
    second.write_text("second terminal\n")
    state = {"phase": "first", "sleeps": 0}
    queued = {
        **_follow_payload(first, terminal=False),
        "phase": "queued",
        "selected_attempt": None,
        "observation_state": "pending",
        "observation_reason": "not_started",
    }

    def inspect(*_args):
        if state["phase"] == "first":
            return _follow_payload(first, terminal=True, attempt=1)
        if state["phase"] == "queued":
            return queued
        return _follow_payload(second, terminal=True, attempt=2)

    class StopFollow(Exception):
        pass

    def advance(_seconds: float) -> None:
        state["sleeps"] += 1
        if state["sleeps"] == 1:
            state["phase"] = "queued"
        elif state["sleeps"] == 2:
            state["phase"] = "second"
        else:
            raise StopFollow

    monkeypatch.setattr(observer, "inspect_current_task", inspect)
    stdout, stderr = StringIO(), StringIO()

    with pytest.raises(StopFollow):
        follow_logs(
            _cfg(tmp_path),
            "task-1",
            follow_retries=True,
            stdout=stdout,
            stderr=stderr,
            sleep=advance,
        )

    assert stdout.getvalue() == "first terminal\nsecond terminal\n"
    assert "no longer selected" in stderr.getvalue()
    assert "following current Attempt attempt-2" in stderr.getvalue()


def test_follow_rejects_nonregular_log_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _follow_payload(tmp_path, terminal=False)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)

    with pytest.raises(RuntimeError, match="not a regular file"):
        follow_logs(_cfg(tmp_path), "task-1", stdout=StringIO(), stderr=StringIO())


def test_follow_replaces_invalid_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    path.write_bytes(b"before\xffafter\n")
    payload = _follow_payload(path, terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: payload)
    stdout = StringIO()

    assert follow_logs(_cfg(tmp_path), "task-1", stdout=stdout, stderr=StringIO(), chunk_size=2) == 0
    assert stdout.getvalue() == "before�after\n"


def test_follow_preserves_invalid_utf8_for_binary_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "attempt.log"
    payload = b"before\xffafter\n"
    path.write_bytes(payload)
    observation = _follow_payload(path, terminal=True)
    monkeypatch.setattr(observer, "inspect_current_task", lambda *_args: observation)
    stdout = BytesIO()

    assert follow_logs(_cfg(tmp_path), "task-1", stdout=stdout, stderr=StringIO(), chunk_size=2) == 0
    assert stdout.getvalue() == payload
