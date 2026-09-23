from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli import entrypoint as cli_entrypoint
from qqtools.plugins.qexp.cli import project_handlers

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _base_args(cfg) -> list[str]:
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["task", "show", "task-1", "--interval-seconds", "2"], "requires --watch"),
        (["task", "show", "task-1", "--follow-retries"], "requires --watch"),
        (["task", "show", "task-1", "--watch", "--format=json"], "cannot be combined"),
        (["task", "logs", "task-1", "--interval-seconds", "2"], "requires --follow"),
        (["task", "logs", "task-1", "--follow-retries"], "requires --follow"),
        (["task", "logs", "task-1", "--follow", "--tail", "-1"], "non-negative"),
        (["task", "logs", "task-1", "--follow", "--tail", "1.5"], "base-10 integer"),
        (["task", "logs", "task-1", "--follow", "--interval-seconds", "nan"], "finite number"),
        (["task", "logs", "task-1", "--follow", "--interval-seconds", "0"], "at least 1"),
    ],
)
def test_continuous_option_errors_precede_project_resolution(argv: list[str], message: str, capsys) -> None:
    assert cli_entrypoint.main(argv) == 2
    captured = capsys.readouterr()
    assert message in (captured.out if "--format=json" in argv else captured.err)


def test_watch_rejects_redirected_stdout_before_project_resolution(capsys) -> None:
    assert cli_entrypoint.main(["task", "show", "task-1", "--watch"]) == 2
    assert "requires terminal stdout" in capsys.readouterr().err


def test_continuous_cli_routes_defaults_and_explicit_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    base = _base_args(cfg)
    calls: list[tuple[str, object, object]] = []

    class TerminalOutput(StringIO):
        def isatty(self) -> bool:
            return True

    terminal = TerminalOutput()
    monkeypatch.setattr(cli_entrypoint.sys, "stdout", terminal)

    def watch(_cfg, task_id, *, interval_seconds, follow_retries, details, observer_attempt_id):
        assert details is False
        assert observer_attempt_id is None
        calls.append((task_id, interval_seconds, follow_retries))
        return 0

    def follow(_cfg, task_id, *, tail_lines, interval_seconds, follow_retries):
        calls.append((task_id, (tail_lines, interval_seconds), follow_retries))
        return 0

    monkeypatch.setattr(project_handlers.watch_commands, "watch_task", watch)
    monkeypatch.setattr(project_handlers.log_commands, "follow_logs", follow)

    assert cli_entrypoint.main([*base, "task", "show", task.task_id, "--watch"]) == 0
    assert (
        cli_entrypoint.main(
            [
                *base,
                "task",
                "logs",
                task.task_id,
                "--follow",
                "--tail",
                "7",
                "--interval-seconds",
                "2.5",
                "--follow-retries",
            ]
        )
        == 0
    )

    assert calls == [(task.task_id, 2, False), (task.task_id, (7, 2.5), True)]


def test_continuous_cli_translates_interruptions_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    base = _base_args(cfg)

    class TerminalOutput(StringIO):
        def isatty(self) -> bool:
            return True

    terminal = TerminalOutput()
    monkeypatch.setattr(cli_entrypoint.sys, "stdout", terminal)
    monkeypatch.setattr(
        project_handlers.watch_commands,
        "watch_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert cli_entrypoint.main([*base, "task", "show", task.task_id, "--watch"]) == 130
    assert terminal.getvalue() == "\x1b[0m\n"

    monkeypatch.setattr(
        project_handlers.log_commands,
        "follow_logs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert cli_entrypoint.main([*base, "task", "logs", task.task_id, "--follow"]) == 130


def test_continuous_help_explains_refresh_tail_and_retry_contract(capsys) -> None:
    with pytest.raises(SystemExit) as show_exit:
        cli_entrypoint.main(["task", "show", "--help"])
    assert show_exit.value.code == 0
    show_help = " ".join(capsys.readouterr().out.split())
    assert "screen refreshes" in show_help
    assert "does not change progress reporting" in show_help
    assert "later retry" in show_help

    with pytest.raises(SystemExit) as logs_exit:
        cli_entrypoint.main(["task", "logs", "--help"])
    assert logs_exit.value.code == 0
    logs_help = " ".join(capsys.readouterr().out.split())
    assert "each Attempt and file generation" in logs_help
    assert "later retry" in logs_help
