import shlex
import threading
import time
from pathlib import Path

from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.executor import Executor, LaunchHandle
from qqtools.plugins.qexp.layout import project_id
from qqtools.plugins.qexp.observer_provisioning import observer_lock, observer_lock_path
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace


class _Process:
    pid = 4321


def _cfg(tmp_path: Path) -> RootConfig:
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    atomic_replace(
        identity_path,
        {"project": {"project_id": project_id(cfg.shared_root), "shared_root": str(cfg.shared_root)}},
    )
    return cfg


def test_slow_window_creation_does_not_block_launch_caller(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def slow_window(*_args):
        entered.set()
        assert release.wait(2)
        return "@17"

    executor = Executor(
        create_window=slow_window,
        tmux_available=lambda: True,
        observer_decision=lambda _cfg, _task_id: {"enabled": True},
        find_observer_window=lambda *_args: None,
        mark_observer_window=lambda *_args: True,
    )
    handle = LaunchHandle("detached", _Process(), runner_process=_Process())
    started = time.monotonic()
    try:
        assert executor.provision_observer(_cfg(tmp_path), "task-1", "attempt-1", handle)
        assert time.monotonic() - started < 0.1
        assert entered.wait(1)
        assert handle.backend == "detached"
    finally:
        release.set()
    deadline = time.monotonic() + 1
    while handle.observer_window_id is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert handle.observer_window_id == "@17"


def test_observer_lock_does_not_write_to_shared_root(tmp_path):
    shared_root = tmp_path / "read-only-shared-root"
    shared_root.mkdir()
    path = observer_lock_path(shared_root, "task-1", "attempt-1")
    assert shared_root not in path.parents
    with observer_lock(shared_root, "task-1", "attempt-1") as acquired:
        assert acquired
        assert path.is_file()
    assert list(shared_root.iterdir()) == []


def test_tagging_failure_removes_only_new_viewer_and_keeps_runner(tmp_path):
    destroyed = []
    handle = LaunchHandle("detached", _Process(), runner_process=_Process())
    executor = Executor(
        create_window=lambda *_args: "@17",
        destroy_window=destroyed.append,
        tmux_available=lambda: True,
        observer_decision=lambda _cfg, _task_id: {"enabled": True},
        find_observer_window=lambda *_args: None,
        mark_observer_window=lambda *_args: False,
    )
    executor.attach_observer(_cfg(tmp_path), "task-1", "attempt-1", handle)
    assert destroyed == ["@17"]
    assert handle.backend == "detached"
    assert handle.observer_window_id is None


def test_lookup_failure_does_not_create_another_viewer(tmp_path):
    def fail_lookup(*_args):
        raise OSError("tmux query failed")

    executor = Executor(
        create_window=lambda *_args: (_ for _ in ()).throw(AssertionError("duplicate viewer")),
        tmux_available=lambda: True,
        observer_decision=lambda _cfg, _task_id: {"enabled": True},
        find_observer_window=fail_lookup,
    )
    handle = LaunchHandle("detached", _Process(), runner_process=_Process())
    executor.attach_observer(_cfg(tmp_path), "task-1", "attempt-1", handle)
    assert handle.observer_window_id is None


def test_selected_viewer_command_parses_project_and_attempt(tmp_path):
    command = Executor.build_live_progress_command(_cfg(tmp_path), "task-1", "attempt-1")
    argv = shlex.split(command)[4:]
    args = build_parser().parse_args(argv)
    assert args.project == str(tmp_path)
    assert args.task_id == "task-1"
    assert args.observer_attempt_id == "attempt-1"
