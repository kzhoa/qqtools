from __future__ import annotations

import importlib
import shutil

TMUX_SESSION_INTERNAL = "qqtools_internal"
TMUX_SESSION_EXPERIMENTS = "experiments"


def is_tmux_executable_available() -> bool:
    return shutil.which("tmux") is not None


def require_libtmux():
    try:
        return importlib.import_module("libtmux")
    except Exception as exc:
        raise RuntimeError(
            "libtmux is required for qexp. Install optional dependencies with "
            "'pip install qqtools[exp]'."
        ) from exc


def create_window_for_task(task_id: str, session_name: str = TMUX_SESSION_EXPERIMENTS) -> str:
    libtmux = require_libtmux()
    server = libtmux.Server()
    session = server.sessions.get(session_name=session_name)
    if session is None:
        session = server.new_session(session_name=session_name, detached=True)

    window_name = task_id[:48]
    window = session.new_window(window_name=window_name, attach=False)
    return str(window.window_id)


def window_exists(window_id: str | None) -> bool:
    if not window_id:
        return False

    libtmux = require_libtmux()
    server = libtmux.Server()
    return server.windows.get(window_id=window_id) is not None


def kill_window(window_id: str | None) -> None:
    if not window_id:
        return

    libtmux = require_libtmux()
    server = libtmux.Server()
    window = server.windows.get(window_id=window_id)
    if window is not None:
        window.kill_window()
