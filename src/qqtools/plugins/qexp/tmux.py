from __future__ import annotations

import importlib
import shlex
import shutil
import sys
from pathlib import Path

TMUX_SESSION_INTERNAL = "qqtools_internal"
TMUX_SESSION_EXPERIMENTS = "experiments"
QQTOOLS_SESSION_ROLE_OPTION = "@qqtools_role"
QQTOOLS_SESSION_ROLE_INTERNAL = "internal"
QQTOOLS_SESSION_ROLE_EXPERIMENTS = "experiments"


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


def _get_server():
    libtmux = require_libtmux()
    return libtmux.Server()


def _normalize_option_value(value):
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value or None
    return value


def _get_session_role(session) -> str | None:
    try:
        return _normalize_option_value(session.show_option(QQTOOLS_SESSION_ROLE_OPTION))
    except Exception:
        return None


def _mark_session_role(session, role: str) -> None:
    session.set_option(QQTOOLS_SESSION_ROLE_OPTION, role)


def _safe_get(query_list, **kwargs):
    try:
        return query_list.get(**kwargs)
    except Exception:
        return None


def ensure_managed_session(
    session_name: str,
    role: str,
    *,
    initial_window_name: str,
):
    server = _get_server()
    session = _safe_get(server.sessions, session_name=session_name)
    if session is None:
        session = server.new_session(
            session_name=session_name,
            window_name=initial_window_name,
            detached=True,
        )
        _mark_session_role(session, role)
        return session

    existing_role = _get_session_role(session)
    if existing_role != role:
        raise RuntimeError(
            f"tmux session '{session_name}' already exists but is not owned by qexp "
            f"(expected role={role!r}, found role={existing_role!r})."
        )
    return session


def ensure_internal_session():
    return ensure_managed_session(
        TMUX_SESSION_INTERNAL,
        QQTOOLS_SESSION_ROLE_INTERNAL,
        initial_window_name="daemon",
    )


def ensure_experiments_session():
    return ensure_managed_session(
        TMUX_SESSION_EXPERIMENTS,
        QQTOOLS_SESSION_ROLE_EXPERIMENTS,
        initial_window_name="shell",
    )


def _get_window(window_id: str | None):
    if not window_id:
        return None
    server = _get_server()
    return _safe_get(server.windows, window_id=window_id)


def _get_primary_pane(window):
    panes = getattr(window, "panes", None) or []
    if panes:
        return panes[0]
    attached_pane = getattr(window, "attached_pane", None)
    if attached_pane is not None:
        return attached_pane
    raise RuntimeError("tmux window has no controllable pane.")


def send_command_to_window(window_id: str, command: str) -> None:
    window = _get_window(window_id)
    if window is None:
        raise RuntimeError(f"tmux window {window_id!r} does not exist.")
    pane = _get_primary_pane(window)
    pane.send_keys(command, enter=True)


def launch_background_daemon(root: Path | str) -> str:
    root_path = Path(root).expanduser().resolve()
    session = ensure_internal_session()
    window = _safe_get(session.windows, window_name="daemon")
    if window is None:
        window = session.new_window(window_name="daemon", attach=False)

    command = " ".join(
        [
            shlex.quote(sys.executable),
            "-m",
            "qqtools.plugins.qexp.manager",
            "--foreground",
            "--root",
            shlex.quote(str(root_path)),
        ]
    )
    send_command_to_window(str(window.window_id), command)
    return str(window.window_id)


def create_window_for_task(task_id: str, session_name: str = TMUX_SESSION_EXPERIMENTS) -> str:
    if session_name == TMUX_SESSION_EXPERIMENTS:
        session = ensure_experiments_session()
    else:
        session = ensure_managed_session(
            session_name,
            QQTOOLS_SESSION_ROLE_EXPERIMENTS,
            initial_window_name="shell",
        )

    window_name = task_id[:48]
    window = session.new_window(window_name=window_name, attach=False)
    return str(window.window_id)


def window_exists(window_id: str | None) -> bool:
    if not window_id:
        return False

    return _get_window(window_id) is not None


def kill_window(window_id: str | None) -> None:
    window = _get_window(window_id)
    if window is not None:
        window.kill_window()
