import pytest

from qqtools.plugins.qexp import tmux


class _FakePane:
    def __init__(self):
        self.commands: list[tuple[str, bool]] = []

    def send_keys(self, command: str, enter: bool = True):
        self.commands.append((command, enter))


class _FakeWindow:
    def __init__(self, window_id: str, window_name: str):
        self.window_id = window_id
        self.window_name = window_name
        self.panes = [_FakePane()]
        self.killed = False

    def kill_window(self):
        self.killed = True


class _FakeCollection:
    def __init__(self, items, lookup_key: str):
        self._items = items
        self._lookup_key = lookup_key

    def get(self, **kwargs):
        expected = kwargs.get(self._lookup_key)
        for item in self._items:
            if getattr(item, self._lookup_key) == expected:
                return item
        return None


class _FakeSession:
    def __init__(self, session_name: str, role: str | None = None):
        self.session_name = session_name
        self._role = role
        self.windows_list = [_FakeWindow("@base", "shell")]
        self.windows = _FakeCollection(self.windows_list, "window_name")
        self.last_window_start_directory = None

    def show_option(self, _name: str):
        return self._role

    def set_option(self, _name: str, value: str):
        self._role = value

    def new_window(
        self,
        window_name: str,
        attach: bool = False,
        start_directory: str | None = None,
    ):
        self.last_window_start_directory = start_directory
        window = _FakeWindow(f"@{len(self.windows_list) + 1}", window_name)
        self.windows_list.append(window)
        return window


class _FakeServer:
    def __init__(self, sessions=None, windows=None):
        self.sessions_list = sessions or []
        self.windows_list = windows or []
        self.sessions = _FakeCollection(self.sessions_list, "session_name")
        self.windows = _FakeCollection(self.windows_list, "window_id")
        self.last_session_start_directory = None

    def new_session(
        self,
        session_name: str,
        window_name: str,
        detached: bool = True,
        start_directory: str | None = None,
    ):
        self.last_session_start_directory = start_directory
        session = _FakeSession(session_name)
        session.windows_list = [_FakeWindow("@new", window_name)]
        session.windows = _FakeCollection(session.windows_list, "window_name")
        self.sessions_list.append(session)
        return session


def test_ensure_managed_session_rejects_unowned_collision(monkeypatch):
    server = _FakeServer(sessions=[_FakeSession("experiments", role=None)])
    monkeypatch.setattr(tmux, "_get_server", lambda: server)

    with pytest.raises(RuntimeError, match="not owned by qexp"):
        tmux.ensure_experiments_session()


def test_send_command_to_window_targets_primary_pane(monkeypatch):
    window = _FakeWindow("@1", "task_demo")
    server = _FakeServer(windows=[window])
    monkeypatch.setattr(tmux, "_get_server", lambda: server)

    tmux.send_command_to_window("@1", "python train.py")

    assert window.panes[0].commands == [("python train.py", True)]


def test_require_libtmux_reports_install_hint(monkeypatch):
    def _raise_import_error(_name: str):
        raise ImportError("missing libtmux")

    monkeypatch.setattr(tmux.importlib, "import_module", _raise_import_error)

    with pytest.raises(RuntimeError, match="pip install qqtools\\[exp\\]"):
        tmux.require_libtmux()


def test_create_window_for_task_uses_custom_session(monkeypatch):
    session = _FakeSession("custom", role=tmux.QQTOOLS_SESSION_ROLE_EXPERIMENTS)
    server = _FakeServer(sessions=[session])
    monkeypatch.setattr(tmux, "_get_server", lambda: server)

    window_id = tmux.create_window_for_task(
        "task-demo",
        session_name="custom",
        start_directory="/tmp/project",
    )

    assert window_id == "@2"
    assert session.windows_list[-1].window_name == "task-demo"
    assert session.last_window_start_directory == "/tmp/project"
