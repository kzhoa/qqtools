from types import SimpleNamespace

from qqtools.plugins.qexp.cli.parser import build_parser
from qqtools.plugins.qexp.commands import attach


def test_task_attach_defaults_to_read_only_existing_window(monkeypatch):
    args = build_parser().parse_args(["task", "attach", "task-1"])
    assert args.command_spec.handler == "task_attach"
    seen = {}

    def fake_observer(self, cfg, task_id):
        seen["task_id"] = task_id
        return SimpleNamespace(window_id="@37")

    def fake_call(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(attach.Executor, "attach_task_observer", fake_observer)
    monkeypatch.setattr(attach.subprocess, "call", fake_call)
    assert attach.attach_task(object(), "task-1") == 0
    assert seen == {"task_id": "task-1", "argv": ["tmux", "attach-session", "-r", "-t", "@37"]}
