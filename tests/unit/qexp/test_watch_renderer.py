import builtins
from io import StringIO

from qqtools.plugins.qexp.commands import watch_renderer


def test_append_viewer_sanitizes_input_and_suppresses_unchanged_blocks(monkeypatch):
    monkeypatch.setattr(watch_renderer, "_terminal_size", lambda: (40, 8))
    output = StringIO()
    renderer = watch_renderer.WatchRenderer(output)
    renderer.render({"revision": 1}, "Stage: train\x1b[31m\nMessage: hello")
    first = output.getvalue()
    assert "Stage: train" in first
    assert "\x1b" not in first
    renderer.render({"revision": 1}, "Stage: train\x1b[31m\nMessage: hello")
    assert output.getvalue() == first
    renderer.render({"revision": 2}, "Stage: train\x1b[31m\nMessage: hello")
    assert output.getvalue() == first
    renderer.render({"revision": 2, "phase": "succeeded"}, "Stage: validation")
    assert output.getvalue().count("Stage:") == 2


def test_ansi_viewer_clears_owned_rows_after_resize(monkeypatch):
    sizes = iter(((20, 5), (12, 5)))
    monkeypatch.setattr(watch_renderer, "_terminal_size", lambda: next(sizes))
    output = StringIO()
    renderer = watch_renderer.WatchRenderer(output)
    renderer._backend = "ansi"
    renderer._cursor_up = "\x1b[A"
    renderer._clear_line = "\x1b[K"
    renderer.render({"revision": 1}, "First row\nSecond row\nThird row")
    first = output.getvalue()
    renderer.render({"revision": 2}, "New row")
    second = output.getvalue()[len(first) :]
    assert second.count("\x1b[A") == 2
    assert second.count("\x1b[K") == 3
    assert second.endswith("New row")


def test_rich_import_failure_uses_ansi_without_touching_payload(monkeypatch):
    monkeypatch.setattr(watch_renderer, "_cursor_capabilities", lambda _stdout: ("\x1b[A", "\x1b[K"))
    original_import = builtins.__import__

    def missing_rich(name, *args, **kwargs):
        if name.startswith("rich."):
            raise ImportError("Rich unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_rich)
    output, errors = StringIO(), StringIO()
    renderer = watch_renderer.WatchRenderer(output, errors)
    assert renderer._backend == "ansi"
    assert errors.getvalue().count("Rich renderer unavailable") == 1


def test_rich_runtime_failure_falls_back_to_append_and_reports_once(monkeypatch):
    monkeypatch.setattr(watch_renderer, "_terminal_size", lambda: (40, 8))
    output, errors = StringIO(), StringIO()
    renderer = watch_renderer.WatchRenderer(output, errors)
    renderer._backend = "rich"

    def fail(_lines):
        raise RuntimeError("render failed")

    monkeypatch.setattr(renderer, "_render_rich", fail)
    payload = {"phase": "running", "progress": {"reported_at": "before"}}
    renderer.render(payload, "Stage: train")
    assert renderer._backend == "append"
    assert "Stage: train" in output.getvalue()
    assert errors.getvalue().count("terminal renderer failed") == 1
    first = output.getvalue()
    renderer.render(payload, "Stage: train")
    assert output.getvalue() == first
