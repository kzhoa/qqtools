import pytest

from qqtools.plugins.qexp.cli import build_parser


def test_submit_tmux_flags_are_nullable_and_application_arguments_stay_after_separator():
    parser = build_parser()

    inherited = parser.parse_args(["submit", "--", "python", "entry.py", "--tmux"])
    enabled = parser.parse_args(["submit", "--tmux", "--", "python", "entry.py"])
    disabled = parser.parse_args(["submit", "--no-tmux", "--", "python", "entry.py"])

    assert inherited.tmux_override is None
    assert inherited.argv == ["--", "python", "entry.py", "--tmux"]
    assert enabled.tmux_override is True
    assert enabled.argv == ["--", "python", "entry.py"]
    assert disabled.tmux_override is False


def test_batch_tmux_flags_are_nullable_and_mutually_exclusive():
    parser = build_parser()
    inherited = parser.parse_args(["submit", "--file", "runs.yaml"])
    enabled = parser.parse_args(["submit", "--file", "runs.yaml", "--tmux"])

    assert inherited.tmux_override is None
    assert enabled.tmux_override is True
    with pytest.raises(SystemExit):
        parser.parse_args(["submit", "--file", "runs.yaml", "--tmux", "--no-tmux"])


def test_config_tmux_set_rejects_conflicting_values():
    parser = build_parser()

    assert parser.parse_args(["config", "set", "tmux", "--enabled"]).enabled is True
    assert parser.parse_args(["config", "set", "tmux", "--disabled"]).disabled is True
    with pytest.raises(SystemExit):
        parser.parse_args(["config", "set", "tmux", "--enabled", "--disabled"])
