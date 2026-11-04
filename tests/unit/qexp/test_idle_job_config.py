"""Idle Job input validation and fail-closed GPU selection."""

import pytest

from qqtools.plugins.qexp.agent.idle_job_config import IdleJobConfig, command_argv, resolve_idle_job_gpus


@pytest.mark.parametrize(
    ("selectors", "inventory", "expected"),
    [([-1], [3, 1], (3,)), ([1, 3], [1, 3], (1, 3)), ([-1, 3, 1, 1], [1, 3], (3, 1))],
)
def test_select_only_allowed_discovered_gpus(selectors, inventory, expected):
    assert resolve_idle_job_gpus(selectors, inventory) == expected


@pytest.mark.parametrize(
    ("selectors", "inventory"),
    [([-1], None), ([-1], []), ([0], [1]), ([-2], [1]), ([True], [1]), ([], [1]), ([1.0], [1])],
)
def test_gpu_selection_never_falls_back_to_unrestricted(selectors, inventory):
    with pytest.raises(ValueError):
        resolve_idle_job_gpus(selectors, inventory)


@pytest.mark.parametrize(
    "command",
    [
        "CUDA_VISIBLE_DEVICES=0 python job.py",
        "env CUDA_VISIBLE_DEVICES=0,1 python job.py",
        "env -i CUDA_VISIBLE_DEVICES=0 python job.py",
    ],
)
def test_gpu_environment_override_has_actionable_error(command):
    with pytest.raises(ValueError, match="gpu_ids"):
        command_argv(command)


def test_command_is_argv_without_shell_expansion():
    assert command_argv("python '/path with spaces/job.py' '$HOME'") == ("python", "/path with spaces/job.py", "$HOME")
    assert command_argv("env FOO=bar python job.py") == ("env", "FOO=bar", "python", "job.py")


@pytest.mark.parametrize(
    "command", ["", "python job.py && echo done", "env -S 'python job.py'", "FOO=bar python job.py"]
)
def test_unsupported_shell_syntax_is_rejected(command):
    with pytest.raises(ValueError):
        command_argv(command)


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": True},
        {"enabled": 1},
        {"idle_delay_seconds": -1},
        {"idle_delay_seconds": float("nan")},
        {"stop_timeout_seconds": 0},
        {"stop_timeout_seconds": True},
        {"gpu_ids": ()},
        {"gpu_ids": (False,)},
        {"revision": -1},
        {"revision": True},
        {"working_directory": "relative"},
    ],
)
def test_invalid_config_is_not_accepted(changes):
    with pytest.raises((TypeError, ValueError)):
        IdleJobConfig(**changes)


def test_default_disabled_config_round_trips():
    config = IdleJobConfig()
    assert config.enabled is False
    assert config.gpu_ids == (-1,)
    assert IdleJobConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("extra", [{"version": 2}, {"typo": True}, {"gpu_ids": "-1"}])
def test_unknown_config_format_is_not_silently_defaulted(extra):
    with pytest.raises(ValueError):
        IdleJobConfig.from_dict({**IdleJobConfig().to_dict(), **extra})
