from __future__ import annotations

import subprocess
import sys


def test_installed_wheel_exposes_cli_only_from_owner_modules() -> None:
    script = """
from pathlib import Path
import importlib.metadata
import qqtools.plugins.qexp.cli as cli_package
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.cli import local_handlers, project_handlers, submission
from qqtools.plugins.qexp.cli.parser import build_parser

entry_points = [
    item for item in importlib.metadata.entry_points(group='console_scripts') if item.name == 'qexp'
]
assert len(entry_points) == 1
assert entry_points[0].value == 'qqtools.plugins.qexp.cli.entrypoint:main'
assert 'site-packages' in str(Path(cli_package.__file__).resolve())
assert not hasattr(cli_package, 'main')
assert not hasattr(cli_package, 'build_parser')
assert callable(main)
assert build_parser().prog == 'qexp'
assert local_handlers.LOCAL_HANDLERS
assert callable(project_handlers.dispatch_project)
assert callable(submission.dispatch_submission)
"""
    subprocess.run([sys.executable, "-E", "-c", script], check=True)


def test_installed_wheel_supports_direct_cli_package_execution() -> None:
    result = subprocess.run(
        [sys.executable, "-E", "-m", "qqtools.plugins.qexp.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("usage: qexp")
    assert result.stderr == ""


def test_installed_wheel_exposes_runtime_packages_from_owner_modules() -> None:
    script = """
from pathlib import Path
import qqtools
import qqtools.plugins.qexp as qexp
from qqtools.plugins.qexp.runtime import availability, ready
from qqtools.plugins.qexp.runtime.availability import transitions
from qqtools.plugins.qexp.runtime.ready import records
from qqtools.plugins.qexp.runtime.resources import cpu_lane

assert 'site-packages' in str(Path(qqtools.__file__).resolve())
assert ready.ReadyMarkerRef is records.ReadyMarkerRef
assert availability.apply_availability_transition is transitions.apply_availability_transition
assert qexp.CpuLanePolicy is cpu_lane.CpuLanePolicy
assert qexp.get_cpu_lane_policy is cpu_lane.get_cpu_lane_policy
assert qexp.set_cpu_lane_capacity is cpu_lane.set_cpu_lane_capacity
"""
    subprocess.run([sys.executable, "-E", "-c", script], check=True)
