from __future__ import annotations

import subprocess
import sys


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
