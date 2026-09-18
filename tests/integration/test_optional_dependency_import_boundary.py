from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def test_data_imports_do_not_eagerly_load_optional_plotting_stack() -> None:
    source_root = ROOT / "src"
    code = """
import sys
import qqtools.data
from qqtools.data.qdatalist import qDataList, qList
from qqtools.data.qscaladict import qScalaDict

assert qDataList is not None
assert qList is not None
assert qScalaDict is not None
assert 'matplotlib' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert 'pandas' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
