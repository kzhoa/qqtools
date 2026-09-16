from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_plotting_and_pandas_are_optional_but_retained_by_full_install() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    base = set(project["dependencies"])
    extras = project["optional-dependencies"]

    assert "matplotlib" not in base
    assert "pandas" not in base
    assert set(extras["data"]) == {"matplotlib", "pandas"}
    assert "qqtools[data]" in extras["full"]


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
        env={"PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
