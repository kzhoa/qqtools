from __future__ import annotations

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
