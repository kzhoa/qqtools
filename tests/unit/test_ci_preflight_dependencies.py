from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CI_PREFLIGHT = {
    "libtmux>=0.62.0",
    "lmdb",
    "prompt_toolkit",
    "pytest",
    "pytest-xdist",
    "ruff",
    "scikit-learn",
    "tox==4.61.5",
    "tox-uv==1.36.0",
    "uv==0.12.16",
    "tqdm",
}


def test_ci_preflight_extra_is_explicit_and_minimal() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(config["project"]["optional-dependencies"]["ci-preflight"])

    assert dependencies == EXPECTED_CI_PREFLIGHT
    assert not {"build", "nvidia-ml-py", "psutil", "requests"}.intersection(dependencies)


def test_dev_preflight_uses_cpu_only_focused_profile() -> None:
    workflow = (ROOT / ".github/workflows/dev-preflight.yml").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert 'pip install -e ".[ci-preflight]"' in workflow
    assert 'pip install -e ".[full]"' not in workflow
    assert "pytest-xdist" not in workflow
    assert "import libtmux, lmdb, prompt_toolkit, sklearn" in workflow
