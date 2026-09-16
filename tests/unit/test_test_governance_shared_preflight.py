from __future__ import annotations

from pathlib import Path

from scripts.checks.check_test_lanes import check_test_lanes


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_repository(root: Path, runner: str) -> None:
    _write(root / "tests/integration/qexp/test_agent_lifecycle_independence.py", "def test_li01():\n    pass\n")
    _write(root / "tests/integration/test_example.py", "def test_integration():\n    pass\n")
    _write(root / "tests/e2e/test_example.py", "def test_e2e():\n    pass\n")
    _write(root / "scripts/ci/run_preflight.py", runner)
    _write(
        root / "tox.ini",
        """[testenv:preflight]
commands = python scripts/ci/run_preflight.py

[testenv:artifact-e2e]
commands = pytest tests/e2e

[testenv:qexp-lifecycle-parallel]
commands = pytest --lifecycle-gate=full tests/integration/qexp/test_agent_lifecycle_independence.py -n 4 --dist load

[testenv:qexp-integration]
commands = python scripts/qexp_integration_gate.py --budget-seconds 600
""",
    )
    _write(
        root / ".github/workflows/ci.yml",
        """on:
  push:
jobs:
  artifact-e2e:
    steps:
      - run: tox run -e artifact-e2e
""",
    )


def test_lane_governance_accepts_shared_preflight_manifest(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        """COMMANDS = (
    ("python", "-m", "pytest", "tests/unit", "-q"),
    ("python", "-m", "pytest", "tests/integration", "-q"),
    ("python", "-m", "pytest", "--lifecycle-gate=representative", "tests/integration/qexp/test_agent_lifecycle_independence.py", "-q"),
)
""",
    )

    assert check_test_lanes(tmp_path) == []


def test_lane_governance_rejects_shared_manifest_without_lifecycle_gate(tmp_path: Path) -> None:
    _write_repository(
        tmp_path,
        """COMMANDS = (
    ("python", "-m", "pytest", "tests/unit", "-q"),
    ("python", "-m", "pytest", "tests/integration", "-q"),
)
""",
    )

    assert check_test_lanes(tmp_path) == ["preflight must run the representative qexp lifecycle gate"]
