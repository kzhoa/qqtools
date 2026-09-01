from __future__ import annotations

from pathlib import Path

from scripts.checks.check_contract_matrix import check_contract_matrix
from scripts.checks.check_test_lanes import check_test_lanes


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_valid_repository(root: Path) -> None:
    _write(root / "tests/integration/test_example.py", "def test_integration():\n    pass\n")
    _write(root / "tests/e2e/test_example.py", "def test_e2e():\n    pass\n")
    _write(
        root / "tox.ini",
        """[testenv:preflight]
commands =
    pytest tests/unit
    pytest tests/integration

[testenv:artifact-e2e]
commands = pytest tests/e2e
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
    _write(root / "tests/example.py", "evidence\n")
    _write(
        root / "tests/CONTRACT_MATRIX.md",
        "[evidence](example.py)\n",
    )


def test_test_governance_accepts_current_lane_contract(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)

    assert check_test_lanes(tmp_path) == []
    assert check_contract_matrix(tmp_path) == []


def test_lane_marker_check_ignores_comments_and_string_literals(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / "tests/integration/test_example.py",
        """# host_exclusive documents a forbidden marker here.
DESCRIPTION = "host_exclusive"

def test_integration():
    pass
""",
    )

    assert check_test_lanes(tmp_path) == []


def test_lane_marker_check_rejects_marker_in_wrong_layer(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / "tests/integration/test_example.py",
        """import pytest

pytestmark = pytest.mark.host_exclusive

def test_integration():
    pass
""",
    )

    assert check_test_lanes(tmp_path) == [
        "Integration may not use host_exclusive: tests/integration/test_example.py"
    ]


def test_lane_check_rejects_e2e_collection_from_preflight(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / "tox.ini",
        """[testenv:preflight]
commands = pytest tests/e2e

[testenv:artifact-e2e]
commands = pytest tests/e2e
""",
    )

    assert check_test_lanes(tmp_path) == ["preflight must collect only Unit and Integration tests"]


def test_lane_check_rejects_source_tests_in_ordinary_ci(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / ".github/workflows/ci.yml",
        """on:
  push:
jobs:
  source-tests:
    steps:
      - run: tox run -e unit
      - run: tox run -e artifact-e2e
""",
    )

    assert check_test_lanes(tmp_path) == [
        "ordinary CI may not run source-test lane: tox run -e unit"
    ]


def test_lane_check_rejects_equivalent_tox_source_test_syntax(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / ".github/workflows/ci.yml",
        """on:
  push:
jobs:
  source-tests:
    steps:
      - run: tox -eunit
      - run: tox run -e artifact-e2e
""",
    )

    assert check_test_lanes(tmp_path) == [
        "ordinary CI may not run source-test lane: tox -eunit"
    ]


def test_contract_matrix_rejects_missing_test_link(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(
        tmp_path / "tests/CONTRACT_MATRIX.md",
        "[missing evidence](unit/test_missing.py)\n",
    )

    assert check_contract_matrix(tmp_path) == [
        "contract matrix links to a missing test: unit/test_missing.py"
    ]


def test_contract_matrix_rejects_retired_lane_term(tmp_path: Path) -> None:
    _write_valid_repository(tmp_path)
    _write(tmp_path / "tests/CONTRACT_MATRIX.md", "qexp-fast\n")

    assert check_contract_matrix(tmp_path) == [
        "contract matrix contains retired lane term: qexp-fast"
    ]
