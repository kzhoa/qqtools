from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest


def _case_directory_generator(nodeid: str, name: str) -> Generator[Path, None, None]:
    request = SimpleNamespace(node=SimpleNamespace(nodeid=nodeid, name=name))
    return conftest.tmp_path.__wrapped__(request)


def test_tmp_path_keeps_evidence_when_artifact_retention_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "TMP_ROOT", tmp_path / "artifacts")
    monkeypatch.setenv(conftest.PRESERVE_TEST_ARTIFACTS_ENV, "1")
    case_directory = _case_directory_generator(
        "tests/e2e/test_flow.py::test_failure", "test_failure"
    )

    retained_path = next(case_directory)
    (retained_path / "runtime.log").write_text("diagnostic evidence", encoding="utf-8")

    with pytest.raises(StopIteration):
        next(case_directory)

    assert (retained_path / "runtime.log").read_text(encoding="utf-8") == "diagnostic evidence"


def test_tmp_path_cleans_evidence_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "TMP_ROOT", tmp_path / "artifacts")
    monkeypatch.delenv(conftest.PRESERVE_TEST_ARTIFACTS_ENV, raising=False)
    case_directory = _case_directory_generator(
        "tests/e2e/test_flow.py::test_failure", "test_failure"
    )

    cleaned_path = next(case_directory)
    (cleaned_path / "runtime.log").write_text("diagnostic evidence", encoding="utf-8")

    with pytest.raises(StopIteration):
        next(case_directory)

    assert not cleaned_path.exists()
