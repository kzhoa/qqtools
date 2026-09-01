from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest


def _case_directory_generator(nodeid: str, name: str) -> Generator[Path, None, None]:
    request = SimpleNamespace(node=SimpleNamespace(nodeid=nodeid, name=name))
    return conftest.tmp_path.__wrapped__(request)


def test_test_tmp_base_prefers_usable_system_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_root = tmp_path / "system-tmp"
    fallback_root = tmp_path / "repository-tmp"
    monkeypatch.setattr(conftest, "_is_usable_temp_root", lambda root: root == system_root)

    selected = conftest._select_test_tmp_base(system_root, fallback_root)

    assert selected == system_root
    assert not fallback_root.exists()


def test_test_tmp_base_falls_back_when_system_root_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_root = tmp_path / "system-tmp"
    fallback_root = tmp_path / "repository-tmp"
    monkeypatch.setattr(conftest, "_is_usable_temp_root", lambda root: root == fallback_root)

    selected = conftest._select_test_tmp_base(system_root, fallback_root)

    assert selected == fallback_root


def test_test_tmp_base_falls_back_when_system_root_cannot_bind_unix_sockets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_root = tmp_path / "system-tmp"
    fallback_root = tmp_path / "repository-tmp"
    original_socket = conftest.socket.socket

    class SocketWithoutSystemRootAccess:
        def __init__(self, *args, **kwargs) -> None:
            self._socket = original_socket(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            self._socket.close()

        def bind(self, address: str) -> None:
            if str(system_root) in address:
                raise PermissionError("system root denies Unix sockets")
            Path(address).touch()

    monkeypatch.setattr(conftest.socket, "socket", SocketWithoutSystemRootAccess)

    selected = conftest._select_test_tmp_base(system_root, fallback_root)

    assert selected == fallback_root


def test_test_tmp_base_fails_clearly_when_neither_root_is_usable(tmp_path: Path) -> None:
    system_root = tmp_path / "system-tmp"
    fallback_root = tmp_path / "repository-tmp"
    system_root.write_text("not a directory", encoding="utf-8")
    fallback_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No usable test temporary root"):
        conftest._select_test_tmp_base(system_root, fallback_root)


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
