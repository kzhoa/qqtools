import hashlib
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp"


def _build_case_tmp_dir_name(nodeid: str, name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe_name:
        safe_name = "test_case"
    safe_name = safe_name[:48]
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name}-{digest}"


def _workspace_mkdtemp(suffix=None, prefix=None, dir=None):
    suffix = "" if suffix is None else suffix
    prefix = "tmp" if prefix is None else prefix
    base_dir = Path(dir) if dir is not None else TMP_ROOT
    base_dir.mkdir(parents=True, exist_ok=True)

    while True:
        candidate = base_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            continue


class _WorkspaceTemporaryDirectory:
    def __init__(
        self,
        suffix=None,
        prefix=None,
        dir=None,
        ignore_cleanup_errors=False,
        *,
        delete=True,
    ):
        self.name = _workspace_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete
        self._closed = False

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    def cleanup(self):
        if self._closed:
            return

        self._closed = True
        if not self._delete:
            return

        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


@pytest.fixture(autouse=True, scope="session")
def _configure_temp_root_for_session():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    previous_tempdir = tempfile.tempdir
    previous_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
    original_mkdtemp = tempfile.mkdtemp
    original_temporary_directory = tempfile.TemporaryDirectory
    tmp_root_str = str(TMP_ROOT)

    # Make all tempfile-based APIs resolve under project-local ./tmp.
    tempfile.tempdir = tmp_root_str
    os.environ["TMPDIR"] = tmp_root_str
    os.environ["TMP"] = tmp_root_str
    os.environ["TEMP"] = tmp_root_str
    tempfile.mkdtemp = _workspace_mkdtemp
    tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory

    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        tempfile.mkdtemp = original_mkdtemp
        tempfile.TemporaryDirectory = original_temporary_directory

        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def tmp_path(request):
    """
    Provide a per-test child directory under ./tmp and clean only that child directory.
    """
    case_dir_name = _build_case_tmp_dir_name(request.node.nodeid, request.node.name)
    case_dir = TMP_ROOT / case_dir_name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
