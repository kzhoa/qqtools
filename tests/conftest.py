import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SYSTEM_TEST_TMP_ROOT = Path("/tmp")
FALLBACK_TEST_TMP_ROOT = PROJECT_ROOT / "tmp"
RELEASE_E2E_ROOT = PROJECT_ROOT / "tests" / "e2e" / "qexp"
RELEASE_E2E_PYTEST_INI = PROJECT_ROOT / "tests" / "e2e" / "release_pytest.ini"
PRESERVE_TEST_ARTIFACTS_ENV = "QQTOOLS_PRESERVE_TEST_ARTIFACTS"


def _is_usable_temp_root(root: Path) -> bool:
    """Return whether a root supports the filesystem operations required by tests."""
    probe_dir = root / f".qqtools-write-probe-{uuid.uuid4().hex}"
    probe_file = probe_dir / "probe"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe_dir.mkdir()
        probe_file.write_bytes(b"ok")
        probe_file.unlink()
        probe_dir.rmdir()
    except OSError:
        shutil.rmtree(probe_dir, ignore_errors=True)
        return False
    return True


def _select_test_tmp_base(
    system_root: Path = SYSTEM_TEST_TMP_ROOT,
    fallback_root: Path = FALLBACK_TEST_TMP_ROOT,
) -> Path:
    """Prefer the system temporary filesystem and fall back to the repository."""
    if _is_usable_temp_root(system_root):
        return system_root
    if _is_usable_temp_root(fallback_root):
        return fallback_root
    raise RuntimeError(
        "No usable test temporary root: both "
        f"{system_root} and {fallback_root} failed a create/write/delete probe."
    )


TEST_TMP_BASE = _select_test_tmp_base()
TMP_ROOT = TEST_TMP_BASE / f"qqtools-pytest-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qexp_healthy_clock(monkeypatch):
    """Provide the bounded-clock deployment prerequisite to qexp tests."""
    from qqtools.plugins.qexp.lease import ClockCapability, ClockObservation

    observation = ClockObservation(
        "test-observation",
        "chrony",
        "2026-08-06T00:00:00Z",
        time.monotonic(),
        "test-boot",
        -0.001,
        0.001,
        0.0,
        0.0,
    )
    capability = ClockCapability("healthy", "healthy", observation, ("chrony",))
    monkeypatch.setattr("qqtools.plugins.qexp.lease.clock_capability", lambda *_args: capability)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.clock_capability", lambda *_args: capability)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.recovery.clock_capability",
        lambda *_args: capability,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.reclaim_allowed_at",
        lambda *_args: datetime.now(timezone.utc) - timedelta(seconds=1),
    )


@pytest.fixture
def qexp_resource_scope(tmp_path: Path, request: pytest.FixtureRequest):
    """Provide isolated filesystem and local-process resources to qexp tests."""
    from tests.helpers.qexp.resources import TestResourceScope

    scope = TestResourceScope.create(tmp_path / "qexp-resources", request.node.nodeid)
    yield scope
    violations = TestResourceScope.cleanup_violations(tmp_path)
    if violations:
        pytest.fail("qexp test resource cleanup failed:\n" + "\n".join(violations))


def pytest_ignore_collect(collection_path, config):
    """Reserve installed-wheel e2e tests for their isolated pytest config."""
    path = Path(str(collection_path)).resolve()
    if path != RELEASE_E2E_ROOT and RELEASE_E2E_ROOT not in path.parents:
        return None
    return config.inipath is None or Path(config.inipath).resolve() != RELEASE_E2E_PYTEST_INI


def _build_case_tmp_dir_name(nodeid: str, name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe_name:
        safe_name = "test_case"
    safe_name = safe_name[:48]
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name}-{digest}"


def _should_preserve_test_artifacts() -> bool:
    """Return whether test case directories should remain available after a run."""
    return os.environ.get(PRESERVE_TEST_ARTIFACTS_ENV) == "1"


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
        if not _should_preserve_test_artifacts():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)


@pytest.fixture
def tmp_path(request):
    """
    Provide a per-test child directory under the selected session root.
    """
    case_dir_name = _build_case_tmp_dir_name(request.node.nodeid, request.node.name)
    case_dir = TMP_ROOT / case_dir_name
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield case_dir
    finally:
        if not _should_preserve_test_artifacts():
            shutil.rmtree(case_dir, ignore_errors=True)


@pytest.fixture
def checkout_subprocess_env() -> dict[str, str]:
    """Build an environment that imports qqtools from the current checkout."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(SRC_ROOT)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env
