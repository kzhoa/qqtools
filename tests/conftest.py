import hashlib
import os
import re
import shutil
import socket
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
INSTALLED_E2E_ROOT = PROJECT_ROOT / "tests" / "e2e"
INSTALLED_E2E_PYTEST_INI = PROJECT_ROOT / "tests" / "e2e" / "installed_artifact_pytest.ini"
PRESERVE_TEST_ARTIFACTS_ENV = "QQTOOLS_PRESERVE_TEST_ARTIFACTS"
TEST_TMUX_BASE_ENV = "QQTOOLS_TEST_TMUX_BASE"


def pytest_addoption(parser):
    parser.addoption("--lifecycle-gate", choices=("representative", "full", "installed"))


def pytest_configure(config):
    if config.getoption("--lifecycle-gate"):
        config.pluginmanager.register(_LifecycleGate(config), "qexp-lifecycle-gate")


class _LifecycleGate:
    """Require selected lifecycle cases to execute successfully, including teardown."""

    representative_names = frozenset(
        {
            "test_li01_training_remains_live_and_is_not_relaunched",
            "test_li02_offline_completion_preserves_exit_result[0-succeeded]",
            "test_li02_offline_completion_preserves_exit_result[7-failed]",
            "test_li04_sigkill_agent_does_not_kill_runner",
        }
    )
    full_names = representative_names | frozenset(
        {
            "test_li03_expired_claim_recovers_same_attempt_without_relaunch",
            "test_li03_real_peer_observes_natural_lease_expiry[False]",
            "test_li03_real_peer_observes_natural_lease_expiry[True]",
            "test_li05_launch_boundary_has_no_duplicate_authorized_process[before_authorization]",
            "test_li05_launch_boundary_has_no_duplicate_authorized_process[after_authorization]",
            "test_li05_agent_crash_between_process_creation_and_registration",
            "test_li06_terminal_publication_is_idempotent[False-attempt]",
            "test_li06_terminal_publication_is_idempotent[False-task]",
            "test_li06_terminal_publication_is_idempotent[False-reservation]",
            "test_li06_terminal_publication_is_idempotent[True-attempt]",
            "test_li06_terminal_publication_is_idempotent[True-task]",
            "test_li06_terminal_publication_is_idempotent[True-reservation]",
            "test_li07_multiple_bindings_keep_identity_and_reservations_separate[2-False]",
            "test_li07_multiple_bindings_keep_identity_and_reservations_separate[4-False]",
            "test_li07_multiple_bindings_keep_identity_and_reservations_separate[4-True]",
            "test_li08_mismatched_exit_evidence_is_retained_as_blocker",
            "test_li08_missing_exit_observation_is_diagnosed",
            "test_li08_superseded_offline_attempt_preserves_evidence",
            "test_li08_cancellation_while_agent_offline_is_honored",
            "test_global_idle_policy_considers_every_binding[modes0]",
            "test_global_idle_policy_considers_every_binding[modes1]",
            "test_global_idle_policy_considers_every_binding[modes2]",
            "test_finished_process_releases_capacity_while_publication_is_unavailable[_finalize]",
            "test_finished_process_releases_capacity_while_publication_is_unavailable[_materialize_registrations]",
            "test_finished_process_releases_capacity_while_publication_is_unavailable[binding]",
            "test_global_idle_waits_for_unresolved_demand",
            "test_failed_binding_is_not_consumed_for_idle_exit",
            "test_global_idle_does_not_reenter_registration_wait",
            "test_pending_repair_prevents_idle_exit[availability]",
            "test_pending_repair_prevents_idle_exit[group_control]",
            "test_pending_repair_prevents_idle_exit[cleanup]",
        }
    )

    def __init__(self, config):
        self.mode = config.getoption("--lifecycle-gate")
        self.started_at = time.monotonic()
        self.budget_seconds = 90 if self.mode == "representative" else 600
        self.required = set()
        self.passed = set()
        self.failed = False

    def pytest_collection_finish(self, session):
        items = [item for item in session.items if item.path.name == "test_agent_lifecycle_independence.py"]
        required_names = self.representative_names if self.mode == "representative" else self.full_names
        missing = sorted(required_names - {item.name for item in items})
        if missing:
            raise pytest.UsageError("Missing lifecycle gate cases: " + ", ".join(missing))
        self.required = {item.nodeid for item in items}

    def pytest_runtest_logreport(self, report):
        if report.nodeid not in self.required:
            return
        if report.skipped or report.failed:
            self.failed = True
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        is_over_budget = time.monotonic() - self.started_at > self.budget_seconds
        if is_over_budget:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(f"Lifecycle gate exceeded {self.budget_seconds}s budget", red=True)
        if self.failed or not self.required or self.passed != self.required or is_over_budget:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _is_usable_temp_root(root: Path) -> bool:
    """Return whether a root supports the filesystem operations required by tests."""
    probe_dir = root / f".qqtools-write-probe-{uuid.uuid4().hex}"
    probe_file = probe_dir / "probe"
    probe_socket = probe_dir / "probe.sock"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe_dir.mkdir()
        probe_file.write_bytes(b"ok")
        probe_file.unlink()
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(probe_socket))
        probe_socket.unlink()
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
        f"No usable test temporary root: both {system_root} and {fallback_root} failed a create/write/delete probe."
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
        "qqtools.plugins.qexp.runtime.attempt_recovery.clock_capability",
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
    violations = TestResourceScope.cleanup_violations(scope.root)
    if violations:
        pytest.fail("qexp test resource cleanup failed:\n" + "\n".join(violations))


def pytest_ignore_collect(collection_path, config):
    """Reserve installed-wheel e2e tests for their isolated pytest config."""
    path = Path(str(collection_path)).resolve()
    if path != INSTALLED_E2E_ROOT and INSTALLED_E2E_ROOT not in path.parents:
        return None
    return config.inipath is None or Path(config.inipath).resolve() != INSTALLED_E2E_PYTEST_INI


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
def _configure_temp_root_for_session(request):
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    previous_tempdir = tempfile.tempdir
    previous_env = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP", TEST_TMUX_BASE_ENV)}
    original_mkdtemp = tempfile.mkdtemp
    original_temporary_directory = tempfile.TemporaryDirectory
    tmp_root_str = str(TMP_ROOT)

    # Make all tempfile-based APIs resolve under project-local ./tmp.
    tempfile.tempdir = tmp_root_str
    os.environ["TMPDIR"] = tmp_root_str
    os.environ["TMP"] = tmp_root_str
    os.environ["TEMP"] = tmp_root_str
    os.environ[TEST_TMUX_BASE_ENV] = str(TEST_TMP_BASE)
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
        if not _should_preserve_test_artifacts() and not getattr(request.config, "_has_lifecycle_artifacts", False):
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
    should_preserve_lifecycle = request.node.path.name == "test_agent_lifecycle_independence.py"
    if should_preserve_lifecycle:
        request.config._has_lifecycle_artifacts = True

    try:
        yield case_dir
    finally:
        if not _should_preserve_test_artifacts() and not should_preserve_lifecycle:
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
