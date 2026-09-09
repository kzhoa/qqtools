import time
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--installed-lifecycle-gate", action="store_true")


def pytest_configure(config):
    if config.getoption("--installed-lifecycle-gate"):
        config.pluginmanager.register(_InstalledLifecycleGate(), "installed-lifecycle-gate")


class _InstalledLifecycleGate:
    def __init__(self):
        self.required = set()
        self.passed = set()
        self.failed = False
        self.duration_seconds = 0.0

    def pytest_collection_finish(self, session):
        self.required = {
            item.nodeid for item in session.items if item.name == "test_installed_cli_stop_offline_completion_start"
        }
        if not self.required:
            raise pytest.UsageError("Missing installed lifecycle workflow")

    def pytest_runtest_logreport(self, report):
        if report.nodeid in self.required:
            self.duration_seconds += report.duration
            self.failed |= report.failed or report.skipped
            if report.when == "call" and report.passed:
                self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.failed or not self.required or self.required != self.passed or self.duration_seconds > 90:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True, scope="session")
def _require_installed_qqtools() -> None:
    """Reject installed-artifact E2E sessions that import the checkout source tree."""
    import qqtools

    imported_from = Path(qqtools.__file__).resolve()
    if "site-packages" not in str(imported_from):
        pytest.fail(f"qqtools was not imported from site-packages: {imported_from}")
