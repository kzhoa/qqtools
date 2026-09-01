from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _require_installed_qqtools() -> None:
    """Reject installed-artifact E2E sessions that import the checkout source tree."""
    import qqtools

    imported_from = Path(qqtools.__file__).resolve()
    if "site-packages" not in str(imported_from):
        pytest.fail(f"qqtools was not imported from site-packages: {imported_from}")
