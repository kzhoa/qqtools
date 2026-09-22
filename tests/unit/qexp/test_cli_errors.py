from __future__ import annotations

import pytest

from qqtools.plugins.qexp.cli.errors import CliOperationalError, CliUsageError, UserFacingError, classify_cli_error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            CliUsageError("bad value"),
            UserFacingError(code="invalid_argument", message="bad value", exit_code=2),
        ),
        (
            CliOperationalError("runtime unavailable", code="runtime_unavailable", next_action="qexp agent status"),
            UserFacingError(
                code="runtime_unavailable",
                message="runtime unavailable",
                exit_code=1,
                next_action="qexp agent status",
            ),
        ),
    ],
)
def test_named_cli_errors_have_one_format_independent_classification(
    error: Exception, expected: UserFacingError
) -> None:
    assert classify_cli_error(error) == expected


@pytest.mark.parametrize("error", [ValueError("bad"), RuntimeError("bad"), OSError("bad")])
def test_bare_legacy_exceptions_are_not_downgraded(error: Exception) -> None:
    assert classify_cli_error(error) is None
