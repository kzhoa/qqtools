"""Explicit user-facing error types for qexp CLI adapter boundaries."""

from __future__ import annotations

from dataclasses import dataclass


class CliUsageError(Exception):
    """A user-supplied invocation value is invalid."""


class CliOperationalError(Exception):
    """A recognized environmental failure prevented an operation."""

    def __init__(self, message: str, *, code: str = "operational_failure", next_action: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.next_action = next_action


@dataclass(frozen=True, slots=True)
class UserFacingError:
    """One format-independent classification emitted at the CLI boundary."""

    code: str
    message: str
    exit_code: int
    next_action: str | None = None


def classify_cli_error(error: BaseException) -> UserFacingError | None:
    """Classify only explicitly translated CLI errors."""
    if isinstance(error, CliUsageError):
        return UserFacingError("invalid_argument", str(error), 2)
    if isinstance(error, CliOperationalError):
        return UserFacingError(error.code, str(error), 1, error.next_action)
    return None


__all__ = ["CliOperationalError", "CliUsageError", "UserFacingError", "classify_cli_error"]
