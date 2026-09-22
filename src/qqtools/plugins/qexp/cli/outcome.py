"""Typed dispatch outcomes for the qexp CLI boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .command_spec import CommandSpec, OutputMode
from .output import CliOutput


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Exit status and optional finite structured output from one handler."""

    exit_code: int
    output: CliOutput[object] | None = None


def validate_command_outcome(spec: CommandSpec, mode: OutputMode, outcome: CommandOutcome) -> None:
    """Validate dispatch output against the resolved command classification."""
    if mode not in spec.modes:
        raise RuntimeError(f"command {spec.handler!r} does not support output mode {mode.value!r}")
    output = outcome.output
    if mode is OutputMode.FINITE:
        if output is None:
            raise RuntimeError(f"finite command {spec.handler!r} returned no structured output")
        if not isinstance(output, CliOutput):
            raise RuntimeError(f"finite command {spec.handler!r} returned an invalid structured output")
        if output.kind not in spec.output_kinds:
            raise RuntimeError(
                f"command {spec.handler!r} returned output kind {output.kind.value!r}; "
                f"expected one of {sorted(kind.value for kind in spec.output_kinds)!r}"
            )
        return
    if output is not None:
        raise RuntimeError(f"non-finite command {spec.handler!r} returned structured output")


__all__ = ["CommandOutcome", "validate_command_outcome"]
