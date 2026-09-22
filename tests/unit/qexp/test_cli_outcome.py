from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qqtools.plugins.qexp.cli.command_spec import CommandAudience, CommandSpec, ContextKind, OutputMode
from qqtools.plugins.qexp.cli.outcome import CommandOutcome, validate_command_outcome
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind


def _spec(
    *,
    modes: frozenset[OutputMode],
    output_kinds: frozenset[OutputKind] = frozenset(),
) -> CommandSpec:
    return CommandSpec(
        handler="example",
        context=ContextKind.NONE,
        modes=modes,
        output_kinds=output_kinds,
        audience=CommandAudience.NORMAL,
    )


def test_command_outcome_is_format_independent_and_immutable() -> None:
    output = CliOutput(OutputKind.CONTEXT, {"shared_root": "/project/.qexp"})
    outcome = CommandOutcome(exit_code=7, output=output)

    assert outcome.exit_code == 7
    assert outcome.output is output
    assert "format" not in outcome.__dataclass_fields__
    with pytest.raises(FrozenInstanceError):
        outcome.exit_code = 0  # type: ignore[misc]


def test_finite_mode_requires_one_allowed_canonical_output() -> None:
    spec = _spec(modes=frozenset({OutputMode.FINITE}), output_kinds=frozenset({OutputKind.CONTEXT}))
    validate_command_outcome(
        spec,
        OutputMode.FINITE,
        CommandOutcome(0, CliOutput(OutputKind.CONTEXT, {"shared_root": None})),
    )

    with pytest.raises(RuntimeError, match="finite.*output"):
        validate_command_outcome(spec, OutputMode.FINITE, CommandOutcome(0))

    with pytest.raises(RuntimeError, match="context|submission|allowed"):
        validate_command_outcome(
            spec,
            OutputMode.FINITE,
            CommandOutcome(0, CliOutput(OutputKind.SUBMISSION, {})),
        )


@pytest.mark.parametrize("mode", [OutputMode.RAW, OutputMode.CONTINUOUS, OutputMode.DIAGNOSTIC])
def test_non_finite_modes_reject_structured_output(mode: OutputMode) -> None:
    spec = _spec(modes=frozenset({mode}))

    validate_command_outcome(spec, mode, CommandOutcome(0))
    with pytest.raises(RuntimeError, match="output|non-finite"):
        validate_command_outcome(
            spec,
            mode,
            CommandOutcome(0, CliOutput(OutputKind.CONTEXT, {"shared_root": None})),
        )


def test_outcome_mode_must_be_declared_by_the_leaf() -> None:
    spec = _spec(modes=frozenset({OutputMode.FINITE}), output_kinds=frozenset({OutputKind.CONTEXT}))

    with pytest.raises(RuntimeError, match="mode|finite|raw"):
        validate_command_outcome(spec, OutputMode.RAW, CommandOutcome(0))
