from __future__ import annotations

import argparse
import json

import pytest

from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.cli.output.core import OutputContract, _build_registry
from qqtools.plugins.qexp.cli.parser import build_parser


def _validate_any(_payload: object) -> None:
    return None


def _render_any(_payload: object, _presentation: object) -> str:
    return "ok"


def test_registry_is_exhaustive_and_each_kind_renders_from_one_contract() -> None:
    from qqtools.plugins.qexp.cli.output.core import _REGISTRY

    assert set(_REGISTRY) == set(OutputKind)
    assert all(callable(contract.validator) and callable(contract.renderer) for contract in _REGISTRY.values())


def test_every_registered_kind_is_reachable_from_a_parser_leaf() -> None:
    def leaf_kinds(parser: argparse.ArgumentParser) -> set[OutputKind]:
        subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
        if not subparsers:
            return set(parser.get_default("command_spec").output_kinds)
        return {kind for subparser in subparsers for child in subparser.choices.values() for kind in leaf_kinds(child)}

    assert leaf_kinds(build_parser()) == set(OutputKind)


def test_registry_builder_rejects_duplicate_output_kind_ownership() -> None:
    contract = OutputContract(_validate_any, _render_any)

    with pytest.raises(RuntimeError, match="duplicate|CONTEXT|context"):
        _build_registry(
            {OutputKind.CONTEXT: contract},
            {OutputKind.CONTEXT: contract},
        )


def test_registry_builder_rejects_missing_output_kind_ownership() -> None:
    from qqtools.plugins.qexp.cli.output.core import _REGISTRY

    incomplete = dict(_REGISTRY)
    incomplete.pop(OutputKind.CONTEXT)

    with pytest.raises(RuntimeError, match="missing|CONTEXT|context"):
        _build_registry(incomplete)


def test_json_serializes_only_the_validated_canonical_payload() -> None:
    payload = {
        "action": "shown",
        "machine_runtime_root": "/machine-runtime",
        "cpu_lane": {"capacity": 3, "revision": 7},
    }
    output = CliOutput(OutputKind.CPU_LANE, payload, {"action": "ignored"})

    assert json.loads(render(output, "json")) == payload


def test_unknown_output_kind_has_no_generic_human_fallback() -> None:
    output = CliOutput("not-registered", {"value": 1})  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match="not-registered"):
        render(output, "human")
