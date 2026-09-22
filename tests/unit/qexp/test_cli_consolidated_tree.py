from __future__ import annotations

import argparse
import inspect
import shlex

import pytest

from qqtools.plugins.qexp.cli import _DYNAMIC_OUTPUT_KINDS, build_parser, main
from qqtools.plugins.qexp.commands.registry import CommandSpec
from qqtools.plugins.qexp.formatter import OutputKind


def _leaf_paths(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not subparsers:
        return {prefix}
    result: set[tuple[str, ...]] = set()
    for subparser in subparsers:
        for name, child in subparser.choices.items():
            result.update(_leaf_paths(child, (*prefix, name)))
    return result


def _leaf_specs(parser: argparse.ArgumentParser) -> list[CommandSpec]:
    subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not subparsers:
        spec = parser.get_default("command_spec")
        assert isinstance(spec, CommandSpec)
        return [spec]
    return [spec for subparser in subparsers for child in subparser.choices.values() for spec in _leaf_specs(child)]


def _leaf_parsers(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
    subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not subparsers:
        return [parser]
    return [leaf for subparser in subparsers for child in subparser.choices.values() for leaf in _leaf_parsers(child)]


def test_consolidated_cli_owns_the_canonical_resource_action_paths() -> None:
    leaves = _leaf_paths(build_parser())
    required = {
        ("status",),
        ("task", "wait"),
        ("task", "unshare"),
        ("group", "retry"),
        ("group", "worker", "resume"),
        ("machine", "list"),
        ("machine", "show"),
        ("agent", "config", "gpus", "show"),
        ("agent", "config", "cpu", "show"),
        ("config", "show"),
        ("config", "set"),
        ("config", "reset"),
        ("admin", "check"),
        ("admin", "repair"),
        ("admin", "clean"),
        ("admin", "operation", "show"),
        ("admin", "upgrade", "advance"),
        ("admin", "migrate", "schema"),
        ("admin", "migrate", "schema6", "resume"),
        ("admin", "migrate", "agent"),
    }
    retired = {
        ("top",),
        ("machines",),
        ("doctor",),
        ("clean",),
        ("lease-policy", "show"),
        ("task", "keep-local"),
        ("group", "retry-failed"),
        ("group", "machines", "add"),
        ("agent", "gpus", "show"),
        ("agent", "cpu-lane", "show"),
        ("agent", "upgrade", "coordinate"),
        ("upgrade", "coordinator"),
        ("migrate",),
    }

    assert required <= leaves
    assert retired.isdisjoint(leaves)


def test_root_help_has_descriptions_without_suppressed_placeholders(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["--help"])

    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "==SUPPRESS==" not in output
    for command in ("status", "task", "group", "machine", "agent", "config", "admin"):
        assert command in output


def test_short_daily_options_have_command_local_meanings() -> None:
    parser = build_parser()
    finite = parser.parse_args(["task", "logs", "task-1", "-n", "12"])
    following = parser.parse_args(["task", "logs", "task-1", "-f", "-n", "0"])

    assert finite.tail == "12"
    assert not finite.follow
    assert following.follow
    assert following.tail == "0"


def test_retired_retry_acknowledgement_is_not_accepted() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["task", "retry", "task-1", "--acknowledge-duplicate-risk"])


def test_every_leaf_has_an_executable_handler_and_output_contract() -> None:
    specs = _leaf_specs(build_parser())
    handlers = [spec.handler for spec in specs]
    structured_outputs = {kind.value for kind in OutputKind}
    unstructured_outputs = {"diagnostic", "raw-logs"}

    assert len(handlers) == len(set(handlers))
    assert all(
        spec.output in structured_outputs or spec.output in _DYNAMIC_OUTPUT_KINDS or spec.output in unstructured_outputs
        for spec in specs
    )


def test_every_leaf_help_discloses_its_operational_contract() -> None:
    root = build_parser()
    for parser in _leaf_parsers(root):
        help_text = parser.format_help()
        for heading in ("Scope:", "Prerequisite:", "Activation:", "Output:", "Example:"):
            assert heading in help_text, (parser.prog, heading)
        example = next(
            line.removeprefix("Example: ") for line in parser.epilog.splitlines() if line.startswith("Example: ")
        )
        example_argv = shlex.split(example)
        assert example_argv[0] == "qexp"
        root.parse_args(example_argv[1:])


def test_leaf_help_discloses_dynamic_scope_and_activation_contracts() -> None:
    parsers = {parser.prog: parser for parser in _leaf_parsers(build_parser())}

    retry_help = parsers["qexp task retry"].format_help()
    assert "QEXP_SHARED_ROOT" in retry_help
    assert "QEXP_PROJECT" not in retry_help

    config_help = parsers["qexp config set"].format_help()
    assert "agent section is machine-global" in config_help
    assert "Agent reset is unsupported" in parsers["qexp config reset"].format_help()

    submit_help = parsers["qexp submit"].format_help()
    assert "Activates the local agent after commit by default" in submit_help
    assert "--dry-run and --no-activate suppress it" in submit_help

    assert "--visible VISIBLE" in parsers["qexp agent config gpus set"].epilog
    assert "--reason REASON --project PATH" in parsers["qexp admin upgrade pause"].epilog
    assert "--to-schema 6" in parsers["qexp admin migrate schema"].epilog


def test_main_dispatches_leaf_execution_from_command_spec_handler() -> None:
    source = inspect.getsource(main)

    for argparse_destination in (
        "task_action",
        "dependencies_action",
        "group_action",
        "worker_action",
        "project_action",
        "agent_action",
        "config_action",
        "machine_action",
        "admin_action",
    ):
        assert f"args.{argparse_destination}" not in source
