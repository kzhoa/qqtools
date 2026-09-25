from __future__ import annotations

import argparse
import hashlib
import inspect
import shlex

import pytest

from qqtools.plugins.qexp.cli.command_spec import CommandAudience, CommandSpec, ContextKind, OutputMode
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.cli.output import OutputKind
from qqtools.plugins.qexp.cli.parser import build_parser


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


def _leaf_spec_map(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], tuple[str, str, str]]:
    subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not subparsers:
        spec = parser.get_default("command_spec")
        assert isinstance(spec, CommandSpec)
        dynamic_outputs = {
            frozenset({OutputKind.AGENT_OPERATION, OutputKind.AGENT_READINESS}): "agent-operation",
            frozenset({OutputKind.DOCTOR_VERIFY}): "doctor-check",
            frozenset({OutputKind.DOCTOR_REPAIR, OutputKind.MACHINE_IDENTITY_DIAGNOSIS}): "doctor-repair",
            frozenset({OutputKind.TASK_LIST, OutputKind.TASK_PAGE}): "task-list",
            frozenset({OutputKind.TASK_SHOW, OutputKind.TASK_WATCH}): "task-show",
            frozenset(
                {
                    OutputKind.UPGRADE_REGISTRY_STATUS,
                    OutputKind.UPGRADE_ADVANCE,
                    OutputKind.UPGRADE_PROJECT,
                }
            ): "upgrade",
        }
        if spec.handler == "agent_run":
            output = "agent-operation"
        elif spec.handler.startswith("group_") and spec.handler not in {"group_list", "group_show"}:
            output = "group-operation"
        elif spec.handler in {"task_cancel", "task_retry"}:
            output = "task-operation"
        elif spec.handler in {"admin_upgrade_pause", "admin_upgrade_resume"}:
            output = "upgrade"
        elif spec.output_kinds in dynamic_outputs:
            output = dynamic_outputs[spec.output_kinds]
        elif len(spec.output_kinds) == 1:
            output = next(iter(spec.output_kinds)).value
        elif spec.modes == frozenset({OutputMode.DIAGNOSTIC}):
            output = "diagnostic"
        else:
            output = "raw-logs"
        return {prefix: (spec.handler, spec.context.value, output)}
    return {
        path: spec
        for subparser in subparsers
        for name, child in subparser.choices.items()
        for path, spec in _leaf_spec_map(child, (*prefix, name)).items()
    }


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


def test_every_command_leaf_retains_its_frozen_dispatch_contract() -> None:
    expected = {
        ("init",): ("init", "machine", "machine-init"),
        ("project", "init"): ("project_init", "setup", "project-operation"),
        ("project", "register"): ("project_register", "machine", "project-register"),
        ("project", "list"): ("project_list", "machine", "project-list"),
        ("project", "enable"): ("project_enable", "machine", "project-operation"),
        ("project", "disable"): ("project_disable", "machine", "project-operation"),
        ("project", "remove"): ("project_remove", "machine", "project-operation"),
        ("use",): ("use", "none", "context"),
        ("submit",): ("submit", "project-write", "submission"),
        ("status",): ("status", "project-read", "status"),
        ("task", "cancel"): ("task_cancel", "project-write", "task-operation"),
        ("task", "attach"): ("task_attach", "project-read", "raw-logs"),
        ("task", "retry"): ("task_retry", "project-write", "task-operation"),
        ("task", "share"): ("task_share", "project-write", "availability"),
        ("task", "unshare"): ("task_unshare", "project-write", "availability"),
        ("task", "offer"): ("task_offer", "project-write", "availability"),
        ("task", "list"): ("task_list", "project-read", "task-list"),
        ("task", "show"): ("task_show", "project-read", "task-show"),
        ("task", "logs"): ("task_logs", "project-read", "raw-logs"),
        ("task", "wait"): ("task_wait", "project-read", "task-wait"),
        ("task", "dependencies", "show"): ("task_dependencies_show", "project-read", "dependencies"),
        ("task", "dependencies", "replace"): ("task_dependencies_replace", "project-write", "dependencies"),
        ("task", "dependencies", "add"): ("task_dependencies_add", "project-write", "dependencies"),
        ("task", "dependencies", "remove"): ("task_dependencies_remove", "project-write", "dependencies"),
        ("group", "create"): ("group_create", "project-write", "group-operation"),
        ("group", "config", "show"): ("group_config_show", "project-read", "group-operation"),
        ("group", "config", "set"): ("group_config_set", "project-write", "group-operation"),
        ("group", "list"): ("group_list", "project-read", "group-list"),
        ("group", "show"): ("group_show", "project-read", "group-show"),
        ("group", "seal"): ("group_seal", "project-write", "group-operation"),
        ("group", "reopen"): ("group_reopen", "project-write", "group-operation"),
        ("group", "pause"): ("group_pause", "project-write", "group-operation"),
        ("group", "resume"): ("group_resume", "project-write", "group-operation"),
        ("group", "cancel"): ("group_cancel", "project-write", "group-operation"),
        ("group", "retry"): ("group_retry", "project-write", "group-operation"),
        ("group", "worker", "list"): ("group_worker_list", "project-read", "group-operation"),
        ("group", "worker", "add"): ("group_worker_add", "project-write", "group-operation"),
        ("group", "worker", "set"): ("group_worker_set", "project-write", "group-operation"),
        ("group", "worker", "drain"): ("group_worker_drain", "project-write", "group-operation"),
        ("group", "worker", "resume"): ("group_worker_resume", "project-write", "group-operation"),
        ("group", "worker", "remove"): ("group_worker_remove", "project-write", "group-operation"),
        ("machine", "list"): ("machine_list", "project-read", "machines"),
        ("machine", "show"): ("machine_show", "project-read", "machine-show"),
        ("agent", "start"): ("agent_start", "machine", "agent-operation"),
        ("agent", "run"): ("agent_run", "machine", "agent-operation"),
        ("agent", "restart"): ("agent_restart", "machine", "agent-operation"),
        ("agent", "status"): ("agent_status", "machine", "agent-status"),
        ("agent", "stop"): ("agent_stop", "machine", "agent-operation"),
        ("agent", "name"): ("agent_name", "machine", "agent-config"),
        ("agent", "add-project"): ("retired_add-project", "none", "diagnostic"),
        ("agent", "list-projects"): ("retired_list-projects", "none", "diagnostic"),
        ("agent", "enable-project"): ("retired_enable-project", "none", "diagnostic"),
        ("agent", "disable-project"): ("retired_disable-project", "none", "diagnostic"),
        ("agent", "remove-project"): ("retired_remove-project", "none", "diagnostic"),
        ("agent", "migrate-project"): ("retired_migrate-project", "none", "diagnostic"),
        ("agent", "config", "gpus", "show"): ("agent_config_gpus_show", "machine", "gpu-policy"),
        ("agent", "config", "gpus", "set"): ("agent_config_gpus_set", "machine", "gpu-policy"),
        ("agent", "config", "gpus", "reset"): ("agent_config_gpus_reset", "machine", "gpu-policy"),
        ("agent", "config", "cpu", "show"): ("agent_config_cpu_show", "machine", "cpu-lane"),
        ("agent", "config", "cpu", "set"): ("agent_config_cpu_set", "machine", "cpu-lane"),
        ("notifications", "setup"): ("notifications_setup", "machine", "config"),
        ("notifications", "show"): ("notifications_show", "machine", "config"),
        ("notifications", "test"): ("notifications_test", "machine", "config"),
        ("notifications", "set"): ("notifications_set", "machine", "config"),
        ("notifications", "reset"): ("notifications_reset", "machine", "config"),
        ("notifications", "resolve"): ("notifications_resolve", "machine", "config"),
        ("config", "show"): ("config_show", "section", "config"),
        ("config", "set"): ("config_set", "section", "config"),
        ("config", "reset"): ("config_reset", "section", "config"),
        ("admin", "check"): ("admin_check", "project-read", "doctor-check"),
        ("admin", "repair"): ("admin_repair", "project-write", "doctor-repair"),
        ("admin", "clean"): ("admin_clean", "project-write", "clean"),
        ("admin", "operation", "show"): ("admin_operation_show", "project-read", "operation"),
        ("admin", "upgrade", "status"): ("admin_upgrade_status", "machine", "upgrade"),
        ("admin", "upgrade", "advance"): ("admin_upgrade_advance", "machine", "upgrade"),
        ("admin", "upgrade", "pause"): ("admin_upgrade_pause", "registered-project", "upgrade"),
        ("admin", "upgrade", "plan"): ("admin_upgrade_plan", "registered-project", "upgrade-repair"),
        ("admin", "upgrade", "apply"): ("admin_upgrade_apply", "registered-project", "upgrade-repair"),
        ("admin", "upgrade", "validate"): ("admin_upgrade_validate", "registered-project", "upgrade-repair"),
        ("admin", "upgrade", "resume"): ("admin_upgrade_resume", "registered-project", "upgrade"),
        ("admin", "migrate", "schema"): ("admin_migrate_schema", "explicit-project", "schema6-upgrade"),
        ("admin", "migrate", "schema6", "check"): (
            "admin_migrate_schema6_check",
            "explicit-project",
            "schema6-upgrade",
        ),
        ("admin", "migrate", "schema6", "start"): (
            "admin_migrate_schema6_start",
            "explicit-project",
            "schema6-upgrade",
        ),
        ("admin", "migrate", "schema6", "status"): (
            "admin_migrate_schema6_status",
            "explicit-project",
            "schema6-upgrade",
        ),
        ("admin", "migrate", "schema6", "attest"): (
            "admin_migrate_schema6_attest",
            "explicit-project",
            "schema6-upgrade",
        ),
        ("admin", "migrate", "schema6", "resume"): (
            "admin_migrate_schema6_resume",
            "explicit-project",
            "schema6-upgrade",
        ),
        ("admin", "migrate", "agent"): ("admin_migrate_agent", "explicit-project", "agent-operation"),
    }

    assert _leaf_spec_map(build_parser()) == expected


def test_normalized_leaf_help_matches_the_characterization_baseline() -> None:
    text = "\n".join(
        f"{' '.join(parser.prog.split()[1:])}\n{' '.join(parser.format_help().split())}"
        for parser in _leaf_parsers(build_parser())
    )

    assert (
        hashlib.sha256(text.encode()).hexdigest() == "d56032bb744acdde116e7de134e985861a82ff80cd085e445de0254ceb7f1c3b"
    )


def test_root_help_has_descriptions_without_suppressed_placeholders(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["--help"])

    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "==SUPPRESS==" not in output
    for command in ("status", "task", "group", "machine", "agent", "config", "admin"):
        assert command in output


def test_admin_repair_help_discloses_both_scopes() -> None:
    parser = build_parser()
    admin = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction)).choices[
        "admin"
    ]
    repair = next(action for action in admin._actions if isinstance(action, argparse._SubParsersAction)).choices[
        "repair"
    ]
    help_text = repair.format_help()

    assert "qexp admin repair identity --dry-run" in help_text
    assert "Project metadata repair requires --project PATH" in help_text
    assert "Identity diagnosis requires --dry-run and no Project" in help_text


def test_short_daily_options_have_command_local_meanings() -> None:
    parser = build_parser()
    finite = parser.parse_args(["task", "logs", "task-1", "-n", "12"])
    following = parser.parse_args(["task", "logs", "task-1", "-f", "-n", "0"])

    assert finite.tail == "12"
    assert not finite.follow
    assert following.follow
    assert following.tail == "0"


def test_common_options_retain_before_after_and_equal_duplicate_placement() -> None:
    parser = build_parser()
    before = parser.parse_args(["--project", "/tmp/example", "task", "list", "--format=json"])
    after = parser.parse_args(["task", "list", "--project", "/tmp/example", "--format=json"])
    duplicate = parser.parse_args(
        ["--project", "/tmp/example", "task", "list", "--project", "/tmp/example", "--format=json"]
    )

    assert (before.project, before.format, before.command_spec) == (after.project, after.format, after.command_spec)
    assert (duplicate.project, duplicate.format, duplicate.command_spec) == (
        before.project,
        before.format,
        before.command_spec,
    )


def test_conflicting_common_option_duplicates_remain_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(["--project", "/tmp/first", "task", "list", "--project", "/tmp/second"])

    assert result.value.code == 2
    assert "conflicting values for --project" in capsys.readouterr().err


def test_retired_retry_acknowledgement_is_not_accepted() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["task", "retry", "task-1", "--acknowledge-duplicate-risk"])


def test_every_leaf_has_an_executable_handler_and_output_contract() -> None:
    specs = _leaf_specs(build_parser())
    handlers = [spec.handler for spec in specs]

    assert len(handlers) == len(set(handlers))
    assert all(isinstance(spec.context, ContextKind) for spec in specs)
    assert all(spec.modes and all(isinstance(mode, OutputMode) for mode in spec.modes) for spec in specs)
    assert all(all(isinstance(kind, OutputKind) for kind in spec.output_kinds) for spec in specs)
    assert all(isinstance(spec.audience, CommandAudience) for spec in specs)


def test_parser_leaf_modes_and_output_kinds_are_explicit() -> None:
    parser = build_parser()
    specs = {
        path: leaf.get_default("command_spec")
        for leaf in _leaf_parsers(parser)
        for path in [tuple(leaf.prog.split()[1:])]
    }

    assert specs[("task", "show")].modes == frozenset({OutputMode.FINITE, OutputMode.CONTINUOUS})
    assert specs[("task", "show")].output_kinds == frozenset({OutputKind.TASK_SHOW, OutputKind.TASK_WATCH})
    assert specs[("task", "logs")].modes == frozenset({OutputMode.RAW, OutputMode.CONTINUOUS})
    assert specs[("task", "logs")].output_kinds == frozenset()
    assert specs[("agent", "run")].modes == frozenset({OutputMode.CONTINUOUS})
    assert specs[("agent", "run")].output_kinds == frozenset()
    assert specs[("agent", "run")].audience is CommandAudience.DEBUG
    assert specs[("init",)].modes == frozenset({OutputMode.FINITE, OutputMode.DIAGNOSTIC})
    assert specs[("agent", "add-project")].modes == frozenset({OutputMode.DIAGNOSTIC})
    assert specs[("agent", "add-project")].output_kinds == frozenset()
    assert specs[("task", "list")].modes == frozenset({OutputMode.FINITE})
    assert specs[("task", "list")].output_kinds == frozenset({OutputKind.TASK_LIST, OutputKind.TASK_PAGE})


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

    agent_run_help = parsers["qexp agent run"].format_help()
    assert "foreground" in agent_run_help.lower()
    assert "debug" in agent_run_help.lower()
    assert "qexp agent start" in agent_run_help
    assert "--format" not in parsers["qexp agent run"]._option_string_actions
    assert "--format" not in parsers["qexp task logs"]._option_string_actions


def test_agent_parent_help_directs_ordinary_users_to_start(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(["agent", "--help"])

    assert result.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "foreground" in help_text.lower()
    assert "debugging" in help_text.lower()
    assert "qexp agent start" in help_text


def test_agent_log_size_is_parsed_once_at_the_config_boundary() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["config", "set", "agent", "--log-max-size", "10MiB"])

    assert parsed.log_max_size == 10 * 1024 * 1024
    with pytest.raises(SystemExit):
        parser.parse_args(["config", "set", "agent", "--log-max-size", "10MB"])


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
