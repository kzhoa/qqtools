"""Small parser binding records used by the qexp command tree.

This is intentionally a command-tree aid, not a plugin framework.  A parser
leaf stores the one handler/context/output contract that the CLI can inspect
without maintaining a second command classification table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Static contract attached to one parser leaf."""

    handler: str
    context: str
    output: str


def _detailed_help(parser: argparse.ArgumentParser, spec: CommandSpec) -> str:
    """Build the disclosure block shared by every executable leaf."""
    machine_wide = spec.context == "machine" or spec.handler.startswith("agent_config_")
    explicit_project = spec.handler in {
        "admin_check",
        "admin_repair",
        "admin_clean",
        "admin_operation_show",
    } or spec.handler.startswith("admin_migrate_")
    if spec.handler.startswith("config_"):
        scope = "The agent section is machine-global; other sections belong to the selected Project."
        prerequisite = (
            "Agent show/set requires an initialized MachineRuntime; Project sections use --project, "
            "QEXP_SHARED_ROOT, saved context, or cwd discovery. Agent reset is unsupported."
        )
    elif spec.handler == "init":
        scope = "The MachineRuntime selected by --machine-runtime-root."
        prerequisite = "Pass an explicit --machine NAME; no Project selection is used."
    elif machine_wide:
        scope = "Machine-wide; --project does not narrow this operation."
        prerequisite = "Initialize the MachineRuntime with `qexp init --machine NAME` first."
    elif explicit_project:
        scope = "One explicitly selected Project."
        prerequisite = "Pass --project PATH; cwd, environment, and saved-context fallback are not accepted."
    elif spec.context in {"project-read", "project-write", "section"}:
        scope = "The selected Project."
        prerequisite = "Select a Project with --project, QEXP_SHARED_ROOT, saved context, or cwd discovery."
    elif spec.context == "registered-project":
        scope = "One Project registered in the local MachineRuntime."
        prerequisite = "Pass --project PATH after registering the Project on this machine."
    else:
        scope = "Local command context."
        prerequisite = "No implicit Project mutation is performed."

    activates = spec.handler in {
        "task_retry",
        "task_offer",
        "task_share",
        "task_unshare",
        "group_retry",
        "group_resume",
        "group_cancel",
        "group_worker_remove",
    }
    if spec.handler == "submit":
        activation = "Activates the local agent after commit by default; --dry-run and --no-activate suppress it."
    elif spec.handler == "admin_migrate_agent":
        activation = "Starts or wakes the machine-wide agent after importing the legacy Project."
    elif spec.handler in {"agent_start", "agent_run", "agent_restart", "agent_stop"}:
        activation = "Explicitly controls the machine-wide agent lifecycle."
    elif activates:
        activation = "May activate the local machine agent to converge accepted work."
    else:
        activation = "Does not implicitly activate an agent unless the command result requires lifecycle convergence."
    if spec.output == "diagnostic":
        output = "Non-executing compatibility diagnostic on stderr."
    elif spec.output == "raw-logs" or spec.handler == "agent_run":
        output = "Continuous/raw stream; finite JSON wrapping does not apply."
    elif spec.handler == "task_show":
        output = "Finite human/JSON result by default; --watch is a terminal-only continuous stream."
    else:
        output = "Finite result supports --format human or --format json."
    positional: list[str] = []
    for action in parser._actions:
        if action.option_strings or action.dest in {"help", argparse.SUPPRESS}:
            continue
        if action.dest == "argv":
            positional.append("-- COMMAND [ARG...]")
        elif action.nargs in ("*", "?", argparse.REMAINDER):
            continue
        elif action.nargs == "+":
            positional.append(action.dest.upper())
        elif action.choices:
            positional.append(str(next(iter(action.choices))))
        else:
            positional.append(action.dest.upper())
    required_options: list[str] = []
    mutually_exclusive_actions = {
        action for group in parser._mutually_exclusive_groups if group.required for action in group._group_actions[:1]
    }
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        if not action.required and action not in mutually_exclusive_actions:
            continue
        option = action.option_strings[0]
        required_options.append(option)
        if action.nargs != 0:
            if action.choices:
                value = str(next(iter(action.choices)))
            elif action.type in {int, float}:
                value = "1"
            else:
                value = action.metavar or action.dest.upper()
            required_options.append(value)
    example_overrides = {
        "init": "qexp init --machine NAME",
        "project_register": "qexp project register PATH",
        "use": "qexp use --project PATH",
        "submit": "qexp submit --project PATH -- COMMAND [ARG...]",
        "config_set": "qexp config set progress --interval-seconds 30 --project PATH",
        "admin_migrate_schema": "qexp admin migrate schema --project PATH --to-schema 6",
        "admin_migrate_schema6_attest": (
            "qexp admin migrate schema6 attest --project PATH --machine NAME "
            "--activation-id ID --confirm-clients-stopped"
        ),
        "admin_migrate_agent": "qexp admin migrate agent --project PATH --machine NAME",
    }
    example = example_overrides.get(spec.handler, " ".join((parser.prog, *positional, *required_options)))
    if spec.context == "registered-project" or (explicit_project and "--project" not in example):
        example += " --project PATH"
    return "\n".join(
        (
            f"Scope: {scope}",
            f"Prerequisite: {prerequisite}",
            f"Activation: {activation}",
            f"Output: {output}",
            f"Example: {example}",
        )
    )


def bind_command(parser: Any, *, handler: str, context: str, output: str) -> Any:
    """Attach one command specification to an argparse parser."""
    spec = CommandSpec(handler, context, output)
    parser.set_defaults(command_spec=spec)
    parser.epilog = _detailed_help(parser, spec)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    return parser


__all__ = ["CommandSpec", "bind_command"]
