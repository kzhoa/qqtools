"""Argparse construction and parser-only diagnostics for qexp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..commands import configuration as configuration_commands
from ..commands import context as context_commands
from ..launch_policy import validate_launch_handoff_timeout_seconds
from .command_spec import CommandAudience, ContextKind, OutputMode, bind_command
from .output import OutputKind

_PAGINATION_JSON_PARSE_MODE = False
_SUBMISSION_JSON_PARSE_MODE = False
_JSON_PARSE_MODE = False


class _PaginationParseError(RuntimeError):
    """An argparse error that belongs to paginated JSON output."""


class _SubmissionParseError(RuntimeError):
    """An argparse error that belongs to structured submission JSON output."""


class _JsonParseError(RuntimeError):
    """An argparse error that belongs to another finite JSON invocation."""


class _QexpArgumentParser(argparse.ArgumentParser):
    """Use structured parse errors only for the narrowly scoped page mode."""

    def error(self, message: str) -> None:
        if _PAGINATION_JSON_PARSE_MODE:
            raise _PaginationParseError(message)
        if _SUBMISSION_JSON_PARSE_MODE:
            raise _SubmissionParseError(message)
        if _JSON_PARSE_MODE:
            raise _JsonParseError(message)
        super().error(message)

    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None):
        raw = list(sys.argv[1:] if args is None else args)
        conflict = _common_option_conflict(raw)
        if conflict is not None:
            self.error(conflict)
        return super().parse_args(raw, namespace)


_COMMON_OPTIONS = frozenset({"--project", "--machine", "--runtime-root", "--machine-runtime-root", "--format"})


def _normalized_common_value(name: str, value: str) -> str:
    if name == "--project":
        return str(context_commands.normalize_project_path(value))
    if name in {"--runtime-root", "--machine-runtime-root"}:
        return str(Path(value).expanduser().resolve())
    return value


def _common_option_conflict(argv: list[str]) -> str | None:
    """Reject conflicting duplicate common options before argparse dispatches."""
    values: dict[str, str] = {}
    stop = len(argv)
    try:
        submit_index = argv.index("submit")
    except ValueError:
        submit_index = -1
    if submit_index >= 0:
        try:
            separator = argv.index("--", submit_index + 1)
        except ValueError:
            separator = len(argv)
        stop = separator
    index = 0
    while index < stop:
        token = argv[index]
        name = token.split("=", 1)[0]
        if name not in _COMMON_OPTIONS:
            index += 1
            continue
        if "=" in token:
            value = token.split("=", 1)[1]
        elif index + 1 < stop:
            value = argv[index + 1]
            index += 1
        else:
            index += 1
            continue
        normalized = _normalized_common_value(name, value)
        previous = values.get(name)
        if previous is not None and previous != normalized:
            return f"conflicting values for {name}: {previous!r} and {value!r}."
        values[name] = normalized
        index += 1
    return None


def _is_paginated_json_argv(argv: list[str]) -> bool:
    """Return whether raw arguments select JSON task-list pagination."""
    task_index = next(
        (index for index in range(len(argv) - 1) if argv[index : index + 2] == ["task", "list"]),
        None,
    )
    if task_index is None:
        return False
    has_pagination_flag = any(
        item == "--page-size" or item.startswith("--page-size=") or item == "--cursor" or item.startswith("--cursor=")
        for item in argv[task_index + 2 :]
    )
    if not has_pagination_flag:
        return False
    return any(
        item == "--format=json" or (item == "--format" and index + 1 < len(argv) and argv[index + 1] == "json")
        for index, item in enumerate(argv)
    )


def _submission_raw_prefix(argv: list[str]) -> list[str]:
    """Return submit options before the literal command separator."""
    try:
        submit_index = argv.index("submit")
    except ValueError:
        return []
    prefix = argv[submit_index + 1 :]
    try:
        return prefix[: prefix.index("--")]
    except ValueError:
        return prefix


def _is_submission_json_argv(argv: list[str]) -> bool:
    prefix = _submission_raw_prefix(argv)
    return bool(prefix) and any(
        item == "--format=json" or (item == "--format" and index + 1 < len(prefix) and prefix[index + 1] == "json")
        for index, item in enumerate(prefix)
    )


def _is_json_argv(argv: list[str]) -> bool:
    """Return whether a pre-payload option requests JSON diagnostics."""
    stop = len(argv)
    try:
        submit_index = argv.index("submit")
        stop = argv.index("--", submit_index + 1)
    except (ValueError, IndexError):
        pass
    return any(
        item == "--format=json" or (item == "--format" and index + 1 < stop and argv[index + 1] == "json")
        for index, item in enumerate(argv[:stop])
    )


def _set_parse_modes(argv: list[str] | None) -> None:
    """Install or reset raw-argv parse diagnostics for one invocation."""
    global _PAGINATION_JSON_PARSE_MODE, _SUBMISSION_JSON_PARSE_MODE, _JSON_PARSE_MODE
    if argv is None:
        _PAGINATION_JSON_PARSE_MODE = False
        _SUBMISSION_JSON_PARSE_MODE = False
        _JSON_PARSE_MODE = False
        return
    _PAGINATION_JSON_PARSE_MODE = _is_paginated_json_argv(argv)
    _SUBMISSION_JSON_PARSE_MODE = _is_submission_json_argv(argv)
    _JSON_PARSE_MODE = _is_json_argv(argv)


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    """Add a leaf format option without overwriting a root-level value."""
    if "--format" not in parser._option_string_actions:
        parser.add_argument("--format", choices=("human", "json"), default=argparse.SUPPRESS)


def _add_common_options(parser: argparse.ArgumentParser, *, include_format: bool = True) -> None:
    """Allow common context options at every ordinary command level."""
    options = (
        ("--project", {"help": "Project directory or its .qexp control directory."}),
        ("--machine", {"help": "Assert the local logical machine identity."}),
        ("--runtime-root", {"help": "Override the project-local runtime root."}),
        ("--machine-runtime-root", {"help": "Override the machine-global runtime root."}),
    )
    for option, kwargs in options:
        if option not in parser._option_string_actions:
            parser.add_argument(option, default=argparse.SUPPRESS, **kwargs)
    # Foreground debugging intentionally has no finite structured result.
    # Keep the option absent even when this recursive helper visits the leaf.
    path = parser.prog.rsplit(" ", 2)[-2:]
    is_nonfinite_leaf = path in (["agent", "run"], ["task", "logs"])
    if include_format and not is_nonfinite_leaf and "--format" not in parser._option_string_actions:
        parser.add_argument("--format", choices=("human", "json"), default=argparse.SUPPRESS)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _add_common_options(child, include_format=include_format)


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical qexp resource/action command tree."""
    parser = _QexpArgumentParser(
        prog="qexp",
        description=(
            "qexp schema-6 experiment queue. Project commands use --project; --machine asserts local identity, "
            "while --home-machine selects Task placement. Selecting placement does not remotely start an agent."
        ),
        epilog=(
            "Machine setup: qexp init --machine NAME. Create shared truth with qexp project init PATH, "
            "then enroll it with qexp project register PATH. qexp use only saves local CLI fallback context."
        ),
    )
    parser.add_argument("--project", help="Project directory or its .qexp control directory.")
    # Parse the former root-position spelling only far enough to provide the
    # bounded QQTOOLS-COMPAT-0014 diagnostic.  It is intentionally hidden from
    # help and never participates in ordinary Project selection.
    parser.add_argument("--shared-root", dest="compat_shared_root", help=argparse.SUPPRESS)
    parser.add_argument("--machine", help="Assert the local logical machine identity.")
    parser.add_argument("--runtime-root", help="Override the project-local runtime root.")
    parser.add_argument("--machine-runtime-root", help="Override the machine-global runtime root.")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init",
        help="Initialize or replace the machine runtime.",
        description=(
            "Initialize the local machine identity and global agent policy. Project creation and enrollment are "
            "separate; use qexp project register PATH after creating shared truth."
        ),
    )
    init.add_argument("--shared-root", dest="init_shared_root")
    init.add_argument("--machine", dest="init_machine")
    init.add_argument("--agent-mode", choices=("on_demand", "daemon"))
    init.add_argument("--detach-old-runtime", action="store_true")
    init.add_argument("--yes", action="store_true")
    init.add_argument("--cpu-lane-capacity", type=int)
    _add_output_format(init)
    bind_command(
        init,
        handler="init",
        context=ContextKind.MACHINE,
        modes={OutputMode.FINITE, OutputMode.DIAGNOSTIC},
        output_kinds=OutputKind.MACHINE_INIT,
    )

    project = commands.add_parser(
        "project",
        help="Create shared Projects and manage local enrollment.",
        description="Project setup owns shared truth and machine enrollment.",
    )
    project_sub = project.add_subparsers(dest="project_action", required=True)
    project_init = project_sub.add_parser("init", help="Create shared Project truth without enrollment.")
    project_init.add_argument("path", nargs="?")
    _add_output_format(project_init)
    bind_command(
        project_init,
        handler="project_init",
        context=ContextKind.SETUP,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.PROJECT_OPERATION,
    )
    project_register = project_sub.add_parser("register", help="Enroll explicit Projects or saved inventory entries.")
    project_register.add_argument("paths", nargs="*")
    project_register.add_argument("--from-pool", action="store_true")
    project_register.add_argument("--machine", dest="project_machine")
    project_register.add_argument("--name-source", choices=("default", "explicit"))
    _add_output_format(project_register)
    bind_command(
        project_register,
        handler="project_register",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.PROJECT_REGISTER,
    )
    project_list = project_sub.add_parser("list", help="List local Project registrations.")
    _add_output_format(project_list)
    bind_command(
        project_list,
        handler="project_list",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.PROJECT_LIST,
    )
    for project_action in ("enable", "disable", "remove"):
        selector = project_sub.add_parser(project_action, help=f"{project_action.capitalize()} a Project registration.")
        selector.add_argument("selector")
        _add_output_format(selector)
        bind_command(
            selector,
            handler=f"project_{project_action}",
            context=ContextKind.MACHINE,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.PROJECT_OPERATION,
        )

    use = commands.add_parser(
        "use",
        help="Save or inspect fallback Project context without initialization.",
        description=(
            "Use --project to save a fallback locator, or --show/--clear to inspect or clear it. "
            "This does not initialize a shared root or register the project with the local machine agent."
        ),
    )
    use.add_argument("--project", dest="use_project", help="Project directory or .qexp control directory to save.")
    use.add_argument("--show", action="store_true", help="Show the saved fallback Project.")
    use.add_argument("--clear", action="store_true", help="Clear the saved fallback Project.")
    _add_output_format(use)
    bind_command(
        use,
        handler="use",
        context=ContextKind.NONE,
        modes={OutputMode.FINITE, OutputMode.RAW},
        output_kinds=OutputKind.CONTEXT,
    )

    submit = commands.add_parser(
        "submit",
        help="Submit one Task using the verified local identity.",
        description=(
            "Submit one command or a manifest using the verified local identity. "
            "qexp does not remotely start the target agent."
        ),
    )
    _add_output_format(submit)
    submit.add_argument("--project", dest="project", default=argparse.SUPPRESS)
    submit.add_argument("-f", "--file", dest="manifest_file")
    submit.add_argument("--task-id")
    submit.add_argument("--name")
    submit.add_argument("--group")
    submit.add_argument("--gpus", type=int, default=None)
    submit.add_argument("--cpus", type=int, default=None)
    submit.add_argument("--cwd")
    submit.add_argument("--home-machine", default=None)
    submit.add_argument("--sharing", choices=["private", "spillover"], default=None)
    submit.add_argument("--offer-after-seconds", type=int, default=None)
    submit.add_argument("--idempotency-key")
    submit.add_argument("--depends-on", action="append", default=None)
    submit.add_argument("--no-activate", action="store_true", help="Submit without activating the local agent.")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--quiet", action="store_true")
    tmux_values = submit.add_mutually_exclusive_group()
    tmux_values.add_argument("--tmux", dest="tmux_override", action="store_true")
    tmux_values.add_argument("--no-tmux", dest="tmux_override", action="store_false")
    submit.set_defaults(tmux_override=None)
    submit.add_argument("argv", nargs=argparse.REMAINDER)
    bind_command(
        submit,
        handler="submit",
        context=ContextKind.PROJECT_WRITE,
        modes={OutputMode.FINITE, OutputMode.RAW},
        output_kinds=OutputKind.SUBMISSION,
    )

    status = commands.add_parser("status", help="Show a bounded Project overview and next actions.")
    _add_output_format(status)
    bind_command(
        status,
        handler="status",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.STATUS,
    )

    task = commands.add_parser("task", help="Inspect and operate on Tasks in the selected Project.")
    task_sub = task.add_subparsers(dest="task_action", required=True)
    cancel = task_sub.add_parser("cancel", help="Cancel one Task and preserve its lifecycle identity.")
    cancel.add_argument("task_id")
    _add_output_format(cancel)
    bind_command(
        cancel,
        handler="task_cancel",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.TASK_CANCEL,
    )
    retry = task_sub.add_parser("retry", help="Queue the next Attempt for one failed Task.")
    retry.add_argument("task_id")
    retry.add_argument("--quiet", action="store_true", help="Print only the retained Task ID.")
    _add_output_format(retry)
    bind_command(
        retry,
        handler="task_retry",
        context=ContextKind.PROJECT_WRITE,
        modes={OutputMode.FINITE, OutputMode.RAW},
        output_kinds=OutputKind.TASK_RETRY,
    )
    share = task_sub.add_parser("share", help="Enable immediate or delayed Task sharing.")
    share.add_argument("task_id")
    share.add_argument("--after")
    share.add_argument("--with", dest="helper_machines", action="append")
    _add_output_format(share)
    bind_command(
        share,
        handler="task_share",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.AVAILABILITY,
    )
    unshare = task_sub.add_parser("unshare", help="Return a queued Task to private home placement.")
    unshare.add_argument("task_id")
    _add_output_format(unshare)
    bind_command(
        unshare,
        handler="task_unshare",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.AVAILABILITY,
    )
    offer = task_sub.add_parser("offer", help="Offer eligible spillover work immediately.")
    offer.add_argument("task_id")
    _add_output_format(offer)
    bind_command(
        offer,
        handler="task_offer",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.AVAILABILITY,
    )
    listing = task_sub.add_parser("list", help="List Tasks with bounded filters and pagination.")
    listing.add_argument("--phase")
    listing.add_argument("--group")
    listing.add_argument("--name", help="Exact, case-sensitive Task name filter.")
    listing.add_argument("--limit", type=int, default=None)
    listing.add_argument("--page-size", default=None)
    listing.add_argument("--cursor", default=None)
    _add_output_format(listing)
    bind_command(
        listing,
        handler="task_list",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds={OutputKind.TASK_LIST, OutputKind.TASK_PAGE},
    )
    show = task_sub.add_parser(
        "show",
        help="Show one Task and its bounded Attempt history.",
        description=(
            "Show one Task. --watch performs screen refreshes; it does not change progress reporting. "
            "--follow-retries continues onto a later retry."
        ),
    )
    show.add_argument("task_id")
    show.add_argument("--watch", action="store_true")
    show.add_argument("--interval-seconds", default=None)
    show.add_argument("--follow-retries", action="store_true")
    _add_output_format(show)
    bind_command(
        show,
        handler="task_show",
        context=ContextKind.PROJECT_READ,
        modes={OutputMode.FINITE, OutputMode.CONTINUOUS},
        output_kinds={OutputKind.TASK_SHOW, OutputKind.TASK_WATCH},
    )
    logs = task_sub.add_parser(
        "logs",
        help="Read Task logs or explicitly follow a live byte stream.",
        description=(
            "Read finite logs or follow each Attempt and file generation. --follow-retries continues onto a later "
            "retry."
        ),
    )
    logs.add_argument("task_id")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--tail", default=None)
    logs.add_argument("--interval-seconds", default=None)
    logs.add_argument("--follow-retries", action="store_true")
    bind_command(
        logs,
        handler="task_logs",
        context=ContextKind.PROJECT_READ,
        modes={OutputMode.RAW, OutputMode.CONTINUOUS},
        output_kinds=(),
    )
    wait = task_sub.add_parser("wait", help="Wait for one selected Task lifecycle without mutating it.")
    wait.add_argument("task_id")
    wait.add_argument("--timeout", default=None)
    _add_output_format(wait)
    bind_command(
        wait,
        handler="task_wait",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.TASK_WAIT,
    )
    dependencies = task_sub.add_parser("dependencies", help="Inspect or atomically edit Task dependencies.")
    dependencies_sub = dependencies.add_subparsers(dest="dependencies_action", required=True)
    for name in ("show", "replace", "add", "remove"):
        action = dependencies_sub.add_parser(name, help=f"{name.capitalize()} Task dependencies.")
        action.add_argument("task_id")
        if name != "show":
            action.add_argument("--depends-on", action="append", default=[])
        _add_output_format(action)
        bind_command(
            action,
            handler=f"task_dependencies_{name}",
            context=ContextKind.PROJECT_WRITE if name != "show" else ContextKind.PROJECT_READ,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.DEPENDENCIES,
        )

    group = commands.add_parser("group", help="Inspect Groups and control admission, dispatch, and workers.")
    group_sub = group.add_subparsers(dest="group_action", required=True)
    create = group_sub.add_parser("create", help="Create a Group and its initial Worker Set.")
    create.add_argument("name")
    create.add_argument("--workers", nargs="*", default=None)
    _add_output_format(create)
    bind_command(
        create,
        handler="group_create",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GROUP_STATE_CHANGE,
    )
    group_list = group_sub.add_parser("list", help="List Groups in the selected Project.")
    _add_output_format(group_list)
    bind_command(
        group_list,
        handler="group_list",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GROUP_LIST,
    )
    show_group = group_sub.add_parser("show", help="Show one Group and its Worker Set.")
    show_group.add_argument("name")
    _add_output_format(show_group)
    bind_command(
        show_group,
        handler="group_show",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GROUP_SHOW,
    )
    for name in ("seal", "reopen", "pause", "resume", "cancel", "retry"):
        action = group_sub.add_parser(name, help=f"{name.capitalize()} Group control state.")
        action.add_argument("name")
        if name == "cancel":
            action.add_argument("--all", action="store_true", help="Include launch-authorized and running Tasks.")
        _add_output_format(action)
        bind_command(
            action,
            handler=f"group_{name}",
            context=ContextKind.PROJECT_WRITE,
            modes=OutputMode.FINITE,
            output_kinds=(
                OutputKind.GROUP_CANCEL
                if name == "cancel"
                else OutputKind.GROUP_RETRY
                if name == "retry"
                else OutputKind.GROUP_STATE_CHANGE
            ),
        )
    worker = group_sub.add_parser("worker", help="Manage Group Worker Set membership and lifecycle.")
    worker_sub = worker.add_subparsers(dest="worker_action", required=True)
    for name in ("list", "add", "set", "drain", "resume", "remove"):
        action = worker_sub.add_parser(name, help=f"{name.capitalize()} a Group worker.")
        action.add_argument("group_name")
        if name != "list":
            action.add_argument("worker_machine")
        if name in {"add", "set"}:
            action.add_argument("--role", choices=("primary", "borrow"))
            action.add_argument("--gpu-limit-gpus", type=_gpu_limit_gpus)
        if name == "remove":
            action.add_argument("--all", action="store_true", help="Terminate running work while removing.")
        _add_output_format(action)
        bind_command(
            action,
            handler=f"group_worker_{name}",
            context=ContextKind.PROJECT_WRITE if name != "list" else ContextKind.PROJECT_READ,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.GROUP_MACHINES if name == "list" else OutputKind.GROUP_WORKER_CHANGE,
        )

    machine = commands.add_parser("machine", help="Inspect declared Project machines.")
    machine_sub = machine.add_subparsers(dest="machine_action", required=True)
    machine_list = machine_sub.add_parser("list", help="List bounded machine observations.")
    _add_output_format(machine_list)
    bind_command(
        machine_list,
        handler="machine_list",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.MACHINES,
    )
    machine_show = machine_sub.add_parser("show", help="Show one machine declaration and observations.")
    machine_show.add_argument("name")
    _add_output_format(machine_show)
    bind_command(
        machine_show,
        handler="machine_show",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.MACHINE_SHOW,
    )

    agent = commands.add_parser("agent", help="Manage the local machine agent and its global resources.")
    agent_sub = agent.add_subparsers(dest="agent_action", required=True)
    for name in ("start", "run", "restart", "status", "stop"):
        help_text = (
            "Run the machine-wide local agent in the foreground for debugging; use qexp agent start for ordinary use."
            if name == "run"
            else f"{name.capitalize()} the machine-wide local agent."
        )
        action = agent_sub.add_parser(name, help=help_text)
        if name != "run":
            _add_output_format(action)
        if name == "run":
            bind_command(
                action,
                handler="agent_run",
                context=ContextKind.MACHINE,
                modes=OutputMode.CONTINUOUS,
                output_kinds=(),
                audience=CommandAudience.DEBUG,
            )
        else:
            bind_command(
                action,
                handler=f"agent_{name}",
                context=ContextKind.MACHINE,
                modes=OutputMode.FINITE,
                output_kinds=(
                    {OutputKind.AGENT_OPERATION, OutputKind.AGENT_READINESS}
                    if name == "start"
                    else OutputKind.AGENT_STATUS
                    if name == "status"
                    else OutputKind.AGENT_OPERATION
                ),
            )
    agent_sub.choices["start"].add_argument("--timeout", type=float, default=30.0)
    agent_name = agent_sub.add_parser("name", help="Show or change the machine-global agent name.")
    agent_name.add_argument("--set-to", dest="set_to")
    _add_output_format(agent_name)
    bind_command(
        agent_name,
        handler="agent_name",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.AGENT_CONFIG,
    )
    for legacy_name in (
        "add-project",
        "list-projects",
        "enable-project",
        "disable-project",
        "remove-project",
        "migrate-project",
    ):
        legacy = agent_sub.add_parser(
            legacy_name,
            help="Retired non-executing compatibility diagnostic for Project enrollment commands.",
        )
        legacy.add_argument("legacy_selector", nargs="?")
        if legacy_name == "add-project":
            legacy.add_argument("--adopt-existing", action="store_true")
        bind_command(
            legacy,
            handler=f"retired_{legacy_name}",
            context=ContextKind.NONE,
            modes=OutputMode.DIAGNOSTIC,
            output_kinds=(),
        )
    agent_config = agent_sub.add_parser("config", help="Inspect or set machine-wide agent resources.")
    agent_config_sub = agent_config.add_subparsers(dest="agent_config_resource", required=True)
    gpus = agent_config_sub.add_parser("gpus", help="Inspect or change GPU admission policy.")
    gpus_sub = gpus.add_subparsers(dest="gpu_action", required=True)
    gpu_show = gpus_sub.add_parser("show", help="Show GPU admission policy.")
    _add_output_format(gpu_show)
    bind_command(
        gpu_show,
        handler="agent_config_gpus_show",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GPU_POLICY,
    )
    gpu_set = gpus_sub.add_parser("set", help="Set visible GPU IDs or disable GPU admission.")
    gpu_set_values = gpu_set.add_mutually_exclusive_group(required=True)
    gpu_set_values.add_argument("--visible")
    gpu_set_values.add_argument("--none", action="store_true")
    gpu_set.add_argument("--expected-revision", type=int)
    _add_output_format(gpu_set)
    bind_command(
        gpu_set,
        handler="agent_config_gpus_set",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GPU_POLICY,
    )
    gpu_reset = gpus_sub.add_parser("reset", help="Reset GPU admission policy.")
    gpu_reset.add_argument("--expected-revision", type=int)
    _add_output_format(gpu_reset)
    bind_command(
        gpu_reset,
        handler="agent_config_gpus_reset",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.GPU_POLICY,
    )
    cpu = agent_config_sub.add_parser("cpu", help="Inspect or set machine-wide CPU lane capacity.")
    cpu_sub = cpu.add_subparsers(dest="cpu_action", required=True)
    cpu_show = cpu_sub.add_parser("show", help="Show CPU lane capacity.")
    _add_output_format(cpu_show)
    bind_command(
        cpu_show,
        handler="agent_config_cpu_show",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CPU_LANE,
    )
    cpu_set = cpu_sub.add_parser("set", help="Set CPU lane capacity.")
    cpu_set.add_argument("--capacity", type=int, required=True)
    _add_output_format(cpu_set)
    bind_command(
        cpu_set,
        handler="agent_config_cpu_set",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CPU_LANE,
    )

    notifications = commands.add_parser(
        "notifications",
        help="Set up, inspect, test, and maintain Feishu notifications.",
        description=(
            "Convenience commands for Feishu notifications. Global scope applies to the selected MachineRuntime; "
            "select --scope project explicitly to operate on a Project. Use qexp config show/set/reset notifications "
            "for detailed configuration fields."
        ),
    )
    notifications_sub = notifications.add_subparsers(dest="notifications_action", required=True)
    notification_scope_help = "Policy scope; global applies to the selected MachineRuntime."
    notification_setup = notifications_sub.add_parser("setup", help="Set up and enable notifications.")
    notification_setup.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    notification_webhook = notification_setup.add_mutually_exclusive_group()
    notification_webhook.add_argument("--webhook", type=str, metavar="URL")
    notification_webhook.add_argument("--webhook-stdin", action="store_true")
    notification_webhook.add_argument("--webhook-env", metavar="NAME")
    notification_signing = notification_setup.add_mutually_exclusive_group()
    notification_signing.add_argument("--secret-env", metavar="NAME")
    notification_signing.add_argument("--unsigned", action="store_true")
    _add_output_format(notification_setup)
    bind_command(
        notification_setup,
        handler="notifications_setup",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    notification_show = notifications_sub.add_parser("show", help="Show notification configuration.")
    notification_show.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    _add_output_format(notification_show)
    bind_command(
        notification_show,
        handler="notifications_show",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    notification_test = notifications_sub.add_parser("test", help="Send a notification test message.")
    notification_test.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    _add_output_format(notification_test)
    bind_command(
        notification_test,
        handler="notifications_test",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    notification_set = notifications_sub.add_parser("set", help="Edit notification settings.")
    notification_set.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    notification_enabled = notification_set.add_mutually_exclusive_group()
    notification_enabled.add_argument("--enabled", action="store_true", default=None)
    notification_enabled.add_argument("--disabled", action="store_true", default=None)
    notification_set.add_argument("--webhook-env", metavar="NAME")
    notification_set_signing = notification_set.add_mutually_exclusive_group()
    notification_set_signing.add_argument("--secret-env", metavar="NAME")
    notification_set_signing.add_argument("--unsigned", action="store_true")
    notification_set.add_argument("--timeout-seconds", type=float)
    _add_output_format(notification_set)
    bind_command(
        notification_set,
        handler="notifications_set",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    notification_reset = notifications_sub.add_parser("reset", help="Reset notification settings.")
    notification_reset.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    _add_output_format(notification_reset)
    bind_command(
        notification_reset,
        handler="notifications_reset",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    notification_resolve = notifications_sub.add_parser(
        "resolve", help="Resolve a notification configuration conflict."
    )
    notification_resolve.add_argument(
        "--scope",
        choices=("global", "project"),
        default="global",
        help=notification_scope_help,
    )
    notification_resolve.add_argument("--prefer", choices=("canonical", "legacy"), required=True)
    _add_output_format(notification_resolve)
    bind_command(
        notification_resolve,
        handler="notifications_resolve",
        context=ContextKind.MACHINE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    config = commands.add_parser(
        "config",
        help="Inspect or set typed Project and global agent configuration.",
    )
    config_sub = config.add_subparsers(dest="config_action", required=True)
    config_show = config_sub.add_parser("show", help="Show all Project sections or one typed section.")
    config_show.add_argument("section", nargs="?", choices=configuration_commands.CONFIG_SECTIONS)
    config_show.add_argument("--scope", choices=("global", "project"), default="project")
    _add_output_format(config_show)
    bind_command(
        config_show,
        handler="config_show",
        context=ContextKind.SECTION,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )
    config_set = config_sub.add_parser("set", help="Set typed options in one configuration section.")
    config_set.add_argument("section", choices=configuration_commands.CONFIG_SECTIONS)
    config_set.add_argument("--scope", choices=("global", "project"), default="project")
    config_set.add_argument("--provider")
    config_set.add_argument("--name")
    config_set.add_argument("--agent-mode", choices=("daemon", "on_demand"))
    config_boolean = config_set.add_mutually_exclusive_group()
    config_boolean.add_argument("--enabled", action="store_true", default=None)
    config_boolean.add_argument("--disabled", action="store_true", default=None)
    config_set.add_argument("--interval-seconds", type=float)
    config_set.add_argument("--timeout-seconds", type=float)
    config_set.add_argument("--ttl-seconds", type=int)
    config_set.add_argument("--renew-interval-seconds", type=float)
    config_set.add_argument("--max-clock-skew-seconds", type=float)
    config_set.add_argument("--clock-observation-max-age-seconds", type=float)
    config_set.add_argument("--clock-provider-margin-seconds", type=float)
    config_set.add_argument("--clock-provider-priority")
    config_set.add_argument("--renewal-commit-margin-seconds", type=float)
    config_set.add_argument("--webhook-env")
    config_set.add_argument("--credential-source", choices=("env", "shared_file"))
    config_set.add_argument("--webhook-stdin", action="store_true")
    config_set.add_argument("--acknowledge-shared-secret-risk", action="store_true")
    config_set.add_argument("--secret-env")
    config_set.add_argument("--unset-secret-env", action="store_true")
    config_set.add_argument("--shared-webhook")
    _add_output_format(config_set)
    bind_command(
        config_set,
        handler="config_set",
        context=ContextKind.SECTION,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )
    config_reset = config_sub.add_parser("reset", help="Reset one typed configuration override.")
    config_reset.add_argument("section", choices=configuration_commands.CONFIG_SECTIONS)
    config_reset.add_argument("--scope", choices=("global", "project"), default="project")
    config_reset.add_argument("--provider")
    _add_output_format(config_reset)
    bind_command(
        config_reset,
        handler="config_reset",
        context=ContextKind.SECTION,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CONFIG,
    )

    admin = commands.add_parser("admin", help="Run bounded Project maintenance and machine upgrade operations.")
    admin_sub = admin.add_subparsers(dest="admin_action", required=True)
    for name in ("check", "repair"):
        action = admin_sub.add_parser(name, help=f"Run bounded metadata {name} checks.")
        action.add_argument("--strict", action="store_true")
        action.add_argument("--max-work-items", type=int, default=64)
        _add_output_format(action)
        bind_command(
            action,
            handler=f"admin_{name}",
            context=ContextKind.PROJECT_WRITE if name == "repair" else ContextKind.PROJECT_READ,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.DOCTOR_VERIFY if name == "check" else OutputKind.DOCTOR_REPAIR,
        )
    clean = admin_sub.add_parser(
        "clean",
        help="Clean terminal Task metadata under bounded retention rules.",
        description="Clean retained qexp metadata while preserving experiment work directories.",
    )
    clean_scope = clean.add_mutually_exclusive_group()
    clean_scope.add_argument("--task-id")
    clean_scope.add_argument("--group")
    clean.add_argument("--older-than-days", type=int, default=30)
    clean.add_argument("--limit", type=int, default=100)
    clean.add_argument("--max-work-items", type=int, default=64)
    clean.add_argument("--dry-run", action="store_true")
    _add_output_format(clean)
    bind_command(
        clean,
        handler="admin_clean",
        context=ContextKind.PROJECT_WRITE,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.CLEAN,
    )
    operation = admin_sub.add_parser("operation", help="Inspect one durable asynchronous operation reference.")
    operation_sub = operation.add_subparsers(dest="operation_action", required=True)
    operation_show = operation_sub.add_parser("show", help="Inspect a Group, worker, or cleanup operation reference.")
    operation_show.add_argument("reference")
    _add_output_format(operation_show)
    bind_command(
        operation_show,
        handler="admin_operation_show",
        context=ContextKind.PROJECT_READ,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.OPERATION,
    )
    admin_upgrade = admin_sub.add_parser("upgrade", help="Inspect or advance machine-global Project upgrades.")
    admin_upgrade_sub = admin_upgrade.add_subparsers(dest="admin_upgrade_action", required=True)
    for name in ("status", "advance"):
        action = admin_upgrade_sub.add_parser(name, help=f"{name.capitalize()} registered Project upgrades.")
        _add_output_format(action)
        bind_command(
            action,
            handler=f"admin_upgrade_{name}",
            context=ContextKind.MACHINE,
            modes=OutputMode.FINITE,
            output_kinds={OutputKind.UPGRADE_REGISTRY_STATUS, OutputKind.UPGRADE_ADVANCE, OutputKind.UPGRADE_PROJECT},
        )
    pause = admin_upgrade_sub.add_parser("pause", help="Pause a registered Project upgrade.")
    pause.add_argument("--reason", required=True)
    _add_output_format(pause)
    bind_command(
        pause,
        handler="admin_upgrade_pause",
        context=ContextKind.REGISTERED_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.UPGRADE_PROJECT,
    )
    plan = admin_upgrade_sub.add_parser("plan", help="Plan a registered Project upgrade repair.")
    plan.add_argument("--target", required=True)
    _add_output_format(plan)
    bind_command(
        plan,
        handler="admin_upgrade_plan",
        context=ContextKind.REGISTERED_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.UPGRADE_REPAIR,
    )
    for name in ("apply", "validate"):
        action = admin_upgrade_sub.add_parser(name, help=f"{name.capitalize()} a registered Project upgrade repair.")
        action.add_argument("--repair-id", required=True)
        _add_output_format(action)
        bind_command(
            action,
            handler=f"admin_upgrade_{name}",
            context=ContextKind.REGISTERED_PROJECT,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.UPGRADE_REPAIR,
        )
    resume = admin_upgrade_sub.add_parser("resume", help="Resume a registered Project upgrade.")
    _add_output_format(resume)
    bind_command(
        resume,
        handler="admin_upgrade_resume",
        context=ContextKind.REGISTERED_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.UPGRADE_PROJECT,
    )
    migrate = admin_sub.add_parser("migrate", help="Run explicit schema and legacy-agent migrations.")
    migrate_sub = migrate.add_subparsers(dest="migrate_action", required=True)
    schema = migrate_sub.add_parser("schema", help="Convert an explicit Project to schema 6.")
    schema.add_argument("--to-schema", type=int, required=True)
    _add_output_format(schema)
    bind_command(
        schema,
        handler="admin_migrate_schema",
        context=ContextKind.EXPLICIT_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.SCHEMA6_UPGRADE,
    )
    schema6 = migrate_sub.add_parser("schema6", help="Run drained schema-6 capability activation.")
    schema6_sub = schema6.add_subparsers(dest="schema6_upgrade_action", required=True)
    for name in ("check", "start", "status"):
        action = schema6_sub.add_parser(name, help=f"Schema-6 upgrade {name}.")
        if name != "status":
            action.add_argument(
                "--capability", action="append", dest="capabilities", choices=("cpu-lane-v1", "task-dependencies-v1")
            )
        _add_output_format(action)
        bind_command(
            action,
            handler=f"admin_migrate_schema6_{name}",
            context=ContextKind.EXPLICIT_PROJECT,
            modes=OutputMode.FINITE,
            output_kinds=OutputKind.SCHEMA6_UPGRADE,
        )
    attest = schema6_sub.add_parser("attest", help="Attest stopped clients for schema-6 activation.")
    attest.add_argument("--activation-id", required=True)
    attest.add_argument("--confirm-clients-stopped", action="store_true")
    _add_output_format(attest)
    bind_command(
        attest,
        handler="admin_migrate_schema6_attest",
        context=ContextKind.EXPLICIT_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.SCHEMA6_UPGRADE,
    )
    resume_schema6 = schema6_sub.add_parser("resume", help="Resume schema-6 activation.")
    resume_schema6.add_argument("--activation-id", required=True)
    _add_output_format(resume_schema6)
    bind_command(
        resume_schema6,
        handler="admin_migrate_schema6_resume",
        context=ContextKind.EXPLICIT_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.SCHEMA6_UPGRADE,
    )
    legacy = migrate_sub.add_parser("agent", help="Migrate legacy Project agent metadata.")
    _add_output_format(legacy)
    bind_command(
        legacy,
        handler="admin_migrate_agent",
        context=ContextKind.EXPLICIT_PROJECT,
        modes=OutputMode.FINITE,
        output_kinds=OutputKind.AGENT_OPERATION,
    )

    # Add common options to every ordinary parser level so options can appear
    # before or after a command path.  Setup parsers retain their feature-owned
    # explicit options; ``use`` has its own --project selector.
    ordinary = {"status", "task", "group", "machine", "agent", "notifications", "config", "admin", "submit"}
    for name in ordinary:
        _add_common_options(commands.choices[name], include_format=True)
    _add_common_options(parser, include_format=False)
    return parser


def _parse_launch_handoff_timeout_argument(value: str) -> int | float:
    """Parse a launch timeout so malformed values become a specific ValueError."""
    text = value.strip()
    if not text:
        raise ValueError("--timeout-seconds requires a finite number from 1 through 300 seconds.")
    try:
        parsed: int | float = int(text, 10)
    except ValueError:
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError("--timeout-seconds must be a finite number from 1 through 300 seconds.") from exc
    return validate_launch_handoff_timeout_seconds(parsed)


def _gpu_limit_gpus(value: str) -> int | str:
    if value == "unlimited":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gpu-limit-gpus must be a positive integer or 'unlimited'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("gpu-limit-gpus must be a positive integer or 'unlimited'.")
    return parsed
