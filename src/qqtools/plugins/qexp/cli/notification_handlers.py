"""CLI argument mapping for the canonical notification command service."""

from __future__ import annotations

import getpass
import sys
from typing import Any

from ..agent.context import MachineRuntime
from ..commands import notifications as notification_commands
from ..config_types import RootConfig
from ..notification_reconciliation import resolve_legacy_conflict
from .errors import CliOperationalError, CliUsageError
from .outcome import CommandOutcome
from .output import CliOutput, OutputKind


def _webhook_input(args: Any) -> str | None:
    if args.webhook is not None or args.webhook_env is not None:
        return args.webhook
    if args.webhook_stdin:
        webhook = sys.stdin.readline().rstrip("\r\n")
        if not webhook:
            raise CliUsageError("--webhook-stdin requires a non-empty first input line")
        return webhook
    if not sys.stdin.isatty():
        raise CliUsageError("Webhook input is required outside an interactive terminal; use --webhook-stdin.")
    webhook = getpass.getpass("Feishu webhook: ")
    if not webhook:
        raise CliUsageError("Feishu webhook cannot be empty.")
    return webhook


def dispatch_notifications(
    args: Any, *, cfg: RootConfig | None = None, runtime: MachineRuntime | None = None
) -> CommandOutcome:
    """Map convenience arguments into one scope-aware notification service."""
    selected_runtime = runtime or MachineRuntime(getattr(args, "machine_runtime_root", None))
    try:
        selected_runtime.require_initialized()
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliOperationalError(str(exc)) from exc
    if args.scope == "project" and cfg is None:
        raise CliUsageError("Project notification scope requires an entrypoint-resolved Project.")
    action = args.command_spec.handler.removeprefix("notifications_")
    try:
        if action == "setup":
            result = notification_commands.setup_notifications(
                selected_runtime,
                args.scope,
                cfg=cfg,
                webhook=_webhook_input(args),
                webhook_env=args.webhook_env,
                secret_env=args.secret_env,
                unsigned=args.unsigned,
            )
        elif action == "show":
            result = notification_commands.show_notifications(selected_runtime, args.scope, cfg=cfg)
        elif action == "test":
            result = notification_commands.test_notifications(selected_runtime, args.scope, cfg=cfg)
        elif action == "set":
            result = notification_commands.set_notifications(
                selected_runtime,
                args.scope,
                cfg=cfg,
                enabled=True if args.enabled else False if args.disabled else None,
                webhook_env=args.webhook_env,
                secret_env=args.secret_env,
                unsigned=args.unsigned,
                timeout_seconds=args.timeout_seconds,
            )
        elif action == "reset":
            result = notification_commands.reset_notifications(selected_runtime, args.scope, cfg=cfg)
        elif action == "resolve":
            if cfg is None:
                raise CliUsageError("Notification legacy conflicts exist only in project scope; use --scope project.")
            resolve_legacy_conflict(selected_runtime, cfg, prefer=args.prefer)
            result = notification_commands.show_notifications(selected_runtime, "project", cfg=cfg)
            result.update({"action": "resolve", "outcome": "resolved", "preferred": args.prefer})
        else:
            raise CliUsageError("Unknown notification operation.")
    except ValueError as exc:
        raise CliUsageError(str(exc)) from exc
    return CommandOutcome(0, CliOutput(OutputKind.CONFIG, result))


def dispatch_config_notifications(
    args: Any, *, cfg: RootConfig | None = None, runtime: MachineRuntime | None = None
) -> CommandOutcome:
    """Adapt generic notification configuration to the same canonical policy service."""
    selected_runtime = runtime or MachineRuntime(getattr(args, "machine_runtime_root", None))
    try:
        selected_runtime.require_initialized()
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliOperationalError(str(exc)) from exc
    scope = args.scope
    if scope == "project" and cfg is None:
        raise CliUsageError("Project notification config requires an entrypoint-resolved Project.")
    if args.command_spec.handler == "config_show":
        result = notification_commands.show_notifications(selected_runtime, scope, cfg=cfg)
    elif args.command_spec.handler == "config_reset":
        if args.provider not in (None, "feishu"):
            raise CliUsageError("Unknown notification provider; only feishu is supported.")
        result = notification_commands.reset_notifications(selected_runtime, scope, cfg=cfg)
    else:
        if args.provider not in (None, "feishu"):
            raise CliUsageError("Unknown notification provider; only feishu is supported.")
        if args.secret_env is not None and args.unset_secret_env:
            raise CliUsageError("--secret-env and --unset-secret-env are mutually exclusive")
        if args.shared_webhook is not None and args.webhook_stdin:
            raise CliUsageError("--shared-webhook and --webhook-stdin are mutually exclusive")
        if args.credential_source == "shared_file" and not args.acknowledge_shared_secret_risk:
            raise CliUsageError("Legacy shared_file input requires --acknowledge-shared-secret-risk.")
        if (args.shared_webhook is not None or args.webhook_stdin) and (
            args.credential_source != "shared_file" or not args.acknowledge_shared_secret_risk
        ):
            raise CliUsageError("Legacy shared webhook input requires shared_file and explicit risk acknowledgement.")
        if args.credential_source == "env" and args.shared_webhook is not None:
            raise CliUsageError("--shared-webhook cannot be combined with env source.")
        webhook = args.shared_webhook
        if args.webhook_stdin:
            webhook = sys.stdin.readline().rstrip("\r\n")
            if not webhook:
                raise CliUsageError("--webhook-stdin requires a non-empty first input line")
        webhook_env = args.webhook_env
        if args.credential_source == "env" and webhook_env is None:
            webhook_env = "QEXP_FEISHU_WEBHOOK"
        try:
            result = notification_commands.set_notifications(
                selected_runtime,
                scope,
                cfg=cfg,
                enabled=True if args.enabled else False if args.disabled else None,
                webhook=webhook,
                webhook_env=webhook_env,
                secret_env=args.secret_env,
                unsigned=args.unset_secret_env,
                timeout_seconds=args.timeout_seconds,
            )
        except ValueError as exc:
            raise CliUsageError(str(exc)) from exc
        if webhook is not None:
            result["note"] = (
                "Legacy shared-webhook input was converted to a private MachineRuntime credential; "
                "the old shared webhook file was not updated."
            )
    return CommandOutcome(0, CliOutput(OutputKind.CONFIG, result))
