"""Local agent activation control plane for qexp."""
from __future__ import annotations

from .agent import get_agent_status
from .agent_process import spawn_agent_process
from .events import write_event
from .machine_config import load_machine_policy
from .runtime.locks import exclusive
from .scheduler import has_eligible_local_work


def ensure_local_agent_active(cfg, *, reason: str) -> bool:
    policy = load_machine_policy(cfg)
    write_event(
        cfg,
        "agent_activation_requested",
        details={"reason": reason, "agent_mode": policy.agent_mode},
    )
    if not policy.autostart_local:
        write_event(
            cfg,
            "agent_activation_skipped",
            details={"reason": reason, "skip_reason": "manual_mode", "agent_mode": policy.agent_mode},
        )
        return False
    if get_agent_status(cfg).get("is_running"):
        write_event(
            cfg,
            "agent_activation_skipped",
            details={"reason": reason, "skip_reason": "agent_running", "agent_mode": policy.agent_mode},
        )
        return False
    if not has_eligible_local_work(cfg):
        write_event(
            cfg,
            "agent_activation_skipped",
            details={"reason": reason, "skip_reason": "no_local_eligible_work", "agent_mode": policy.agent_mode},
        )
        return False

    activation_lock = cfg.runtime_root / "locks" / "activation.lock"
    with exclusive(activation_lock, blocking=False) as acquired:
        if not acquired:
            write_event(
                cfg,
                "agent_activation_skipped",
                details={"reason": reason, "skip_reason": "activation_in_progress", "agent_mode": policy.agent_mode},
            )
            return False
        if get_agent_status(cfg).get("is_running"):
            write_event(
                cfg,
                "agent_activation_skipped",
                details={"reason": reason, "skip_reason": "agent_running", "agent_mode": policy.agent_mode},
            )
            return False
        if not has_eligible_local_work(cfg):
            write_event(
                cfg,
                "agent_activation_skipped",
                details={"reason": reason, "skip_reason": "no_local_eligible_work", "agent_mode": policy.agent_mode},
            )
            return False
        process = spawn_agent_process(cfg)
        write_event(
            cfg,
            "agent_activation_started",
            details={"reason": reason, "agent_mode": policy.agent_mode, "pid": process.pid},
        )
        return True
