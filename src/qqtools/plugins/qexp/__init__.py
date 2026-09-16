"""qexp public package surface for schema 6.

Public names are resolved lazily so lightweight process entrypoints do not import
the entire scheduler and upgrade graph before they can start.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "clean": ("commands.cleanup", "clean"),
    "get_log_path": ("commands.logs", "get_log_path"),
    "read_logs": ("commands.logs", "read_logs"),
    "tail_log": ("commands.logs", "tail_log"),
    "batch_submit": ("commands.task", "batch_submit"),
    "cancel": ("commands.task", "cancel"),
    "keep_local": ("commands.task", "keep_local"),
    "offer": ("commands.task", "offer"),
    "retry": ("commands.task", "retry"),
    "share": ("commands.task", "share"),
    "submit": ("commands.task", "submit"),
    "RootConfig": ("config_types", "RootConfig"),
    "load_root_config": ("layout", "load_root_config"),
    "init_shared_root": ("machine_config", "init_shared_root"),
    "AGENT_MODE_DAEMON": ("models", "AGENT_MODE_DAEMON"),
    "AGENT_MODE_ON_DEMAND": ("models", "AGENT_MODE_ON_DEMAND"),
    "PHASE_BLOCKED": ("models", "PHASE_BLOCKED"),
    "PHASE_CANCELLED": ("models", "PHASE_CANCELLED"),
    "PHASE_FAILED": ("models", "PHASE_FAILED"),
    "PHASE_QUEUED": ("models", "PHASE_QUEUED"),
    "PHASE_RUNNING": ("models", "PHASE_RUNNING"),
    "PHASE_SUCCEEDED": ("models", "PHASE_SUCCEEDED"),
    "inspect_task": ("observer", "inspect_task"),
    "list_groups": ("observer", "list_groups"),
    "list_machines": ("observer", "list_machines"),
    "list_tasks": ("observer", "list_tasks"),
    "top_view": ("observer", "top_view"),
    "AttemptRecord": ("runtime.records", "AttemptRecord"),
    "TaskRecord": ("runtime.records", "TaskRecord"),
    "TaskSpec": ("runtime.records", "TaskSpec"),
    "CpuLanePolicy": ("runtime.resources.cpu_lane", "CpuLanePolicy"),
    "get_cpu_lane_policy": ("runtime.resources.cpu_lane", "get_cpu_lane_policy"),
    "set_cpu_lane_capacity": ("runtime.resources.cpu_lane", "set_cpu_lane_capacity"),
}


def __getattr__(name: str):
    """Load a public qexp export on first access."""
    if name == "Task":
        value = __getattr__("TaskRecord")
    else:
        target = _LAZY_EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(f"{__name__}.{target[0]}"), target[1])
    globals()[name] = value
    return value

__all__ = [
    "AGENT_MODE_DAEMON",
    "AGENT_MODE_ON_DEMAND",
    "AttemptRecord",
    "RootConfig",
    "Task",
    "TaskRecord",
    "TaskSpec",
    "batch_submit",
    "cancel",
    "clean",
    "get_log_path",
    "init_shared_root",
    "inspect_task",
    "list_groups",
    "list_machines",
    "list_tasks",
    "load_root_config",
    "offer",
    "read_logs",
    "CpuLanePolicy",
    "get_cpu_lane_policy",
    "set_cpu_lane_capacity",
    "retry",
    "share",
    "keep_local",
    "submit",
    "tail_log",
    "top_view",
    "PHASE_BLOCKED",
    "PHASE_CANCELLED",
    "PHASE_FAILED",
    "PHASE_QUEUED",
    "PHASE_RUNNING",
    "PHASE_SUCCEEDED",
]
