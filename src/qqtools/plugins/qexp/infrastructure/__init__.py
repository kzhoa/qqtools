"""Operating-system adapters used by qexp orchestration layers."""

from .host import host_instance_id
from .process import (
    is_process_alive,
    is_process_group_alive,
    process_start_time_ticks,
    terminate_process_group,
)

__all__ = [
    "host_instance_id",
    "is_process_alive",
    "is_process_group_alive",
    "process_start_time_ticks",
    "terminate_process_group",
]
