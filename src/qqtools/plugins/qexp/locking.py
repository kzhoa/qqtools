"""Schema-5 lock facade."""
from .runtime.locks import exclusive, group_lock, schema_lock, task_lock

__all__ = ["exclusive", "group_lock", "schema_lock", "task_lock"]
