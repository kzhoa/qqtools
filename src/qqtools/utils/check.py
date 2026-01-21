"""
Naming/behavior convention:
- `is_*()` functions: Return False on failure.
- `ensure_*()` functions: Raise an exception on failure.
"""

from typing import Iterable, Optional


def check_values_allowed(givens: Iterable, allows: Iterable) -> bool:
    allows_set = set(allows)
    for val in givens:
        if val not in allows_set:
            raise ValueError(f"{val} not allowed. Expected one in {allows_set}.")
    return True


def is_alias_exists(alias_names: Iterable[str], search_targets: Iterable) -> bool:
    search_targets = set(search_targets)
    if isinstance(alias_names, str):
        alias_names = [alias_names]  # Poka-yoke
    return any(name in search_targets for name in alias_names)
