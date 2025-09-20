"""
qq:
For functions here, must obey:
- those starts with "check_", raise Error if not satisfied;
- those starts with "is_", return False if not satisfied;
"""

from typing import Iterable


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
