"""Mutable copies of JSON data at the Project I/O boundary."""

from collections.abc import Mapping, Sequence
from typing import Any


def copy_json_data(value: Any) -> Any:
    """Return a detached mutable copy of protocol-frozen JSON data."""
    if isinstance(value, Mapping):
        return {key: copy_json_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [copy_json_data(item) for item in value]
    return value
