"""Provider protocol shared by qexp notification implementations."""
from __future__ import annotations

from typing import Any, Protocol


class Notifier(Protocol):
    name: str

    def send(self, event: Any, *, webhook: str, secret: str | None,
             timeout_seconds: float) -> dict[str, Any]:
        ...
