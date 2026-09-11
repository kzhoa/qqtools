"""Clock evidence serialization independent of qexp orchestration."""

import time
from typing import Any


def clock_evidence(observation: Any) -> dict[str, Any]:
    return {
        "clock_error_bound_seconds": observation.bound_at(time.monotonic()),
        "provider": observation.provider,
        "observation_id": observation.observation_id,
    }
