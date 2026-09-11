"""Pure resource admission decisions."""
from __future__ import annotations
from .models import AdmissionInput

def admits(candidate: AdmissionInput) -> bool:
    """Return whether the available snapshot satisfies task demand."""
    return (
        candidate.demand.cpu <= candidate.available.cpu
        and candidate.demand.gpus <= candidate.available.gpus
        and candidate.demand.memory_bytes <= candidate.available.memory_bytes
    )
