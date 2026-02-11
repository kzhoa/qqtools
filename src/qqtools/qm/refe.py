from collections import Counter
from typing import Dict, List

__all__ = ["calc_refe"]


def calc_refe(elements: List[int], atom_ref: Dict[int, float]) -> float:
    if not elements:
        return 0.0
    element_counts = Counter(elements)
    return sum(count * atom_ref.get(el) for el, count in element_counts.items())
