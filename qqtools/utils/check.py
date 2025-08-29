from typing import Iterable


def check_value_in(givens: Iterable, allows: Iterable) -> bool:
    for val in givens:
        assert val in allows, f"{val} not allowed. expect one in {allows}."
    return True
