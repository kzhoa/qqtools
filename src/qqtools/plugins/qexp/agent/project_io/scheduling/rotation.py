"""Bounded, fair traversal of a changing set of Project bindings."""

from bisect import bisect_right
from collections.abc import Collection, Iterator

from ..identity import ValidationIdentity


class BindingRotation:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.cursor: tuple[str, str] | None = None

    def candidates(self, current: Collection[ValidationIdentity], *, can_start: bool) -> Iterator[ValidationIdentity]:
        if not current:
            self.cursor = None
            return
        if not can_start:
            return
        for owner, identity in binding_window(current, self.cursor, self.limit):
            self.cursor = owner
            yield identity


def binding_window(
    current: Collection[ValidationIdentity],
    after: tuple[str, str] | None,
    limit: int,
) -> Iterator[tuple[tuple[str, str], ValidationIdentity]]:
    """Visit up to limit owners, starting after the cursor and wrapping once."""
    by_owner = {(item.project_id, item.registration_generation): item for item in current}
    owners = sorted((item.project_id, item.registration_generation) for item in current)
    start = bisect_right(owners, after) if after is not None else 0
    for offset in range(min(limit, len(owners))):
        owner = owners[(start + offset) % len(owners)]
        yield owner, by_owner[owner]
