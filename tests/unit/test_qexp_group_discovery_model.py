"""BFS checks for the finite Group-removal publication/completion model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import pytest

from tests.helpers.qexp.group_discovery_model import INITIAL_STATE, GroupRemovalState, successors

StateKey = tuple[GroupRemovalState, frozenset[int]]


@dataclass(frozen=True, slots=True)
class TraceNode:
    state: GroupRemovalState
    effected_generations: frozenset[int]
    predecessor: StateKey | None
    action: str | None


@dataclass(frozen=True, slots=True)
class Search:
    nodes: dict[StateKey, TraceNode]
    initial: StateKey

    @property
    def visited_count(self) -> int:
        return len(self.nodes)

    def path(self, key: StateKey) -> tuple[str, ...]:
        actions: list[str] = []
        while self.nodes[key].predecessor is not None:
            node = self.nodes[key]
            assert node.action is not None
            actions.append(node.action)
            key = node.predecessor
        return tuple(reversed(actions))


def explore(**flags: bool) -> Search:
    initial: StateKey = (INITIAL_STATE, frozenset())
    nodes = {initial: TraceNode(INITIAL_STATE, frozenset(), None, None)}
    queue: deque[StateKey] = deque([initial])
    while queue:
        key = queue.popleft()
        node = nodes[key]
        for action, next_state in successors(node.state, **flags):
            effected = node.effected_generations
            if action == "converge":
                effected = effected | {next_state.generation}
            next_key = (next_state, effected)
            if next_key not in nodes:
                nodes[next_key] = TraceNode(next_state, effected, key, action)
                queue.append(next_key)
    return Search(nodes, initial)


def find(search: Search, predicate: Callable[[TraceNode], bool]) -> tuple[StateKey, TraceNode] | None:
    return next(((key, node) for key, node in search.nodes.items() if predicate(node)), None)


def find_path(search: Search, expected: tuple[str, ...]) -> tuple[StateKey, TraceNode] | None:
    return next(((key, node) for key, node in search.nodes.items() if search.path(key) == expected), None)


def require(
    search: Search,
    predicate: Callable[[TraceNode], bool],
    description: str,
) -> tuple[StateKey, TraceNode]:
    found = find(search, predicate)
    assert found is not None, f"{description}; visited={search.visited_count}"
    return found


def trace(search: Search, key: StateKey) -> str:
    node = search.nodes[key]
    return f"visited={search.visited_count}, path={' -> '.join(search.path(key))}, state={node.state!r}"


def unsafe_completed(node: TraceNode) -> bool:
    state = node.state
    witnesses = state.outcome_in_task | state.outcome_in_cleanup | state.counted
    retained = state.outcome_in_task | state.outcome_in_cleanup
    return state.completed and (
        state.phase == "queued"
        or state.obligation in {"in_flight", "effected"}
        or not node.effected_generations <= witnesses
        or not retained <= state.counted
    )


def test_correct_model_bfs_completes_only_safe_states() -> None:
    search = explore()
    complete_nodes = [(key, node) for key, node in search.nodes.items() if node.state.completed]
    assert complete_nodes, f"no completed state; visited={search.visited_count}"
    for key, node in complete_nodes:
        assert not unsafe_completed(node), trace(search, key)
        assert not successors(node.state), trace(search, key)


def test_correct_model_reaches_reviewed_scenarios() -> None:
    search = explore()
    expected_paths = (
        ("publish", "abort", "recheck", "complete"),
        (
            "publish",
            "mutation_effect",
            "resolve_effect",
            "converge",
            "cleanup_handoff",
            "cleanup_delete",
            "recheck",
            "complete",
        ),
        ("publish", "abort", "recheck", "publish"),
    )
    for expected in expected_paths:
        found = find_path(search, expected)
        assert found is not None, f"{expected!r}; visited={search.visited_count}"
        key, node = found
        assert len(search.path(key)) == len(expected), trace(search, key)
        if expected[-1] == "publish":
            assert node.state.generation == 2, trace(search, key)
            assert node.state.obligation == "in_flight", trace(search, key)
            assert node.state.consumed_generation == 1, trace(search, key)
    for key, node in search.nodes.items():
        if node.state.generation == 2:
            assert "publish" not in dict(successors(node.state)), trace(search, key)


def test_correct_model_blocks_completion_across_unacknowledged_windows() -> None:
    search = explore()
    before_effect_found = find_path(search, ("publish",))
    assert before_effect_found is not None, f"crash before effect; visited={search.visited_count}"
    before_effect_key, _ = before_effect_found
    before_effect = search.nodes[before_effect_key]
    assert before_effect.state.obligation == "in_flight"
    assert "complete" not in {action for action, _ in successors(before_effect.state)}

    before_receipt_path = ("publish", "mutation_effect", "resolve_effect", "converge")
    before_receipt_found = find_path(search, before_receipt_path)
    assert before_receipt_found is not None, f"effect before receipt; visited={search.visited_count}"
    before_receipt_key, _ = before_receipt_found
    before_receipt = search.nodes[before_receipt_key]
    assert before_receipt.state.receipt_generation == 0
    assert "complete" not in {action for action, _ in successors(before_receipt.state)}
    with_receipt = dict(successors(before_receipt.state))["recheck"]
    assert "complete" in {action for action, _ in successors(with_receipt)}


@pytest.mark.parametrize(
    ("flags", "predicate", "description"),
    [
        (
            {"allow_inflight_ack": True},
            lambda node: node.state.completed and node.state.obligation in {"in_flight", "effected"},
            "in-flight acknowledgement defect",
        ),
        (
            {"allow_unwitnessed_delete": True},
            lambda node: (
                node.state.completed
                and bool(
                    node.effected_generations
                    - (node.state.outcome_in_task | node.state.outcome_in_cleanup | node.state.counted)
                )
            ),
            "unwitnessed deletion defect",
        ),
        (
            {"reuse_aborted_ticket": True},
            lambda node: node.state.completed and node.state.phase == "queued" and node.state.obligation == "effected",
            "aborted-ticket reuse defect",
        ),
    ],
)
def test_defect_modes_have_bfs_counterexamples(flags: dict[str, bool], predicate, description: str) -> None:
    search = explore(**flags)
    key, node = require(search, predicate, description)
    assert unsafe_completed(node), trace(search, key)
