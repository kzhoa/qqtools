from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.group_service_coordinator import (
    GroupServiceCoordinator,
    GroupServiceRound,
    GroupServiceTransition,
)
from qqtools.plugins.qexp.agent.group_service_transport import initial_group_service_continuation
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.working_set import BindingTurn
from qqtools.plugins.qexp.runtime.group_discovery.advance import (
    initial_control_continuation,
    initial_discovery_continuation,
)
from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state


def _turn(*, unknown: bool = False, wake_generation: int = 1) -> BindingTurn:
    return BindingTurn(
        runtime_id="runtime",
        project_id="project",
        registration_generation="generation",
        shared_root=Path("/project"),
        lane="group",
        checkpoint=("epoch", 1),
        unknown=unknown,
        wake_generation=wake_generation,
    )


def _round() -> GroupServiceRound:
    return GroupServiceRound(_turn(), initial_group_service_probe_state())


def _binding() -> ProjectBinding:
    return ProjectBinding.from_canonical_paths(
        project_id="project",
        shared_root=Path("/project"),
        machine_name="machine",
        registration_generation="generation",
        runtime_instance_id="runtime",
        runtime_root="/runtime",
    )


class _WorkingSet:
    def __init__(self, turns: list[BindingTurn] | None = None) -> None:
        self.turns = list(turns or [_turn()])
        self.current: set[BindingTurn] = set()
        self.pending: set[BindingTurn] = set()
        self.quiescent = False
        self.acknowledgements: list[tuple[BindingTurn, bool]] = []

    def begin_turn(self, _binding, _lane):
        turn = self.turns.pop(0)
        self.current.add(turn)
        return turn

    def is_current_turn(self, turn):
        return turn in self.current and not turn.unknown

    def is_turn_observation_pending(self, turn):
        return turn in self.pending

    def is_lane_quiescent(self, _binding, _lane):
        return self.quiescent

    def acknowledge(self, turn, *, quiescent):
        self.acknowledgements.append((turn, quiescent))
        if not quiescent:
            self.current.discard(turn)
        return True


class _Preparer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def prepare_closed_request(self, _binding, _revision, request_spec):
        kind = "probe" if request_spec.operation_kind == "group_service_probe" else "advance"
        request = SimpleNamespace(request_id=f"{kind}-{len(self.calls)}")
        self.calls.append((kind, request_spec.parameters))
        return request


@pytest.mark.parametrize(
    ("state", "candidate", "acknowledge"),
    [
        ("pending", None, False),
        ("active", {"group": "experiment", "lane": "membership", "generation": 3}, False),
        ("quiescent", None, True),
    ],
)
def test_probe_transition_covers_pending_active_and_quiescent(state, candidate, acknowledge):
    round_state = _round()
    round_state.retry_at = 23.0
    probe_state = {**initial_group_service_probe_state(), "offset": 7}

    transition = round_state.apply_probe({"state": state, "probe_state": probe_state, "candidate": candidate})

    assert transition == GroupServiceTransition(acknowledge_quiescent=acknowledge)
    assert round_state.probe_state == probe_state
    assert round_state.probe_state is not probe_state
    assert round_state.is_quiescent is acknowledge
    assert round_state.candidate == candidate
    assert round_state.continuation == (None if candidate is None else initial_group_service_continuation(candidate))
    assert round_state.retry_at == 0.0


def test_probe_transition_does_not_revisit_a_settled_legacy_group():
    round_state = _round()
    round_state.settled_legacy.add("experiment")

    transition = round_state.apply_probe(
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": {"group": "experiment", "lane": "legacy", "generation": None},
        }
    )

    assert transition == GroupServiceTransition()
    assert round_state.candidate is None
    assert round_state.continuation is None


@pytest.mark.parametrize(
    ("state", "expected_retry_at", "cleared"),
    [
        ("progress", 0.0, False),
        ("blocked", 18.0, False),
        ("stale", 0.0, True),
    ],
)
def test_locator_advance_transition_covers_progress_blocked_and_stale(state, expected_retry_at, cleared):
    round_state = _round()
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    continuation = initial_discovery_continuation()
    round_state.candidate = candidate
    round_state.continuation = continuation

    transition = round_state.apply_advance(
        {
            "state": state,
            "continuation": None if state == "stale" else continuation,
            "probe": None,
        },
        monotonic_time=17.0,
    )

    assert transition == GroupServiceTransition(restart_turn=True)
    assert (round_state.candidate is None) is cleared
    assert (round_state.continuation is None) is cleared
    assert round_state.retry_at == expected_retry_at


def test_advance_transition_detaches_nested_continuation_state():
    round_state = _round()
    candidate = {"group": "experiment", "lane": "control", "generation": 3}
    continuation = initial_control_continuation()
    continuation["operation_cursor"]["operation"] = {"sequence": 4}
    round_state.candidate = candidate

    transition = round_state.apply_advance(
        {"state": "progress", "continuation": continuation, "probe": None},
        monotonic_time=17.0,
    )
    continuation["operation_cursor"]["operation"]["sequence"] = 5

    assert transition == GroupServiceTransition(restart_turn=True)
    assert round_state.continuation["operation_cursor"]["operation"]["sequence"] == 4


def test_advance_without_a_retained_candidate_requests_a_fresh_turn():
    round_state = _round()

    transition = round_state.apply_advance(
        {"state": "stale", "continuation": None, "probe": None},
        monotonic_time=17.0,
    )

    assert transition == GroupServiceTransition(restart_turn=True)


def test_settled_legacy_advance_adopts_follow_up_candidate():
    round_state = _round()
    round_state.candidate = {"group": "old", "lane": "legacy", "generation": None}
    next_candidate = {"group": "next", "lane": "control", "generation": 4}
    follow_up_state = {**initial_group_service_probe_state(), "offset": 9}

    transition = round_state.apply_advance(
        {
            "state": "quiescent",
            "continuation": initial_discovery_continuation(),
            "probe": {"state": "active", "probe_state": follow_up_state, "candidate": next_candidate},
        },
        monotonic_time=4.0,
    )

    assert transition == GroupServiceTransition()
    assert round_state.settled_legacy == {"old"}
    assert round_state.probe_state == follow_up_state
    assert round_state.candidate == next_candidate
    assert round_state.continuation == initial_group_service_continuation(next_candidate)


def test_settled_legacy_follow_up_can_acknowledge_the_captured_turn():
    round_state = _round()
    round_state.candidate = {"group": "old", "lane": "legacy", "generation": None}

    transition = round_state.apply_advance(
        {
            "state": "quiescent",
            "continuation": initial_discovery_continuation(),
            "probe": {
                "state": "quiescent",
                "probe_state": initial_group_service_probe_state(),
                "candidate": None,
            },
        },
        monotonic_time=4.0,
    )

    assert transition == GroupServiceTransition(acknowledge_quiescent=True)
    assert round_state.is_quiescent
    assert round_state.candidate is None
    assert round_state.continuation is None


@pytest.mark.parametrize("unknown", [False, True])
def test_restart_replaces_only_the_captured_turn_and_probe_state(unknown):
    round_state = _round()
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    continuation = initial_discovery_continuation()
    round_state.is_quiescent = True
    round_state.candidate = candidate
    round_state.continuation = continuation
    round_state.settled_legacy.add("old")
    round_state.retry_at = 22.0
    replacement = _turn(unknown=unknown, wake_generation=2)
    probe_state = {**initial_group_service_probe_state(), "offset": 11}

    round_state.restart(replacement, probe_state)

    assert round_state.turn is replacement
    assert round_state.probe_state == probe_state
    assert round_state.probe_state is not probe_state
    assert not round_state.is_quiescent
    assert round_state.candidate is candidate
    assert round_state.continuation is continuation
    assert round_state.settled_legacy == {"old"}
    assert round_state.retry_at == 22.0


@pytest.mark.parametrize(
    ("turn", "supersede"),
    [
        (_turn(unknown=True, wake_generation=2), False),
        (_turn(wake_generation=1), True),
    ],
    ids=["unknown", "superseded"],
)
def test_controller_quiescence_rejects_unknown_and_superseded_turns(turn, supersede):
    working_set = _WorkingSet([turn])
    runtime = SimpleNamespace(root=Path("/runtime"), working_set=working_set)
    executor = SimpleNamespace(runtime_id="runtime")
    controller = ProjectIOController(runtime, executor)
    binding = _binding()
    controller._observed_executor_epoch = "epoch"
    identity = controller._binding_identity(binding, 1, "epoch")
    assert identity is not None
    offer = controller._group_service.next_offer(identity, binding, working_set)
    assert offer is not None
    request = controller._group_service.prepare_request(offer, binding, 1, _Preparer(), working_set)
    assert request is not None
    assert controller._group_service.apply_result(
        identity,
        request.request_id,
        binding,
        "group_service_probe",
        {
            "state": "quiescent",
            "probe_state": initial_group_service_probe_state(),
            "candidate": None,
        },
        working_set,
    )
    if supersede:
        working_set.current.discard(turn)

    assert not controller.group_service_is_quiescent(binding, 1)


def test_coordinator_rechecks_a_deferred_first_probe_before_preparing():
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    working_set = _WorkingSet()
    preparer = _Preparer()
    binding = _binding()

    stale_offer = coordinator.next_offer("old-identity", binding, working_set)
    assert stale_offer is not None and stale_offer.operation_kind == "group_service_probe"
    coordinator.reconcile({}, (), working_set)

    assert coordinator.prepare_request(stale_offer, binding, 1, preparer, working_set) is None
    assert preparer.calls == []


def test_coordinator_rejects_an_offer_after_its_existing_round_is_replaced():
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    first_turn = _turn(wake_generation=1)
    replacement_turn = _turn(wake_generation=2)
    working_set = _WorkingSet([first_turn, replacement_turn])
    preparer = _Preparer()
    binding = _binding()
    identity = "identity"

    probe_offer = coordinator.next_offer(identity, binding, working_set)
    assert probe_offer is not None
    probe = coordinator.prepare_request(probe_offer, binding, 1, preparer, working_set)
    assert probe is not None
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    assert coordinator.apply_result(
        identity,
        probe.request_id,
        binding,
        "group_service_probe",
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": candidate,
        },
        working_set,
    )
    stale_advance = coordinator.next_offer(identity, binding, working_set)
    assert stale_advance is not None

    working_set.current.clear()
    coordinator.reconcile({}, (), working_set)
    coordinator.reconcile({identity: binding}, (), working_set)
    replacement_offer = coordinator.next_offer(identity, binding, working_set)
    assert replacement_offer is not None
    replacement = coordinator.prepare_request(replacement_offer, binding, 1, preparer, working_set)
    assert replacement is not None

    call_count = len(preparer.calls)
    assert coordinator.prepare_request(stale_advance, binding, 1, preparer, working_set) is None
    assert len(preparer.calls) == call_count


def test_failed_first_preparation_does_not_create_a_group_turn():
    class FailingPreparer(_Preparer):
        def prepare_closed_request(self, _binding, _revision, request_spec):
            raise OSError("preparation failed")

    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    turn = _turn()
    working_set = _WorkingSet([turn])
    binding = _binding()
    offer = coordinator.next_offer("identity", binding, working_set)
    assert offer is not None

    with pytest.raises(OSError, match="preparation failed"):
        coordinator.prepare_request(offer, binding, 1, FailingPreparer(), working_set)

    assert working_set.turns == [turn]
    retry = coordinator.next_offer("identity", binding, working_set)
    assert retry is not None and retry.operation_kind == "group_service_probe"


def test_coordinator_owns_request_association_and_group_progress():
    now = [10.0]
    coordinator = GroupServiceCoordinator(monotonic=lambda: now[0])
    first_turn = _turn(wake_generation=1)
    second_turn = _turn(wake_generation=2)
    working_set = _WorkingSet([first_turn, second_turn])
    preparer = _Preparer()
    binding = _binding()
    identity = "identity"

    probe_offer = coordinator.next_offer(identity, binding, working_set)
    assert probe_offer is not None and probe_offer.operation_kind == "group_service_probe"
    probe = coordinator.prepare_request(probe_offer, binding, 1, preparer, working_set)
    assert probe is not None
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    assert coordinator.apply_result(
        identity,
        probe.request_id,
        binding,
        "group_service_probe",
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": candidate,
        },
        working_set,
    )

    advance_offer = coordinator.next_offer(identity, binding, working_set)
    assert advance_offer is not None and advance_offer.operation_kind == "group_service_advance"
    advance = coordinator.prepare_request(advance_offer, binding, 1, preparer, working_set)
    assert advance is not None
    continuation = initial_discovery_continuation()
    assert coordinator.apply_result(
        identity,
        advance.request_id,
        binding,
        "group_service_advance",
        {"state": "blocked", "continuation": continuation, "probe": None},
        working_set,
    )

    assert working_set.acknowledgements == [(first_turn, False)]
    assert coordinator.next_offer(identity, binding, working_set) is None
    now[0] = 11.0
    retry = coordinator.next_offer(identity, binding, working_set)
    assert retry is not None and retry.operation_kind == "group_service_advance"
    assert [kind for kind, _payload in preparer.calls] == ["probe", "advance"]


def test_coordinator_rejects_a_late_result_after_binding_replacement():
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    working_set = _WorkingSet()
    preparer = _Preparer()
    binding = _binding()
    identity = "old-identity"

    offer = coordinator.next_offer(identity, binding, working_set)
    assert offer is not None
    request = coordinator.prepare_request(offer, binding, 1, preparer, working_set)
    assert request is not None
    coordinator.reconcile({"replacement": binding}, {request.request_id}, working_set)

    assert not coordinator.apply_result(
        identity,
        request.request_id,
        binding,
        "group_service_probe",
        {
            "state": "quiescent",
            "probe_state": initial_group_service_probe_state(),
            "candidate": None,
        },
        working_set,
    )
    assert working_set.acknowledgements == []
    assert not coordinator.is_quiescent(identity, working_set)


def test_coordinator_retains_unknown_turn_until_observation_then_reproves():
    unknown_turn = _turn(unknown=True, wake_generation=1)
    current_turn = _turn(wake_generation=2)
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    working_set = _WorkingSet([unknown_turn, current_turn])
    preparer = _Preparer()
    binding = _binding()
    identity = "identity"

    offer = coordinator.next_offer(identity, binding, working_set)
    assert offer is not None
    request = coordinator.prepare_request(offer, binding, 1, preparer, working_set)
    assert request is not None
    working_set.current.discard(unknown_turn)
    working_set.pending.add(unknown_turn)
    assert coordinator.apply_result(
        identity,
        request.request_id,
        binding,
        "group_service_probe",
        {
            "state": "quiescent",
            "probe_state": initial_group_service_probe_state(),
            "candidate": None,
        },
        working_set,
    )

    assert coordinator.next_offer(identity, binding, working_set) is None
    working_set.pending.clear()
    retry = coordinator.next_offer(identity, binding, working_set)

    assert retry is not None and retry.operation_kind == "group_service_probe"
    assert working_set.acknowledgements == []


def test_coordinator_failure_restarts_only_an_exact_current_advance():
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    first_turn = _turn(wake_generation=1)
    second_turn = _turn(wake_generation=2)
    working_set = _WorkingSet([first_turn, second_turn])
    preparer = _Preparer()
    binding = _binding()
    identity = "identity"

    probe_offer = coordinator.next_offer(identity, binding, working_set)
    assert probe_offer is not None
    probe = coordinator.prepare_request(probe_offer, binding, 1, preparer, working_set)
    assert probe is not None
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    assert coordinator.apply_result(
        identity,
        probe.request_id,
        binding,
        "group_service_probe",
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": candidate,
        },
        working_set,
    )
    advance_offer = coordinator.next_offer(identity, binding, working_set)
    assert advance_offer is not None
    advance = coordinator.prepare_request(advance_offer, binding, 1, preparer, working_set)
    assert advance is not None

    assert coordinator.apply_failure(
        identity,
        advance.request_id,
        binding,
        "group_service_advance",
        working_set,
    )
    assert working_set.acknowledgements == [(first_turn, False)]
    assert not coordinator.apply_failure(
        identity,
        advance.request_id,
        binding,
        "group_service_advance",
        working_set,
    )


def test_superseded_advance_failure_preserves_the_replacement_round():
    coordinator = GroupServiceCoordinator(monotonic=lambda: 0.0)
    turns = [_turn(wake_generation=1), _turn(wake_generation=2)]
    working_set = _WorkingSet(turns)
    preparer = _Preparer()
    binding = _binding()
    identity = "identity"

    probe_offer = coordinator.next_offer(identity, binding, working_set)
    assert probe_offer is not None
    probe = coordinator.prepare_request(probe_offer, binding, 1, preparer, working_set)
    assert probe is not None
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    assert coordinator.apply_result(
        identity,
        probe.request_id,
        binding,
        "group_service_probe",
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": candidate,
        },
        working_set,
    )
    advance_offer = coordinator.next_offer(identity, binding, working_set)
    assert advance_offer is not None
    advance = coordinator.prepare_request(advance_offer, binding, 1, preparer, working_set)
    assert advance is not None

    working_set.current.clear()
    coordinator.reconcile({}, {advance.request_id}, working_set)
    coordinator.reconcile({identity: binding}, {advance.request_id}, working_set)
    replacement_offer = coordinator.next_offer(identity, binding, working_set)
    assert replacement_offer is not None
    replacement = coordinator.prepare_request(replacement_offer, binding, 1, preparer, working_set)
    assert replacement is not None

    assert not coordinator.apply_failure(
        identity,
        advance.request_id,
        binding,
        "group_service_advance",
        working_set,
    )
    assert coordinator.apply_result(
        identity,
        replacement.request_id,
        binding,
        "group_service_probe",
        {
            "state": "quiescent",
            "probe_state": initial_group_service_probe_state(),
            "candidate": None,
        },
        working_set,
    )
    assert coordinator.is_quiescent(identity, working_set)
