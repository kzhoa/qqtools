from __future__ import annotations

from functools import partial

import pytest

from qqtools.plugins.qexp.agent.project_io_admission import ProjectIOAdmission
from qqtools.plugins.qexp.agent.project_io_arbiter import ServiceIntent


def _intent(project: str, service: str = "primary", *, epoch: str = "epoch-1") -> ServiceIntent:
    return ServiceIntent(("runtime", project, "generation"), service, "validate_binding", (epoch,))


def _offer_all(admission, intents, executed):
    for item in intents:
        admission.offer(item, partial(executed.append, item))
        assert admission.pending_count <= 64


def test_initial_recovery_admission_precedes_background_without_starving_other_families():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    recovery = ServiceIntent(owner, "background", "recovery_admission", ())
    ordinary = [
        ServiceIntent(owner, "background", kind, ())
        for kind in ("activation_observe", "notification_service", "submission_control_service")
    ]
    executed = []
    for _ in range(8):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [*ordinary, recovery], executed)
        admission.finish()
    expected = [recovery, ordinary[0], ordinary[2], ordinary[1]]
    assert executed == expected * 2


def test_capture_and_upgrade_prerequisites_do_not_starve_periodic_metadata():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    kinds = [
        "recovery_admission",
        "upgrade_service",
        "activation_consumer_register",
        "activation_observe",
        "scheduler_ready_index_build",
        "machine_snapshot_publish",
        "submission_control_service",
    ]
    intents = [ServiceIntent(owner, "background", kind, ()) for kind in kinds]
    executed = []
    for _ in range(len(intents) * 2):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, list(reversed(intents)), executed)
        admission.finish()
    expected = [*intents[:5], intents[6], intents[5]]
    assert executed == expected * 2


def test_progress_publish_continuation_runs_once_before_ordinary_background_rotation():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    observe = ServiceIntent(owner, "background", "progress_projection", ("epoch-1", "progress", "attempt-1", "observe"))
    publish = ServiceIntent(owner, "background", "progress_projection", ("epoch-1", "progress", "attempt-1", "publish"))
    ordinary = ServiceIntent(owner, "background", "notification_service", ("epoch-1",))
    executed = []

    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [ordinary, observe], executed)
    admission.finish()
    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [ordinary, publish], executed)
    admission.finish()
    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [ordinary, observe], executed)
    admission.finish()

    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [ordinary, observe], executed)
    admission.finish()

    assert executed == [observe, publish, ordinary, observe]


def test_cursor_commit_then_finite_quiescence_and_discovery_precede_borrow_refresh():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    primary = ServiceIntent(owner, "primary", "scheduler_observe", (), "gpu_primary")
    quiescence = ServiceIntent(owner, "primary", "scheduler_quiescence_probe", ())
    borrow = ServiceIntent(owner, "primary", "scheduler_observe", (), "gpu_borrow")
    periodic = ServiceIntent(owner, "primary", "scheduler_cursor_commit", (), "gpu_primary")
    intents = [periodic, borrow, quiescence, primary]
    executed = []
    for _ in range(len(intents) * 2):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, intents, executed)
        admission.finish()
    assert executed == [periodic, quiescence, primary, borrow] * 2


def test_due_authority_override_preserves_ordinary_family_rotation():
    now = [10.0]
    admission = ProjectIOAdmission(monotonic=lambda: now[0])
    owner = _intent("a").owner
    due = ServiceIntent(owner, "authority", "registration_renew", (), deadline=9.0)
    ordinary = [
        ServiceIntent(owner, "authority", kind, ())
        for kind in ("activation_consumer_ack", "authority_running_publish", "authority_terminal_publish")
    ]
    executed = []
    for _ in range(6):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [*ordinary, due], executed)
        admission.finish()
    assert executed[::2] == [due] * 3
    assert executed[1::2] == ordinary


def test_future_deadline_does_not_override_ready_family_and_blocked_does_not_spend_override():
    now = [10.0]
    admission = ProjectIOAdmission(monotonic=lambda: now[0])
    owner = _intent("a").owner
    future = ServiceIntent(owner, "authority", "registration_renew", (), deadline=11.0)
    ordinary = ServiceIntent(owner, "authority", "authority_running_publish", ())
    executed = []
    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [future, ordinary], executed)
    admission.finish()
    assert executed == [ordinary]
    now[0] = 12.0
    admission.begin([owner], blocked=[owner], free_slots=0)
    _offer_all(admission, [future, ordinary], executed)
    admission.finish()
    assert executed == [ordinary]
    admission.begin([owner], blocked=(), free_slots=1)
    _offer_all(admission, [future, ordinary], executed)
    admission.finish()
    assert executed == [ordinary, future]


def test_due_authority_backlog_preserves_one_ordinary_class_slot():
    admission = ProjectIOAdmission(monotonic=lambda: 10.0)
    due = [
        ServiceIntent(_intent(f"renew-{index}").owner, "authority", "registration_renew", (), deadline=9.0)
        for index in range(4)
    ]
    ordinary = [_intent("primary"), _intent("background", "background")]
    executed = []

    admission.begin([intent.owner for intent in [*due, *ordinary]], blocked=(), free_slots=4)
    _offer_all(admission, [*ordinary, *due], executed)
    admission.finish()

    assert executed == [*due[:3], ordinary[0]]


def test_newly_blocked_owner_cannot_hide_healthy_work_in_remaining_slots():
    admission = ProjectIOAdmission(monotonic=lambda: 10.0)
    owner = _intent("mixed").owner
    due = ServiceIntent(owner, "authority", "registration_renew", (), deadline=9.0)
    critical = ServiceIntent(owner, "primary", "scheduler_claim", ("epoch", "attempt"), "gpu_primary")
    healthy_background = [
        ServiceIntent(_intent(project).owner, "background", "notification_service", ())
        for project in ("healthy-a", "healthy-b")
    ]
    intents = [due, critical, *healthy_background]
    executed = []

    admission.begin([intent.owner for intent in intents], blocked=(), free_slots=4)
    _offer_all(admission, intents, executed)
    admission.finish()

    assert executed == [due, *healthy_background]


@pytest.mark.parametrize("free_slots", [1, 2])
def test_continuous_due_authority_backlog_yields_across_passes(free_slots):
    admission = ProjectIOAdmission(monotonic=lambda: 10.0)
    due = [
        ServiceIntent(_intent(f"renew-{index}").owner, "authority", "registration_renew", (), deadline=9.0)
        for index in range(3)
    ]
    primary = _intent("primary")
    background = _intent("background", "background")
    intents = [*due, primary, background]
    executed = []

    for _ in range(20):
        admission.begin([intent.owner for intent in intents], blocked=(), free_slots=free_slots)
        _offer_all(admission, intents, executed)
        admission.finish()

    assert primary in executed
    assert background in executed
    authority_run = 0
    longest_authority_run = 0
    for intent in executed:
        if intent.service_class == "authority":
            authority_run += 1
            longest_authority_run = max(longest_authority_run, authority_run)
        else:
            authority_run = 0
    assert longest_authority_run == 3


def test_due_authority_burst_cannot_phase_lock_background_behind_primary():
    admission = ProjectIOAdmission(monotonic=lambda: 10.0)
    owners = [_intent(f"mixed-{index}").owner for index in range(3)]
    due = [ServiceIntent(owner, "authority", "registration_renew", (), deadline=9.0) for owner in owners]
    critical = [
        ServiceIntent(owner, "primary", "scheduler_claim", ("epoch", f"attempt-{index}"), "gpu_primary")
        for index, owner in enumerate(owners)
    ]
    ordinary_primary = [ServiceIntent(owner, "primary", "scheduler_observe", (), "cpu_primary") for owner in owners]
    background = [ServiceIntent(owner, "background", "notification_service", ()) for owner in owners]
    intents = [*due, *critical, *ordinary_primary, *background]
    executed = []

    for _ in range(60):
        admission.begin(owners, blocked=(), free_slots=3)
        _offer_all(admission, intents, executed)
        admission.finish()

    classes = {intent.service_class for intent in executed}
    assert classes == {"authority", "primary", "background"}
    assert {intent.owner for intent in executed if intent.service_class == "background"} == set(owners)
    background_positions = [index for index, intent in enumerate(executed) if intent.service_class == "background"]
    assert max(right - left for left, right in zip(background_positions, background_positions[1:], strict=False)) <= 18


@pytest.mark.parametrize("deadline", [True, -1.0, float("nan"), float("inf"), "10"])
def test_invalid_authority_deadline_is_not_a_service_intent(deadline):
    with pytest.raises(ValueError, match="deadline"):
        ServiceIntent(_intent("a").owner, "authority", "registration_renew", (), deadline=deadline)


def test_actions_execute_only_at_same_turn_finish():
    admission = ProjectIOAdmission()
    intent = _intent("a")
    executed = []
    admission.begin([intent.owner], blocked=(), free_slots=4)
    admission.offer(intent, partial(executed.append, intent))
    admission.offer(intent, partial(executed.append, "duplicate"))
    assert executed == []
    assert admission.active
    assert admission.pending_count == 1
    admission.finish()
    assert executed == [intent]
    assert not admission.active
    assert admission.pending_count == 0
    assert not admission.has_deferred_work


@pytest.mark.parametrize(
    "operation_kind",
    ["scheduler_claim", "scheduler_launch_authorize", "scheduler_reservation_reconcile"],
)
def test_prepared_critical_chain_precedes_discovery_for_three_grants_then_yields(operation_kind):
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", operation_kind, ("epoch-1",), "gpu_primary")
    discovery = ServiceIntent(owner, "primary", "scheduler_observe", ("epoch-1",), "gpu_borrow")
    background = ServiceIntent(owner, "background", "notification_service", ("epoch-1",))
    executed = []
    for _ in range(6):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [discovery, background, critical], executed)
        admission.finish()
    assert executed == [critical, critical, critical, background, discovery, critical]


def test_critical_chain_does_not_replace_authority_opportunities():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", "scheduler_launch_authorize", ("epoch-1",))
    authority = ServiceIntent(owner, "authority", "registration_renew", ("epoch-1",), deadline=0.0)
    background = ServiceIntent(owner, "background", "observation_service", ("epoch-1",))
    executed = []
    for _ in range(10):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [background, critical, authority], executed)
        admission.finish()
    assert executed[0] == authority
    assert executed.count(critical) == 3
    assert executed.count(authority) == 6
    assert executed[-1] == background


def test_failing_critical_actions_spend_the_bounded_preference():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", "scheduler_launch_authorize", ("epoch-1",))
    background = ServiceIntent(owner, "background", "notification_service", ("epoch-1",))
    executed = []

    def fail():
        executed.append(critical)
        raise OSError("preparation failed")

    for _ in range(3):
        admission.begin([owner], blocked=(), free_slots=1)
        admission.offer(background, partial(executed.append, background))
        admission.offer(critical, fail)
        with pytest.raises(OSError, match="preparation failed"):
            admission.finish()
    admission.begin([owner], blocked=(), free_slots=1)
    admission.offer(background, partial(executed.append, background))
    admission.offer(critical, fail)
    admission.finish()
    assert executed == [critical, critical, critical, background]


@pytest.mark.parametrize("transition", ["epoch", "replacement"])
def test_critical_credit_does_not_survive_epoch_or_binding_replacement(transition):
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", "scheduler_claim", ("epoch-1",), "gpu_primary")
    executed = []
    for _ in range(3):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [critical], executed)
        admission.finish()
    new_owner = owner if transition == "epoch" else (*owner[:2], "new-generation")
    new_critical = ServiceIntent(new_owner, "primary", "scheduler_claim", ("epoch-2",), "gpu_primary")
    background = ServiceIntent(new_owner, "background", "notification_service", ("epoch-2",))
    admission.begin([new_owner], blocked=(), free_slots=1)
    _offer_all(admission, [background, new_critical], executed)
    admission.finish()
    assert executed[-1] == new_critical
    if transition == "replacement":
        assert owner not in admission._critical_chain_grants


def test_blocked_critical_intent_does_not_spend_preference_or_block_a_peer():
    admission = ProjectIOAdmission()
    blocked_owner = _intent("blocked").owner
    healthy = _intent("healthy", "background")
    critical = ServiceIntent(blocked_owner, "primary", "scheduler_claim", ("epoch-1",), "gpu_primary")
    executed = []
    admission.begin([blocked_owner, healthy.owner], blocked=[blocked_owner], free_slots=1)
    _offer_all(admission, [critical, healthy], executed)
    admission.finish()
    assert executed == [healthy]
    assert admission._critical_chain_grants == {}


def test_critical_burst_yield_does_not_idle_when_only_critical_work_remains():
    admission = ProjectIOAdmission()
    owners = [_intent(f"owner-{index}").owner for index in range(4)]
    critical = [ServiceIntent(owner, "primary", "scheduler_claim", ("epoch-1",), "gpu_primary") for owner in owners]
    executed = []

    admission.begin(owners, blocked=(), free_slots=4)
    _offer_all(admission, critical, executed)
    admission.finish()

    assert executed == critical


def test_spent_critical_chain_uses_normal_primary_rotation_without_background():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", "scheduler_launch_authorize", ("epoch", "attempt"), "cpu_primary")
    ordinary = ServiceIntent(owner, "primary", "scheduler_observe", ("epoch", "observe"), "cpu_primary")
    executed = []

    for _ in range(3):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [ordinary, critical], executed)
        admission.finish()
    for _ in range(6):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [ordinary, critical], executed)
        admission.finish()

    assert executed[:3] == [critical, critical, critical]
    assert ordinary in executed[3:]
    assert critical in executed[3:]


def test_spent_critical_chain_uses_normal_class_cycle_without_ordinary_primary():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    critical = ServiceIntent(owner, "primary", "scheduler_launch_authorize", ("epoch", "attempt"), "cpu_primary")
    background = ServiceIntent(owner, "background", "notification_service", ("epoch", "notify"))
    executed = []

    for _ in range(9):
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, [background, critical], executed)
        admission.finish()

    assert executed[:3] == [critical, critical, critical]
    assert background in executed[3:]
    assert critical in executed[3:]


def test_critical_flood_serves_discovery_and_background_beyond_the_candidate_window():
    admission = ProjectIOAdmission()
    blocked = [_intent(name).owner for name in ("hung-a", "hung-b")]
    owners = [_intent(f"healthy-{index:03d}").owner for index in range(65)]
    executed = []
    all_intents = [
        intent
        for owner in [*blocked, *owners]
        for intent in (
            ServiceIntent(owner, "primary", "scheduler_claim", ("epoch-1",), "gpu_primary"),
            ServiceIntent(owner, "primary", "scheduler_observe", ("epoch-1",), "cpu_primary"),
            ServiceIntent(owner, "background", "notification_service", ("epoch-1",)),
        )
    ]
    for _ in range(400):
        previous_count = len(executed)
        admission.begin([*blocked, *owners], blocked=blocked, free_slots=2)
        _offer_all(admission, all_intents, executed)
        admission.finish()
        granted = executed[previous_count:]
        assert len(granted) <= 2
        assert len({intent.owner for intent in granted}) == len(granted)
        assert not any(intent.owner in blocked for intent in granted)
        served = {(intent.owner, intent.operation_kind) for intent in executed}
        if all(
            (owner, kind) in served
            for owner in owners
            for kind in ("scheduler_claim", "scheduler_observe", "notification_service")
        ):
            break
    else:
        raise AssertionError("continuous critical work hid an incumbent discovery or background family")


@pytest.mark.parametrize("service", ["authority", "primary", "background"])
def test_two_blocked_bindings_leave_both_slots_available_to_healthy_work(service):
    admission = ProjectIOAdmission()
    intents = [_intent(name, service) for name in ("hung-a", "hung-b", "healthy-a", "healthy-b")]
    owners = [item.owner for item in intents]
    executed = []
    admission.begin(owners, blocked=owners[:2], free_slots=2)
    _offer_all(admission, intents, executed)
    assert admission.pending_count == 4
    admission.finish()
    assert executed == intents[2:]
    assert not admission.has_deferred_work


def test_unselected_actions_are_not_retained_or_replayed_next_turn():
    admission = ProjectIOAdmission()
    old, new = _intent("a"), _intent("a", epoch="epoch-2")
    executed = []
    admission.begin([old.owner], blocked=(), free_slots=0)
    admission.offer(old, partial(executed.append, old))
    admission.finish()
    assert executed == []
    assert admission.pending_count == 0
    assert admission.has_deferred_work
    admission.begin([new.owner], blocked=(), free_slots=4)
    admission.offer(new, partial(executed.append, new))
    admission.finish()
    assert executed == [new]
    assert not admission.has_deferred_work


def test_consumed_owner_can_offer_successor_in_same_turn():
    admission = ProjectIOAdmission()
    intent = _intent("a")
    executed = []
    admission.begin([intent.owner], blocked=[intent.owner], free_slots=0)
    admission.refresh_capacity(blocked=(), free_slots=1)
    admission.offer(intent, partial(executed.append, intent))
    admission.finish()
    assert executed == [intent]


def test_later_result_consumption_preserves_earlier_family_rotation():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    primary = ServiceIntent(owner, "primary", "scheduler_observe", (), "gpu_primary")
    borrow = ServiceIntent(owner, "primary", "scheduler_observe", (), "gpu_borrow")
    executed = []
    for _ in range(4):
        admission.begin([owner], blocked=[owner], free_slots=0)
        admission.offer(primary, partial(executed.append, primary))
        # Borrow consumes the previous request after primary has advertised its
        # interest. Releasing occupancy must not erase that earlier interest.
        admission.refresh_capacity(blocked=(), free_slots=1)
        admission.offer(borrow, partial(executed.append, borrow))
        admission.finish()
    assert executed.count(primary) == 2
    assert executed.count(borrow) == 2


def test_refreshed_occupancy_blocks_already_collected_action():
    admission = ProjectIOAdmission()
    intent = _intent("a")
    executed = []
    admission.begin([intent.owner], blocked=(), free_slots=1)
    admission.offer(intent, partial(executed.append, intent))
    admission.refresh_capacity(blocked=[intent.owner], free_slots=0)
    admission.finish()
    assert executed == []


def test_one_binding_gets_at_most_one_action_across_classes_per_turn():
    admission = ProjectIOAdmission()
    intents = [_intent("a", service) for service in ("authority", "primary", "background")]
    classes = []
    for _ in range(8):
        executed = []
        admission.begin([intents[0].owner], blocked=(), free_slots=4)
        _offer_all(admission, intents, executed)
        admission.finish()
        assert len(executed) == 1
        classes.append(executed[0].service_class)
        assert admission.has_deferred_work
    assert classes == ["authority", "primary", "authority", "background"] * 2


@pytest.mark.parametrize(
    ("service", "families"),
    [
        ("primary", [("scheduler_observe", "gpu_primary"), ("scheduler_observe", "cpu_primary")]),
        (
            "background",
            [
                ("activation_observe", "default"),
                ("machine_snapshot_publish", "default"),
                ("maintenance_flush_event", "default"),
            ],
        ),
        (
            "authority",
            [("authority_service", "default"), ("authority_renewal", "default"), ("registration_renew", "default")],
        ),
    ],
)
def test_same_owner_families_rotate_despite_fixed_offer_order(service, families):
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    intents = [ServiceIntent(owner, service, operation, (), family) for operation, family in families]
    executed = []
    for _ in range(len(intents) * 3):
        # An occupied turn must not advance the family cursor independently of
        # actual selection and phase-lock with the producer's completion cycle.
        admission.begin([owner], blocked=[owner], free_slots=3)
        _offer_all(admission, intents, executed)
        admission.finish()
        admission.begin([owner], blocked=(), free_slots=4)
        _offer_all(admission, intents, executed)
        assert admission.pending_count == 1
        admission.finish()
        assert admission.has_deferred_work
    assert set(executed) == set(intents)
    assert all(executed.count(intent) == 3 for intent in intents)


def test_failed_family_rotates_to_other_work_for_same_owner():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    first = ServiceIntent(owner, "background", "activation_observe", ())
    second = ServiceIntent(owner, "background", "machine_snapshot_publish", ())
    executed = []

    def fail():
        raise OSError("failed family")

    admission.begin([owner], blocked=(), free_slots=4)
    admission.offer(first, fail)
    admission.offer(second, partial(executed.append, second))
    with pytest.raises(OSError, match="failed family"):
        admission.finish()
    admission.begin([owner], blocked=(), free_slots=4)
    admission.offer(first, fail)
    admission.offer(second, partial(executed.append, second))
    admission.finish()
    assert executed == [second]


@pytest.mark.parametrize("family", [None, "custom", [], 1])
def test_work_families_are_closed_to_bound_cursor_metadata(family):
    with pytest.raises(ValueError, match="work_family"):
        ServiceIntent(_intent("a").owner, "primary", "scheduler_observe", (), family)


def test_bounded_collection_eventually_services_all_65_bindings_and_classes():
    admission = ProjectIOAdmission()
    intents = [
        _intent(f"p-{index}", service) for index in range(65) for service in ("authority", "primary", "background")
    ]
    owners = list(dict.fromkeys(item.owner for item in intents))
    granted = set()
    for _ in range(250):
        executed = []
        admission.begin(owners, blocked=(), free_slots=4)
        _offer_all(admission, intents, executed)
        admission.finish()
        assert len({item.owner for item in executed}) == len(executed)
        assert len(executed) == 4
        assert admission.pending_count == 0
        granted.update(executed)
    assert granted == set(intents)


def test_candidate_priority_tracks_the_same_roster_advanced_by_grants():
    admission = ProjectIOAdmission()
    intents = [_intent(f"p-{index}") for index in range(3)]
    owners = [intent.owner for intent in intents]
    executed = []

    admission.begin(owners, blocked=(), free_slots=1)
    before = [admission.candidate_priority(owner, "primary") for owner in owners]
    _offer_all(admission, intents, executed)
    admission.finish()

    admission.begin(owners, blocked=(), free_slots=0)
    after = [admission.candidate_priority(owner, "primary") for owner in owners]
    admission.finish()

    assert before == [1, 4, 7]
    assert after == [7, 1, 4]
    assert executed == [intents[0]]


def test_later_attempt_gets_fresh_credit_after_global_yield():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    executed = []

    for kind in ("scheduler_claim", "scheduler_launch_authorize", "scheduler_reservation_reconcile"):
        intent = ServiceIntent(owner, "primary", kind, ("epoch", "attempt-1"), "cpu_primary")
        admission.begin([owner], blocked=(), free_slots=1)
        admission.offer(intent, partial(executed.append, intent))
        admission.finish()

    next_claim = ServiceIntent(owner, "primary", "scheduler_claim", ("epoch", "attempt-2"), "cpu_primary")
    ordinary = ServiceIntent(owner, "primary", "scheduler_observe", ("epoch", "observe"), "cpu_primary")
    admission.begin([owner], blocked=(), free_slots=1)
    admission.offer(ordinary, partial(executed.append, ordinary))
    admission.offer(next_claim, partial(executed.append, next_claim))
    admission.finish()

    background = ServiceIntent(owner, "background", "notification_service", ("epoch", "notify"))
    admission.begin([owner], blocked=(), free_slots=1)
    admission.offer(background, partial(executed.append, background))
    admission.offer(next_claim, partial(executed.append, next_claim))
    admission.finish()

    admission.begin([owner], blocked=(), free_slots=1)
    admission.offer(ordinary, partial(executed.append, ordinary))
    admission.offer(next_claim, partial(executed.append, next_claim))
    admission.finish()

    assert executed[-3:] == [ordinary, background, next_claim]


def test_continuous_new_attempts_cannot_reset_global_critical_burst():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    ordinary = ServiceIntent(owner, "primary", "scheduler_observe", ("epoch", "observe"), "cpu_primary")
    background = ServiceIntent(owner, "background", "notification_service", ("epoch", "notify"))
    executed = []

    for attempt_number in range(1, 5):
        for kind in ("scheduler_claim", "scheduler_launch_authorize", "scheduler_reservation_reconcile"):
            critical = ServiceIntent(
                owner,
                "primary",
                kind,
                ("epoch", f"attempt-{attempt_number}"),
                "cpu_primary",
            )
            admission.begin([owner], blocked=(), free_slots=1)
            _offer_all(admission, [ordinary, background, critical], executed)
            admission.finish()

    assert ordinary in executed
    assert background in executed
    critical_run = 0
    longest_critical_run = 0
    for intent in executed:
        if intent.operation_kind in {
            "scheduler_claim",
            "scheduler_launch_authorize",
            "scheduler_reservation_reconcile",
        }:
            critical_run += 1
            longest_critical_run = max(longest_critical_run, critical_run)
        else:
            critical_run = 0
    assert longest_critical_run == 3


def test_new_attempt_has_independent_credit_after_prior_attempt_exhaustion():
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    executed = []
    first = ServiceIntent(owner, "primary", "scheduler_claim", ("epoch", "attempt-1"), "cpu_primary")
    for _ in range(3):
        admission.begin([owner], blocked=(), free_slots=1)
        admission.offer(first, partial(executed.append, first))
        admission.finish()

    ordinary = ServiceIntent(owner, "primary", "scheduler_observe", ("epoch", "observe"), "cpu_primary")
    background = ServiceIntent(owner, "background", "notification_service", ("epoch", "notify"))
    for intent in (ordinary, background):
        admission.begin([owner], blocked=(), free_slots=1)
        admission.offer(intent, partial(executed.append, intent))
        admission.finish()

    next_claim = ServiceIntent(owner, "primary", "scheduler_claim", ("epoch", "attempt-2"), "cpu_primary")
    admission.begin([owner], blocked=(), free_slots=1)
    admission.offer(ordinary, partial(executed.append, ordinary))
    admission.offer(next_claim, partial(executed.append, next_claim))
    admission.finish()

    assert executed[-1] == next_claim


def test_alternating_producer_windows_do_not_phase_lock_with_admission():
    admission = ProjectIOAdmission()
    intents = [_intent(f"p-{index}") for index in range(128)]
    owners = [item.owner for item in intents]
    executed = []
    for turn in range(64):
        admission.begin(owners, blocked=(), free_slots=4)
        start = (turn % 2) * 64
        _offer_all(admission, intents[start : start + 64], executed)
        admission.finish()
    assert set(executed) == set(intents)


def test_new_arrivals_do_not_reset_discovery_of_incumbents():
    admission = ProjectIOAdmission()
    incumbents = [_intent(f"old-{index}", "background") for index in range(65)]
    intents = list(incumbents)
    executed = []
    for turn in range(150):
        intents.insert(0, _intent(f"arrival-{turn}", "background"))
        admission.begin([item.owner for item in intents], blocked=(), free_slots=4)
        _offer_all(admission, intents, executed)
        admission.finish()
    assert set(incumbents) <= set(executed)


def test_busy_class_cannot_evict_other_class_behind_idle_roster_entries():
    admission = ProjectIOAdmission()
    owners = [_intent(f"p-{index}").owner for index in range(1000)]
    authority = [_intent(f"p-{index}", "authority") for index in range(65)]
    background = _intent("p-999", "background")
    primary = _intent("p-998", "primary")
    executed = []
    for _ in range(2):
        admission.begin(owners, blocked=(), free_slots=4)
        _offer_all(admission, [*authority, background, primary], executed)
        admission.finish()
    assert background in executed
    assert primary in executed


def test_failed_collection_discards_all_actions():
    admission = ProjectIOAdmission()
    intent = _intent("a")
    executed = []
    admission.begin([intent.owner], blocked=(), free_slots=4)
    _offer_all(admission, [intent], executed)
    admission.finish(failed=True)
    assert executed == []
    assert not admission.active
    assert not admission.has_deferred_work
    assert admission.pending_count == 0
    admission.begin([intent.owner], blocked=(), free_slots=4)
    admission.finish()
    assert executed == []


def test_action_failure_cleans_turn_without_running_later_actions():
    admission = ProjectIOAdmission()
    intents = [_intent("a"), _intent("b")]
    executed = []

    def fail():
        raise OSError("prepare failed")

    admission.begin([item.owner for item in intents], blocked=(), free_slots=4)
    admission.offer(intents[0], fail)
    admission.offer(intents[1], partial(executed.append, intents[1]))
    with pytest.raises(OSError, match="prepare failed"):
        admission.finish()
    assert executed == []
    assert not admission.active
    assert admission.pending_count == 0
    admission.begin([item.owner for item in intents], blocked=(), free_slots=4)
    admission.finish()
    assert executed == []


def test_failed_action_does_not_spend_unattempted_peers_fair_turns():
    admission = ProjectIOAdmission()
    intents = [_intent(name) for name in "abcd"]
    executed = []

    def fail():
        raise OSError("prepare failed")

    for _ in range(2):
        admission.begin([item.owner for item in intents], blocked=(), free_slots=4)
        admission.offer(intents[0], fail)
        _offer_all(admission, intents[1:], executed)
        with pytest.raises(OSError, match="prepare failed"):
            admission.finish()
        assert admission.has_deferred_work
    assert set(executed) == set(intents[1:])


def test_evicted_owner_cannot_reenter_with_newer_duplicate_work():
    admission = ProjectIOAdmission()
    authority = [_intent(f"a-{index}", "authority") for index in range(64)]
    primary = [_intent(f"p-{index}") for index in range(22)]
    background = [_intent(f"b-{index}", "background") for index in range(21)]
    intents = authority + primary + background
    newer = _intent("a-63", "authority", epoch="newer-work")
    executed = []
    for _ in range(100):
        admission.begin([item.owner for item in intents], blocked=(), free_slots=4)
        _offer_all(admission, intents + [newer], executed)
        admission.finish()
    assert newer not in executed
    assert authority[-1] in executed


def test_critical_continuation_reenters_after_its_discovery_owner_crosses_window():
    admission = ProjectIOAdmission()
    discoveries = [
        ServiceIntent(_intent(f"p-{index}").owner, "primary", "scheduler_observe", ("epoch-1",), "cpu_primary")
        for index in range(65)
    ]
    continuation = ServiceIntent(
        discoveries[-1].owner,
        "primary",
        "scheduler_claim",
        ("epoch-1",),
        "cpu_primary",
    )
    executed = []

    for _ in range(70):
        admission.begin([item.owner for item in discoveries], blocked=(), free_slots=1)
        _offer_all(admission, discoveries, executed)
        admission.offer(continuation, partial(executed.append, continuation))
        admission.finish()
        if continuation in executed:
            break

    assert continuation in executed


@pytest.mark.parametrize("free_slots", [True, -1, 5, 1.0])
def test_invalid_turn_does_not_open_collection(free_slots):
    admission = ProjectIOAdmission()
    intent = _intent("a")
    with pytest.raises(ValueError):
        admission.begin([intent.owner], blocked=(), free_slots=free_slots)
    assert not admission.active
    admission.begin([intent.owner], blocked=(), free_slots=1)
    admission.finish()


def test_unknown_owners_and_blocked_only_work_do_not_trigger_polling():
    admission = ProjectIOAdmission()
    known, unknown = _intent("known"), _intent("unknown")
    executed = []
    admission.begin([known.owner], blocked=[known.owner], free_slots=1)
    _offer_all(admission, [known, unknown], executed)
    assert admission.pending_count == 1
    admission.finish()
    assert executed == []
    assert not admission.has_deferred_work


def test_state_guards_reject_nested_collection_and_draining_reentry():
    admission = ProjectIOAdmission()
    intent = _intent("a")
    with pytest.raises(RuntimeError):
        admission.offer(intent, lambda: None)
    with pytest.raises(RuntimeError):
        admission.finish()
    admission.begin([intent.owner], blocked=(), free_slots=1)
    with pytest.raises(RuntimeError):
        admission.begin([intent.owner], blocked=(), free_slots=1)
    with pytest.raises(ValueError):
        admission.offer(intent, None)
    assert admission.pending_count == 0

    def action():
        with pytest.raises(RuntimeError):
            admission.offer(intent, lambda: None)
        with pytest.raises(RuntimeError):
            admission.begin([intent.owner], blocked=(), free_slots=1)
        with pytest.raises(RuntimeError):
            admission.finish()

    admission.offer(intent, action)
    admission.finish()
    assert not admission.active


@pytest.mark.parametrize("reverse", [False, True])
def test_continuous_progress_cannot_rewind_ordinary_background_rotation(reverse):
    admission = ProjectIOAdmission()
    owner = _intent("a").owner
    kinds = ["recovery_admission", "upgrade_service", "activation_observe", "machine_snapshot_publish"]
    ordinary = [ServiceIntent(owner, "background", kind, ()) for kind in kinds]
    executed = []
    for turn in range(3 * len(ordinary) * 2):
        phase = "observe" if turn % 2 == 0 else "publish"
        progress = ServiceIntent(owner, "background", "progress_projection", ("epoch", "progress", "attempt", phase))
        intents = [progress, *ordinary]
        admission.begin([owner], blocked=(), free_slots=1)
        _offer_all(admission, list(reversed(intents)) if reverse else intents, executed)
        admission.finish()
    # Every continuously waiting family gets two full turns despite an endless
    # producer stream. In particular upgrade cannot remain unknown after restart.
    assert [item for item in executed if item.operation_kind != "progress_projection"] == ordinary * 2
