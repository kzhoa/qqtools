from __future__ import annotations

from qqtools.plugins.qexp.agent.dispatch_probe import PrimaryProbeSession


def _complete_route(
    session: PrimaryProbeSession,
    key: tuple[str, str, str],
    *,
    revision: int,
    cursor: object,
) -> None:
    decision = session.begin_route(key, revision)
    assert decision.should_scan
    session.record_progress(key, cursor)
    decision = session.finish_route(key, revision)
    assert not decision.has_index_changed


def test_route_progress_and_completion_are_replaced_as_immutable_state() -> None:
    session = PrimaryProbeSession()
    key = ("project-a", "shared", "gpu")

    initial = session.route(key)
    assert initial.cursor is None
    assert initial.revision is None
    assert not initial.is_complete
    assert initial.recheck is None

    session.begin_round("gpu", (key,))
    _complete_route(session, key, revision=3, cursor="baseline-end")

    completed = session.route(key)
    assert completed.cursor == "baseline-end"
    assert completed.revision == 3
    assert completed.is_complete
    assert initial is not completed
    assert dict(session.completed_revisions("gpu", (key,))) == {key: 3}


def test_revision_change_invalidates_completion_and_dependency_recheck() -> None:
    session = PrimaryProbeSession()
    key = ("project-a", "home", "gpu")
    session.begin_round("gpu", (key,))
    session.begin_route(key, 4)
    session.record_dependency_wait(key, None)
    session.record_progress(key, "baseline-end")
    session.finish_route(key, 4)

    changed = session.begin_route(key, 5)

    assert changed.has_index_changed
    assert session.route(key).cursor is None
    assert session.route(key).revision == 5
    assert not session.route(key).is_complete
    assert session.route(key).recheck is None
    assert session.completed_revisions("gpu", (key,)) is None


def test_null_recheck_cursor_is_distinct_from_no_recheck() -> None:
    session = PrimaryProbeSession()
    key = ("project-a", "shared", "gpu")
    session.begin_round("gpu", (key,))
    session.begin_route(key, 1)

    session.record_dependency_wait(key, None)

    route = session.route(key)
    assert route.recheck is not None
    assert route.recheck.cursor is None
    assert not route.recheck.is_active


def test_recheck_progress_preserves_completed_baseline_across_slices() -> None:
    session = PrimaryProbeSession()
    key = ("project-a", "home", "gpu")
    session.begin_round("gpu", (key,))
    session.begin_route(key, 2)
    session.record_dependency_wait(key, "candidate-start")
    session.record_progress(key, "baseline-end")
    session.finish_route(key, 2)

    session.begin_round("gpu", (key,))
    assert session.next_recheck("gpu") == key
    session.record_progress(key, "temporary-progress")

    during_recheck = session.route(key)
    assert during_recheck.cursor == "baseline-end"
    assert during_recheck.revision == 2
    assert during_recheck.is_complete
    assert during_recheck.recheck is not None
    assert during_recheck.recheck.cursor == "temporary-progress"
    assert during_recheck.recheck.is_active

    session.finish_recheck(key, has_waiting_candidate=True)
    after_recheck = session.route(key)
    assert after_recheck.cursor == "baseline-end"
    assert after_recheck.is_complete
    assert after_recheck.recheck is not None
    assert not after_recheck.recheck.is_active


def test_budget_exhaustion_resumes_the_active_recheck_before_rotating() -> None:
    session = PrimaryProbeSession()
    first = ("project-a", "shared", "gpu")
    second = ("project-b", "shared", "gpu")
    keys = (first, second)
    session.begin_round("gpu", keys)
    for key in keys:
        session.begin_route(key, 1)
        session.record_dependency_wait(key, None)
        session.finish_route(key, 1)

    session.begin_round("gpu", keys)
    assert session.next_recheck("gpu") == first
    session.record_progress(first, "resume-here")

    session.begin_round("gpu", keys)
    assert session.next_recheck("gpu") == first
    assert session.route(first).recheck.cursor == "resume-here"


def test_recheck_wraps_before_removing_a_dependency_observation() -> None:
    session = PrimaryProbeSession()
    key = ("project-a", "shared", "gpu")
    session.begin_round("gpu", (key,))
    session.begin_route(key, 1)
    session.record_dependency_wait(key, "middle")
    session.finish_route(key, 1)

    session.begin_round("gpu", (key,))
    assert session.next_recheck("gpu") == key
    session.finish_recheck(key, has_waiting_candidate=False)
    assert session.route(key).recheck is not None
    assert session.route(key).recheck.cursor is None

    session.begin_round("gpu", (key,))
    assert session.next_recheck("gpu") == key
    session.finish_recheck(key, has_waiting_candidate=False)
    assert session.route(key).recheck is None


def test_dependency_rechecks_rotate_fairly_between_routes() -> None:
    session = PrimaryProbeSession()
    first = ("project-a", "shared", "gpu")
    second = ("project-b", "home", "gpu")
    keys = (first, second)
    session.begin_round("gpu", keys)
    for key in keys:
        session.begin_route(key, 1)
        session.record_dependency_wait(key, None)
        session.record_progress(key, f"{key[0]}-end")
        session.finish_route(key, 1)

    session.begin_round("gpu", keys)
    assert session.next_recheck("gpu") == first
    session.finish_recheck(first, has_waiting_candidate=True)
    session.begin_round("gpu", keys)
    assert session.next_recheck("gpu") == second


def test_incomplete_baseline_prevents_dependency_recheck_selection() -> None:
    session = PrimaryProbeSession()
    completed = ("project-a", "shared", "gpu")
    incomplete = ("project-b", "shared", "gpu")
    keys = (completed, incomplete)
    session.begin_round("gpu", keys)
    session.begin_route(completed, 1)
    session.record_dependency_wait(completed, None)
    session.finish_route(completed, 1)

    assert session.next_recheck("gpu") is None


def test_completed_revisions_require_every_route_and_keep_lanes_independent() -> None:
    session = PrimaryProbeSession()
    gpu = ("project-a", "shared", "gpu")
    cpu = ("project-a", "shared", "cpu")
    session.begin_round("gpu", (gpu,))
    session.begin_round("cpu", (cpu,))
    _complete_route(session, gpu, revision=7, cursor="gpu-end")

    assert dict(session.completed_revisions("gpu", (gpu,))) == {gpu: 7}
    assert session.completed_revisions("cpu", (cpu,)) is None
    assert session.route(cpu).revision is None


def test_invalidate_routes_resets_only_requested_observations() -> None:
    session = PrimaryProbeSession()
    stale = ("project-a", "shared", "gpu")
    retained = ("project-b", "home", "gpu")
    keys = (stale, retained)
    session.begin_round("gpu", keys)
    _complete_route(session, stale, revision=2, cursor="stale-end")
    _complete_route(session, retained, revision=3, cursor="retained-end")

    session.invalidate_routes((stale,))

    assert session.route(stale).revision is None
    assert session.route(stale).cursor is None
    assert not session.route(stale).is_complete
    assert session.route(retained).revision == 3
    assert session.route(retained).is_complete
    assert session.completed_revisions("gpu", keys) is None


def test_begin_round_drops_removed_routes_from_required_snapshot() -> None:
    session = PrimaryProbeSession()
    retained = ("project-a", "shared", "gpu")
    removed = ("project-b", "home", "gpu")
    session.begin_round("gpu", (retained, removed))
    _complete_route(session, retained, revision=1, cursor="retained-end")

    session.begin_round("gpu", (retained,))

    assert dict(session.completed_revisions("gpu", (retained,))) == {retained: 1}
