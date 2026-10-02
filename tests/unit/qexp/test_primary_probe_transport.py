from __future__ import annotations

from copy import deepcopy

import pytest

from qqtools.plugins.qexp.agent.dispatch_probe import PrimaryProbeSession
from qqtools.plugins.qexp.agent.primary_probe_transport import (
    decode_probe_session,
    encode_probe_session,
    validate_probe_state,
)
from qqtools.plugins.qexp.runtime.ready import ReadyCursor


def test_probe_continuation_preserves_baseline_progress_and_dependency_recheck():
    session = PrimaryProbeSession()
    keys = [("project", scope, "gpu") for scope in ("shared", "home")]
    session.begin_round("gpu", keys)
    session.begin_route(keys[0], 7)
    session.record_dependency_wait(
        keys[0], ReadyCursor("project", "gpu-1", "shared", 1, "partition-a", "task-a.json", 3)
    )
    session.record_progress(keys[0], ReadyCursor("project", "gpu-1", "shared", 2, None, None, 4))
    session.finish_route(keys[0], 7)
    session.begin_route(keys[1], 8)
    session.record_progress(keys[1], ReadyCursor("project", "gpu-1", "home", None, None, None, 0))
    encoded = encode_probe_session(session, "project", "gpu")
    restored = decode_probe_session(encoded, "project", "gpu-1", "gpu")
    assert restored.route(keys[0]) == session.route(keys[0])
    assert restored.route(keys[1]) == session.route(keys[1])
    assert restored.completed_revisions("gpu", keys) is None
    restored.finish_route(keys[1], 8)
    assert restored.next_recheck("gpu") == keys[0]
    continued = decode_probe_session(encode_probe_session(restored, "project", "gpu"), "project", "gpu-1", "gpu")
    assert continued.next_recheck("gpu") == keys[0]
    assert continued.route(keys[0]).recheck.is_active


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", True),
        ("revision", -1),
        ("is_complete", 1),
        ("cursor", {"catalog_page": True, "partition": None, "after_name": None, "revision": 0}),
        ("cursor", {"catalog_page": 0, "partition": "../bad", "after_name": None, "revision": 0}),
        ("cursor", {"catalog_page": 0, "partition": None, "after_name": "../bad", "revision": 0}),
        ("recheck", {"cursor": None, "is_active": 1, "started_at_route_start": False}),
    ],
)
def test_probe_continuation_rejects_invalid_route_state(field, value):
    encoded = encode_probe_session(PrimaryProbeSession(), "project", "gpu")
    encoded["routes"]["home"][field] = value
    with pytest.raises(ValueError):
        validate_probe_state(encoded)


def test_completed_probe_requires_revision_and_detaches_input():
    encoded = encode_probe_session(PrimaryProbeSession(), "project", "cpu")
    detached = validate_probe_state(encoded)
    original = deepcopy(detached)
    encoded["routes"]["home"]["is_complete"] = True
    assert detached == original
    with pytest.raises(ValueError, match="baseline revision"):
        validate_probe_state(encoded)
