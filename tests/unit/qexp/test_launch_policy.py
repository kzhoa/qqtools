import json
import math
import threading
from collections import Counter

import pytest

from qqtools.plugins.qexp import launch_policy


def test_default_set_show_and_runtime_resolution(tmp_path):
    assert launch_policy.show_launch_handoff_policy(tmp_path) == {
        "timeout_seconds": 10,
        "source": "default",
        "applies_to": "new_launches",
    }

    launch_policy.set_launch_handoff_policy(tmp_path, 12.5)

    expected = {
        "timeout_seconds": 12.5,
        "source": "configured",
        "applies_to": "new_launches",
    }
    assert launch_policy.show_launch_handoff_policy(tmp_path) == expected
    assert launch_policy.resolve_launch_handoff_policy(tmp_path) == expected


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 0.5, 301, math.inf, -math.inf, math.nan, "10", None],
)
def test_policy_rejects_values_outside_finite_bounded_range(tmp_path, value):
    with pytest.raises(ValueError):
        launch_policy.set_launch_handoff_policy(tmp_path, value)


@pytest.mark.parametrize("value", [1, 1.0, 10, 300, 300.0])
def test_policy_accepts_inclusive_bounds(tmp_path, value):
    result = launch_policy.set_launch_handoff_policy(tmp_path, value)
    assert result["timeout_seconds"] == value


def test_explicit_operations_reject_malformed_policy_but_runtime_falls_back(tmp_path):
    path = launch_policy.launch_handoff_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "timeout_seconds": math.inf}), encoding="utf-8")

    with pytest.raises(ValueError):
        launch_policy.show_launch_handoff_policy(tmp_path)
    with pytest.raises(ValueError):
        launch_policy.set_launch_handoff_policy(tmp_path, 20)

    resolved = launch_policy.resolve_launch_handoff_policy(tmp_path)
    assert resolved["timeout_seconds"] == 10
    assert resolved["source"] == "default"
    assert resolved["applies_to"] == "new_launches"
    assert "diagnostic_reason" in resolved
    assert len(resolved["diagnostic_reason"]) <= 160


def test_concurrent_sets_serialize_and_last_committed_value_wins(tmp_path, monkeypatch):
    commits = []
    original = launch_policy.atomic_replace

    def record_commit(path, value):
        result = original(path, value)
        commits.append(value["timeout_seconds"])
        return result

    monkeypatch.setattr(launch_policy, "atomic_replace", record_commit)

    threads = [
        threading.Thread(target=launch_policy.set_launch_handoff_policy, args=(tmp_path, value))
        for value in range(10, 30)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert Counter(commits) == Counter(range(10, 30))
    assert launch_policy.show_launch_handoff_policy(tmp_path)["timeout_seconds"] == commits[-1]
