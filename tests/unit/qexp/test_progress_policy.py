import json
import math
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import progress_policy
from qqtools.plugins.qexp.runtime import progress as runtime


def test_default_set_show_and_context_freeze(tmp_path):
    cfg = SimpleNamespace(shared_root=tmp_path / "shared", runtime_root=tmp_path / "rt", machine_name="m1")
    task = SimpleNamespace(task_id="task")
    attempt = SimpleNamespace(attempt_id="attempt", attempt_number=1, authorization={"launch_id": "launch"})

    assert progress_policy.show_progress_policy(cfg)["interval_seconds"] == 30
    progress_policy.set_progress_policy(cfg, 60)
    assert progress_policy.show_progress_policy(cfg) == {
        "interval_seconds": 60,
        "source": "configured",
        "applies_to": "new_launches",
    }
    channel = runtime.prepare_progress_channel(
        cfg,
        task,
        attempt,
        wrapper_start_time_ticks=1,
        interval_seconds=60,
    )
    assert channel is not None and channel.interval_seconds == 60

    progress_policy.set_progress_policy(cfg, 90)
    recovered = runtime.prepare_progress_channel(
        cfg,
        task,
        attempt,
        wrapper_start_time_ticks=1,
        interval_seconds=90,
    )
    assert recovered is not None and recovered.interval_seconds == 60


@pytest.mark.parametrize("value", [True, False, 0, 0.5, math.inf, -math.inf, math.nan, "30"])
def test_policy_rejects_non_finite_small_bool_and_non_numeric_values(tmp_path, value):
    with pytest.raises(ValueError):
        progress_policy.set_progress_policy(tmp_path, value)


def test_explicit_operations_reject_malformed_policy_but_runtime_falls_back(tmp_path):
    path = progress_policy.progress_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "interval_seconds": True}))

    with pytest.raises(ValueError):
        progress_policy.show_progress_policy(tmp_path)
    with pytest.raises(ValueError):
        progress_policy.set_progress_policy(tmp_path, 60)

    resolved = progress_policy.resolve_progress_policy(tmp_path)
    assert resolved["interval_seconds"] == 30
    assert resolved["source"] == "default"
    assert "diagnostic_reason" in resolved


def test_concurrent_sets_serialize_and_last_committed_value_wins(tmp_path, monkeypatch):
    commits = []
    original = progress_policy.atomic_replace

    def record_commit(path, value):
        result = original(path, value)
        commits.append(value["interval_seconds"])
        return result

    monkeypatch.setattr(progress_policy, "atomic_replace", record_commit)

    def set_value(value):
        progress_policy.set_progress_policy(tmp_path, value)

    threads = [threading.Thread(target=set_value, args=(value,)) for value in range(30, 50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(commits) == len(threads)
    assert progress_policy.show_progress_policy(tmp_path)["interval_seconds"] == commits[-1]


@pytest.mark.parametrize("interval", [30.0, 60.0])
def test_stable_attempt_phases_bound_scale_writes_and_skip_missed_slots(interval):
    bursts = Counter()
    total = 0
    for index in range(1000):
        attempt_id = f"attempt-{index}"
        deadline = runtime._initial_deadline(attempt_id, 0.0, interval)
        writes = []
        while deadline < 11 * interval:
            writes.append(deadline)
            bursts[int(deadline)] += 1
            deadline = runtime._next_deadline(deadline, deadline, interval)
        assert len(writes) == 10
        assert all(b - a >= interval - 1e-9 for a, b in zip(writes, writes[1:]))
        total += len(writes)

    assert total == 10_000
    assert max(bursts.values()) <= 2 * math.ceil(1000 / interval)

    first = runtime._initial_deadline("attempt-missed", 0.0, interval)
    delayed = first + 4.5 * interval
    resumed = runtime._next_deadline(first, delayed, interval)
    assert resumed >= delayed + interval
    assert math.isclose((resumed - first) % interval, 0.0, abs_tol=1e-9)


@pytest.mark.parametrize("value", [1e308, 10**400])
def test_large_finite_policy_values_remain_valid(tmp_path, value):
    progress_policy.set_progress_policy(tmp_path, value)
    assert progress_policy.show_progress_policy(tmp_path)["interval_seconds"] == value
