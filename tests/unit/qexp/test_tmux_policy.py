import json
import threading

import pytest

from qqtools.plugins.qexp import tmux_policy


def test_missing_policy_is_disabled_without_creating_a_file(tmp_path):
    assert tmux_policy.show_tmux_policy(tmp_path) == {
        "enabled": False,
        "source": "default",
        "applies_to": "new_observer_decisions",
    }
    assert not tmux_policy.tmux_policy_path(tmp_path).exists()


def test_set_show_and_exact_document(tmp_path):
    assert tmux_policy.set_tmux_policy(tmp_path, True) == {
        "enabled": True,
        "source": "configured",
        "applies_to": "new_observer_decisions",
    }
    assert json.loads(tmux_policy.tmux_policy_path(tmp_path).read_text(encoding="utf-8")) == {
        "version": 1,
        "enabled": True,
    }
    assert tmux_policy.show_tmux_policy(tmp_path)["enabled"] is True


@pytest.mark.parametrize(
    "document",
    [
        {"version": True, "enabled": False},
        {"version": 2, "enabled": False},
        {"version": 1, "enabled": 0},
        {"version": 1, "enabled": "false"},
        {"version": 1, "enabled": False, "extra": None},
        [],
    ],
)
def test_explicit_operations_reject_malformed_policy_but_runtime_disables(tmp_path, document):
    path = tmux_policy.tmux_policy_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        tmux_policy.show_tmux_policy(tmp_path)
    with pytest.raises(ValueError):
        tmux_policy.set_tmux_policy(tmp_path, True)

    resolved = tmux_policy.resolve_tmux_policy(tmp_path)
    assert resolved["enabled"] is False
    assert resolved["source"] == "default"
    assert resolved["diagnostic_reason"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_set_requires_an_actual_boolean(tmp_path, value):
    with pytest.raises(ValueError):
        tmux_policy.set_tmux_policy(tmp_path, value)


def test_concurrent_sets_are_serialized(tmp_path, monkeypatch):
    commits = []
    original = tmux_policy.atomic_replace

    def record(path, value):
        result = original(path, value)
        commits.append(value["enabled"])
        return result

    monkeypatch.setattr(tmux_policy, "atomic_replace", record)
    threads = [
        threading.Thread(target=tmux_policy.set_tmux_policy, args=(tmp_path, value)) for value in (True, False) * 8
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(commits) == len(threads)
    assert tmux_policy.show_tmux_policy(tmp_path)["enabled"] is commits[-1]
