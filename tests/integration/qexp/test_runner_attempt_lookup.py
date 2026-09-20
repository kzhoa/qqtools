"""Runner launch reads only the Attempt identified by its authorized claim."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, runner, submit
from qqtools.plugins.qexp.runtime.paths import attempt_path, task_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def prepared(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    value = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))
    return cfg, task, attempt, value


@pytest.mark.parametrize("history_count", [0, 1000])
@pytest.mark.parametrize("is_opaque", [False, True])
def test_runner_reads_one_attempt_without_history_enumeration(tmp_path, monkeypatch, history_count, is_opaque):
    cfg, task, attempt, value = prepared(tmp_path)
    if is_opaque:
        attempt.attempt_id = "persisted-opaque-attempt"
        value["attempt"]["attempt_id"] = attempt.attempt_id
        atomic_replace(attempt_path(cfg.shared_root, task.task_id, 1), value)
        task_value = read_json(task_path(cfg.shared_root, task.task_id))
        task_value["task"]["claim_control"]["active_claim"]["attempt_id"] = attempt.attempt_id
        task_value["task"]["attempt_control"]["current_attempt_id"] = attempt.attempt_id
        atomic_replace(task_path(cfg.shared_root, task.task_id), task_value)
    directory = attempt_path(cfg.shared_root, task.task_id, 1).parent
    for number in range(2, history_count + 2):
        (directory / f"{number}.json").write_text("unrelated history must not be parsed")
    reads = []
    original_read = runner.read_json
    original_scandir = os.scandir

    def read(path):
        if path.parent == directory:
            reads.append(path)
        return original_read(path)

    def scandir(path):
        if not isinstance(path, int) and Path(path) == directory:
            pytest.fail("runner enumerated Attempt history")
        return original_scandir(path)

    monkeypatch.setattr(runner, "read_json", read)
    monkeypatch.setattr(os, "scandir", scandir)
    launches = []

    def spawn(*args, **kwargs):
        launches.append(args)
        return SimpleNamespace(pid=99999991, wait=lambda: 0)

    assert (
        runner.run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            value["attempt"]["authorization"]["launch_id"],
            popen_factory=spawn,
        )
        == 0
    )
    assert len(launches) == 1
    assert reads == [attempt_path(cfg.shared_root, task.task_id, 1)]


@pytest.mark.parametrize("field", ["task_id", "attempt_id", "attempt_number"])
def test_runner_rejects_mismatched_direct_attempt_before_intent_or_spawn(tmp_path, field):
    cfg, task, attempt, value = prepared(tmp_path)
    value["attempt"][field] = 2 if field == "attempt_number" else "another-identity"
    atomic_replace(attempt_path(cfg.shared_root, task.task_id, 1), value)

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched direct Attempt launched a workload")

    with pytest.raises(RuntimeError, match="not authorized"):
        runner.run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            value["attempt"]["authorization"]["launch_id"],
            popen_factory=forbidden,
        )
    assert not runner.launch_intent_path(cfg, attempt.attempt_id).exists()


def test_runner_does_not_recover_missing_number_by_searching_history(tmp_path):
    cfg, task, attempt, value = prepared(tmp_path)
    expected = attempt_path(cfg.shared_root, task.task_id, 1)
    expected.rename(attempt_path(cfg.shared_root, task.task_id, 2))

    def forbidden(*args, **kwargs):
        pytest.fail("missing direct Attempt launched a workload")

    with pytest.raises(FileNotFoundError) as caught:
        runner.run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            value["attempt"]["authorization"]["launch_id"],
            popen_factory=forbidden,
        )
    assert Path(caught.value.filename) == expected
    assert not runner.launch_intent_path(cfg, attempt.attempt_id).exists()
