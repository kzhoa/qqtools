"""Rank-local observation must not change evaluation collective participation."""

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from qqtools.plugins.qpipeline.integrations.qexp_progress import qexp_progress_factory
from qqtools.plugins.qpipeline.runner.agent import RunningAgent
from qqtools.plugins.qpipeline.runner.contracts import ObserverBindings
from qqtools.plugins.qpipeline.runner.observation_plugins import ObservationContext, install_observation_plugins
from qqtools.plugins.qpipeline.runner.runner_utils.progress import ProgressTracker
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunMode
from qqtools.plugins.qpipeline.types import Stage

from .conftest import SimpleModel, SimpleTask

pytestmark = pytest.mark.integration


def _rank_run(rank: int, directory: str, modes: tuple[str, ...]) -> None:
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    original_reduce = dist.all_reduce
    original_gather = dist.all_gather_object
    calls = []

    def record_reduce(*args, **kwargs):
        calls.append("reduce")
        return original_reduce(*args, **kwargs)

    def record_gather(*args, **kwargs):
        calls.append("gather")
        return original_gather(*args, **kwargs)

    dist.all_reduce = record_reduce
    dist.all_gather_object = record_gather
    try:
        task = SimpleTask(num_samples=256)
        outcomes = {}
        for with_builtins in (False, True):
            for mode in modes:
                torch.manual_seed(1234 + rank)
                os.environ.pop("QEXP_PROGRESS_V2_PATH", None)
                model = SimpleModel()
                calls.clear()
                observers = ObserverBindings()
                lifecycle = None
                if with_builtins:
                    tracker = ProgressTracker(None, print_freq=1, render_type="plain", rank=rank)
                    observers.bind("epoch_started", tracker.on_epoch_start)
                    observers.bind("progress_tick", tracker.on_progress_tick)
                    observers.bind("table_update", tracker.on_table_update)
                    observers.bind("epoch_committed", tracker.on_epoch_end)
                    observers.bind("evaluation_started", tracker.on_eval_start)
                    observers.bind("evaluation_committed", tracker.on_eval_end)
                if rank == 0 and mode != "disabled":
                    if mode in {"qexp", "qexp_v2"}:
                        os.environ["QEXP_PROGRESS_PATH"] = str(root / "progress.json")
                        if mode == "qexp_v2":
                            os.environ["QEXP_PROGRESS_V2_PATH"] = str(root / "progress-v2.json")
                        factories = (qexp_progress_factory,)
                    elif mode == "factory_failure":

                        def fail_factory(_):
                            raise RuntimeError("connector unavailable")

                        factories = (fail_factory,)
                    else:

                        class BrokenPlugin:
                            identifier = "broken"

                            def __init__(self):
                                self.subscriptions = (("evaluation_batch_committed", self.fail),)

                            def fail(self, fact):
                                raise RuntimeError("observer failure")

                            def close(self):
                                pass

                        factories = (lambda _: BrokenPlugin(),)
                    lifecycle = install_observation_plugins(
                        factories, ObservationContext(rank=rank, max_steps=1, max_epochs=None), observers
                    )
                observers.freeze()
                agent = RunningAgent(
                    model=model,
                    task=task,
                    loss_fn=nn.MSELoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                    config=RunConfig(
                        device=torch.device("cpu"),
                        distributed=True,
                        rank=rank,
                        run_mode=RunMode.STEP,
                        max_steps=1,
                        eval_interval=1,
                    ),
                    device=torch.device("cpu"),
                    observers=observers,
                )
                agent.run()
                result = agent._evaluate_loader(model, task.val_loader, Stage.VAL)
                if lifecycle is not None:
                    lifecycle.close()
                outcomes[f"{with_builtins}:{mode}"] = {"calls": list(calls), "metrics": result}
        (root / f"rank-{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.all_reduce = original_reduce
        dist.all_gather_object = original_gather
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "modes",
    [
        ("disabled", "qexp", "factory_failure", "callback_failure"),
        ("disabled", "qexp_v2"),
    ],
    ids=("existing-plugins", "qexp-v2"),
)
def test_rank_zero_connector_and_failures_preserve_collective_path(tmp_path, modes):
    if not sys.platform.startswith("linux"):
        pytest.skip("gloo process-group regression requires Linux")
    process_context = mp.start_processes(
        _rank_run, args=(str(tmp_path), modes), nprocs=2, join=False, start_method="fork"
    )
    try:
        deadline = time.monotonic() + 90
        while not process_context.join(timeout=1):
            assert time.monotonic() < deadline, "two-rank evaluation exceeded its deadline"
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    for rank in (0, 1):
        outcomes = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for with_builtins in (False, True):
            baseline = outcomes[f"{with_builtins}:disabled"]
            assert baseline["calls"], "the baseline must exercise real collectives"
            for mode in modes:
                assert outcomes[f"{with_builtins}:{mode}"]["calls"] == baseline["calls"]
                assert outcomes[f"{with_builtins}:{mode}"]["metrics"] == baseline["metrics"]
