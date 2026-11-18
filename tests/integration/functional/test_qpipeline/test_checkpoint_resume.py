"""Real checkpoint round trips protect progress cursors and lifecycle ordering."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import qqtools as qt
from qqtools.plugins.qpipeline.entry_utils.scheduler import qWarmupScheduler
from qqtools.plugins.qpipeline.runner.runner import train_runner
from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointManager
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class RecordingSampler(SequentialSampler):
    def __init__(self, data):
        super().__init__(data)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class BoundedLoader(DataLoader):
    def __iter__(self):
        self.passes = getattr(self, "passes", 0) + 1
        if self.passes > 3:
            raise AssertionError("Training repeated an exhausted loader")
        return super().__iter__()


class ResumeTask(qTaskBase):
    def __init__(self, model=None, *, fail_finalization=False, batches=7):
        super().__init__()
        x = torch.arange(batches, dtype=torch.float32).reshape(-1, 1) / 10
        data = TensorDataset(x, 2 * x)
        self.train_loader = BoundedLoader(data, batch_size=1, sampler=RecordingSampler(data))
        self.val_loader = DataLoader(data, batch_size=1)
        self.test_loader = None
        self.finalized = []
        self.starts = []
        self.updates = []
        self.validations = []
        self.model = model
        self.fail_finalization = fail_finalization

    def batch_forward(self, model, batch_data):
        x, y = batch_data
        return {"pred": model(x), "target": y}

    def batch_loss(self, out, batch_data, loss_fn):
        return {"loss": (loss_fn(out["pred"], out["target"]), 1)}

    def batch_metric(self, out, batch_data):
        return {"mse": (torch.nn.functional.mse_loss(out["pred"], out["target"]), 1)}

    def post_metrics_to_value(self, result):
        return result.get("mse", 0.0)

    def on_epoch_start(self, context):
        self.starts.append((context.epoch, context.global_step))

    def on_train_batch_end(self, context):
        self.updates.append((context.epoch, context.batch_index, context.global_step))

    def on_validation_end(self, context):
        self.validations.append(context.global_step)

    def on_epoch_end(self, context):
        if self.fail_finalization:
            raise RuntimeError("task finalization failed")
        self.finalized.append(context.completed_epoch)
        if self.model is not None:
            with torch.no_grad():
                self.model.weight.add_(10)

    def state_dict(self):
        return {"finalized": self.finalized.copy()}

    def load_state_dict(self, state):
        self.finalized = state["finalized"].copy()


def run_training(
    args,
    directory,
    *,
    task=None,
    restore=None,
    mode="epoch",
    epochs=1,
    steps=None,
    save_interval=1,
    completion=False,
    accum_grad=1,
    model=None,
):
    args = args.copy()
    args.ckp_file = str(restore) if restore else None
    args.runner.checkpoint.regular_latest_only = False
    args.runner.early_stop.patience = 999
    args.runner.completion = {"save": completion, "eval": False}
    model = model if model is not None else torch.nn.Linear(1, 1)
    task = task if task is not None else ResumeTask()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    scheduler = qWarmupScheduler(optimizer, 0, 1.0, main_scheduler, step_on="valid_end")
    result = train_runner(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=torch.nn.MSELoss(),
        args=args,
        run_mode=mode,
        max_epochs=epochs,
        max_steps=steps,
        save_interval=save_interval,
        eval_interval=1 if mode == "epoch" else 7,
        accum_grad=accum_grad,
        save_dir=str(directory),
        print_freq=100,
        observation_plugins=(),
        log_granularity=["eval"],
    )
    return result, task, model, optimizer, scheduler


@pytest.mark.parametrize("mode,accum_grad", [("epoch", 1), ("step", 1), ("step", 3)])
@pytest.mark.parametrize("completion,save_interval", [(False, 1), (True, 1), (True, 99)])
def test_epoch_end_round_trip(base_args, tmp_path, mode, accum_grad, completion, save_interval):
    updates_per_epoch = (7 + accum_grad - 1) // accum_grad
    source = tmp_path / "source"
    result, task, model, _, scheduler = run_training(
        base_args,
        source,
        mode=mode,
        steps=updates_per_epoch,
        accum_grad=accum_grad,
        completion=completion,
        save_interval=save_interval,
    )
    checkpoint = source / f"epoch0_step{updates_per_epoch}.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["epoch"] == payload["state"]["epoch"] == 0
    assert payload["state"]["batch_idx_in_epoch"] == 7
    assert payload["task_state_dict"]["finalized"] == [0]
    assert result["final_epoch"] == 1
    records = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    saves = [r for r in records if r.get("event") == "checkpoint_saved" and r["type"] == "regular"]
    assert sum(r["global_step"] == updates_per_epoch for r in saves) == 1
    assert saves[-1]["epoch"] == 0

    # A satisfied run limit must exit before opening the loader or dispatching epoch_start.
    done, done_task, done_model, _, done_scheduler = run_training(
        base_args,
        tmp_path / "done",
        restore=checkpoint,
        mode=mode,
        steps=updates_per_epoch,
        accum_grad=accum_grad,
    )
    assert done["final_step"] == updates_per_epoch
    assert done_task.starts == done_task.updates == done_task.validations == []
    assert getattr(done_task.train_loader, "passes", 0) == 0
    assert done_task.finalized == [0]
    assert done_task.train_loader.sampler.epochs == []
    torch.testing.assert_close(done_model.weight, model.weight)
    assert done_scheduler.state_dict() == scheduler.state_dict()

    resumed, resumed_task, _, _, _ = run_training(
        base_args,
        tmp_path / "resumed",
        restore=checkpoint,
        mode=mode,
        epochs=2,
        steps=2 * updates_per_epoch,
        accum_grad=accum_grad,
    )
    assert resumed["final_step"] == 2 * updates_per_epoch
    assert resumed_task.starts == [(1, updates_per_epoch)]
    assert resumed_task.train_loader.sampler.epochs == [1]
    assert resumed_task.updates[0][:2] == (1, 0)
    assert resumed_task.finalized == [0, 1]
    assert updates_per_epoch not in resumed_task.validations


def test_mid_epoch_resume_matches_uninterrupted_training(base_args, tmp_path):
    torch.manual_seed(42)
    _, _, expected_model, expected_optim, expected_scheduler = run_training(
        base_args,
        tmp_path / "full",
        mode="step",
        steps=7,
        save_interval=3,
    )
    checkpoint = tmp_path / "full" / "epoch0_step3.pt"
    result, task, model, optim, scheduler = run_training(
        base_args,
        tmp_path / "resume",
        restore=checkpoint,
        mode="step",
        steps=7,
    )
    assert task.starts == [(0, 3)]
    assert task.updates[0] == (0, 3, 4)
    assert result["final_step"] == 7
    for key, value in expected_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in expected_optim.state_dict()["state"].items():
        torch.testing.assert_close(optim.state_dict()["state"][key]["momentum_buffer"], value["momentum_buffer"])
    assert scheduler.state_dict() == expected_scheduler.state_dict()


def test_best_is_saved_before_finalization_and_can_resume_cursor(base_args, tmp_path):
    model = torch.nn.Linear(1, 1)
    task = ResumeTask(model)
    _, _, _, _, scheduler = run_training(base_args, tmp_path / "source", task=task, model=model)
    best_path = tmp_path / "source" / "best_epoch0_step7.pt"
    best = torch.load(best_path, weights_only=True)
    regular = torch.load(tmp_path / "source" / "epoch0_step7.pt", weights_only=True)
    assert best["task_state_dict"]["finalized"] == []
    assert regular["task_state_dict"]["finalized"] == [0]
    torch.testing.assert_close(regular["model_state_dict"]["weight"], best["model_state_dict"]["weight"] + 10)
    _, restored_task, restored_model, _, restored_scheduler = run_training(
        base_args,
        tmp_path / "restored",
        restore=best_path,
    )
    assert restored_task.finalized == restored_task.starts == []
    torch.testing.assert_close(restored_model.weight, best["model_state_dict"]["weight"])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    result, resumed_task, _, _, _ = run_training(
        base_args,
        tmp_path / "continue",
        restore=best_path,
        epochs=2,
    )
    assert result["final_step"] == 14
    assert resumed_task.starts == [(1, 7)]
    assert resumed_task.finalized == [1]  # No automatic reconstruction of epoch 0 callback state.


def test_best_survives_epoch_callback_failure(base_args, tmp_path):
    with pytest.raises(RuntimeError, match="task finalization failed"):
        run_training(base_args, tmp_path, task=ResumeTask(fail_finalization=True))
    best = torch.load(tmp_path / "best_epoch0_step7.pt", weights_only=True)
    assert best["global_step"] == 7
    assert not list(tmp_path.glob("epoch*.pt"))


def test_regular_save_failure_does_not_advance_cursor(base_args, tmp_path, monkeypatch):
    observed = []
    original_save = CheckpointManager.save

    def fail_regular(self, state, model, task, *args, **kwargs):
        if not kwargs.get("is_best", False):
            observed.append((state, task.finalized.copy()))
            raise OSError("disk failure")
        return original_save(self, state, model, task, *args, **kwargs)

    monkeypatch.setattr(CheckpointManager, "save", fail_regular)
    with pytest.raises(OSError, match="disk failure"):
        run_training(base_args, tmp_path)
    state, finalized = observed[0]
    assert finalized == [0]
    assert (state.epoch, state.batch_idx_in_epoch, state.global_step) == (0, 7, 7)


@pytest.mark.parametrize("cursor", [-1, 8, True, 1.5])
def test_invalid_restored_cursor_fails_before_epoch_start(base_args, tmp_path, cursor):
    run_training(base_args, tmp_path / "source")
    checkpoint = tmp_path / "invalid.pt"
    payload = torch.load(tmp_path / "source" / "epoch0_step7.pt", weights_only=True)
    payload["state"]["batch_idx_in_epoch"] = cursor
    torch.save(payload, checkpoint)
    task = ResumeTask()
    with pytest.raises(ValueError, match="Invalid checkpoint batch cursor"):
        run_training(base_args, tmp_path / "restore", task=task, restore=checkpoint, epochs=2)
    assert task.starts == task.updates == []


def test_empty_loader_fails_before_epoch_start(base_args, tmp_path):
    task = ResumeTask(batches=0)
    with pytest.raises(ValueError, match="non-empty train_loader"):
        run_training(base_args, tmp_path, task=task)
    assert task.starts == []


def _distributed_boundary(rank, directory):
    """Exercise the production finalization/save boundary on two live ranks."""
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    from qqtools.plugins.qpipeline.runner.agent import RunningAgent
    from qqtools.plugins.qpipeline.runner.contracts import EventListenerBindings
    from qqtools.plugins.qpipeline.runner.hooks import RunnerHooks
    from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointPlugin, CheckpointPolicy
    from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig

    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    results = {}
    try:
        for scenario in ("success", "fail_rank0", "fail_rank1", "save_failure"):
            task = ResumeTask()
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            listeners = EventListenerBindings()

            def finalize(context):
                task.on_epoch_end(context)
                (root / f"{scenario}-finalized-{rank}").touch()
                if scenario == f"fail_rank{rank}":
                    raise ValueError("rank-local epoch callback failed")

            listeners.bind("epoch_end", finalize)
            listeners.freeze()
            hooks = RunnerHooks()
            agent = RunningAgent(
                model=model,
                task=task,
                optimizer=optimizer,
                loss_fn=torch.nn.MSELoss(),
                config=RunConfig(
                    run_mode="step", max_steps=7, eval_interval=99, distributed=True, rank=rank, device="cpu"
                ),
                event_listeners=listeners,
                hooks=hooks,
            )
            agent.state.batch_idx_in_epoch = agent.state.global_step = 7
            save_dir = root / scenario
            if scenario == "save_failure" and rank == 0:
                save_dir.touch()  # A file where the checkpoint directory must be.
            dist.barrier()

            class ObservedManager(CheckpointManager):
                def save(self, *args, **kwargs):
                    (root / f"{scenario}-save-entered-{rank}").touch()
                    assert all((root / f"{scenario}-finalized-{r}").exists() for r in range(2))
                    return super().save(*args, **kwargs)

            manager = ObservedManager(str(save_dir), rank=rank)
            CheckpointPlugin(
                manager, model, task, agent.state, CheckpointPolicy(7, None, True, False), optimizer=optimizer
            ).register(hooks)
            hooks.freeze()
            try:
                agent._finalize_training_boundary(did_optimizer_step=True, is_epoch_end=True)
                error = None
            except Exception as exc:
                error = str(exc)
            results[scenario] = {"error": error, "epoch": agent.state.epoch, "cursor": agent.state.batch_idx_in_epoch}
            dist.barrier()
        (root / f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_ddp_finalization_and_save_failures_settle_before_cursor_commit(tmp_path):
    import time

    import torch.multiprocessing as mp

    processes = mp.spawn(_distributed_boundary, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("DDP checkpoint boundary did not settle within 45 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        results = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert results["success"] == {"error": None, "epoch": 1, "cursor": 0}
        for scenario in ("fail_rank0", "fail_rank1", "save_failure"):
            assert results[scenario]["error"]
            assert results[scenario]["epoch"] == 0
            assert results[scenario]["cursor"] == 7
        for scenario in ("fail_rank0", "fail_rank1"):
            assert "rank-local epoch callback failed" in results[scenario]["error"]
            assert not (tmp_path / f"{scenario}-save-entered-{rank}").exists()
    payload = torch.load(tmp_path / "success" / "epoch0_step7.pt", weights_only=True)
    assert payload["task_state_dict"] == {"finalized": [0]}
    assert payload["state"]["epoch"] == 0
    assert payload["state"]["batch_idx_in_epoch"] == 7


def test_public_pipeline_fit_resumes_checkpoint(base_args, tmp_path):
    from qqtools.plugins.qpipeline.qpipeline import qPipeline

    class CpuPipeline(qPipeline):
        @staticmethod
        def prepare_env(args):
            args.device = torch.device("cpu")
            args.distributed = False

        @staticmethod
        def prepare_model(args):
            return torch.nn.Linear(1, 1)

        @staticmethod
        def prepare_task(args):
            return ResumeTask()

        @staticmethod
        def prepare_loss(args):
            return torch.nn.MSELoss()

    args = base_args.copy()
    args.optim = qt.qDict({"optimizer": "adam", "optimizer_params": {"lr": 0.01}, "scheduler": None})
    args.runner.max_epochs = 1
    args.runner.max_steps = None
    args.runner.clip_grad = None
    args.runner.eval_interval = 99
    args.runner.save_interval = 1
    args.log_dir = str(tmp_path / "first")
    first = CpuPipeline(args)
    try:
        assert first.fit()["final_step"] == 7
    finally:
        first.logger.close()
    args.ckp_file = str(tmp_path / "first" / "epoch0_step7.pt")
    args.log_dir = str(tmp_path / "second")
    args.runner.max_epochs = 2
    second = CpuPipeline(args)
    try:
        assert second.fit()["final_step"] == 14
        assert second.task.starts == [(1, 7)]
        assert second.task.finalized == [0, 1]
    finally:
        second.logger.close()


def test_explicitly_shortened_epoch_keeps_mid_epoch_recovery_rule(base_args, tmp_path):
    from qqtools.plugins.qpipeline.runner.agent import RunningAgent
    from qqtools.plugins.qpipeline.runner.contracts import EventListenerBindings
    from qqtools.plugins.qpipeline.runner.hooks import LoopDirective, RunnerHooks
    from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointPlugin, CheckpointPolicy
    from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig

    task = ResumeTask()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    listeners = EventListenerBindings()
    listeners.bind("epoch_end", task.on_epoch_end)
    listeners.freeze()
    hooks = RunnerHooks()
    hooks.set_optimizer_step_end_hook(
        lambda context: LoopDirective(end_epoch=context.global_step == 3), provider_id="test.short_epoch"
    )
    agent = RunningAgent(
        model,
        task,
        torch.nn.MSELoss(),
        optimizer,
        config=RunConfig(run_mode="step", max_steps=3, eval_interval=99, device="cpu"),
        event_listeners=listeners,
        hooks=hooks,
    )
    CheckpointPlugin(
        CheckpointManager(str(tmp_path / "source")),
        model,
        task,
        agent.state,
        CheckpointPolicy(3, None, False, False),
        optimizer=optimizer,
    ).register(hooks)
    hooks.freeze()
    agent.run()
    assert agent.state.epoch == 1
    checkpoint = tmp_path / "source" / "epoch0_step3.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["state"]["batch_idx_in_epoch"] == 3
    assert payload["task_state_dict"]["finalized"] == [0]
    _, restored_task, _, _, _ = run_training(base_args, tmp_path / "resumed", restore=checkpoint)
    assert restored_task.updates[0] == (0, 3, 4)
    assert restored_task.finalized == [0, 0]  # Explicitly outside the exact boundary guarantee.


def test_partial_epoch_completion_preserves_cursor(base_args, tmp_path):
    result, task, _, _, _ = run_training(
        base_args, tmp_path / "source", mode="step", steps=3, save_interval=99, completion=True
    )
    assert result["final_epoch"] == 0
    assert task.finalized == []
    checkpoint = tmp_path / "source" / "epoch0_step3.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["state"]["batch_idx_in_epoch"] == 3
    _, resumed, _, _, _ = run_training(base_args, tmp_path / "resumed", restore=checkpoint)
    assert resumed.updates[0] == (0, 3, 4)
