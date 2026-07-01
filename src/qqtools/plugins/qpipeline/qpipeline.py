import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Literal, Optional

import torch
import yaml

import qqtools as qt

from . import entry_utils
from .entry_utils.qema import qEMA
from .qlogger import ConsoleLogger, qLogger
from .runner.runner import evaluate_runner, infer_runner, train_runner
from .task.qtask import qTaskBase


@qt.qdist.ddp_safe
def prepare_logdir(args):
    # ensure log dir exists
    if args["log_dir"]:
        Path(args["log_dir"]).mkdir(parents=True, exist_ok=True)

    # i/o
    if args.config_file is None and args.ckp_file is None:
        assert args.log_dir is not None
        args.config_file = str(Path(args["log_dir"], "config.yaml"))
        with open(args.config_file, "w") as f:
            yaml.dump(args.to_dict(), f)

    elif args.ckp_file is not None:
        assert args.log_dir is not None
        args.config_file = str(Path(args["log_dir"], "config_ckprecover.yaml"))
        with open(args.config_file, "w") as f:
            yaml.dump(args.to_dict(), f)


class qPipeline:
    """
    accept args,
    the format of args must follow `qargs-convention`
    need implement:
        - prepare_task(args),
        - prepare_model(args),
    run:
        - fit (train)
        - infer_only (infer)
    """

    def __init__(
        self,
        args: qt.qDict,
        mode: Optional[Literal["train", "test"]] = None,
        task: Optional[qTaskBase] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.args = args
        if mode is None:
            mode = "test" if getattr(args, "test", False) else "train"
        self.mode = mode
        self.task = task
        self.model = model
        self.loss_fn = None
        self.ema_model = None
        self.extra_ckp_caches = {}
        self._runtime_ready = False
        self.logger: Optional[qLogger] = None

        if self.mode == "train":
            self.init_for_train()
        else:
            self.init_for_test()

    def _ensure_runtime_ready(self) -> None:
        if self._runtime_ready:
            return

        args = self.args
        self.prepare_env(args)

        # Logger creation — log_dir is now known
        rank = qt.qdist.get_rank()
        if args.log_dir:
            self.logger = qLogger(args.log_dir, console=True)
        else:
            self.logger = ConsoleLogger(rank=rank)

        self.logger.info(f"[qPipeline] use args:\n {args}")

        if self.task is None:
            self.task = self.prepare_task(args)

        if self.task.has_implemented("pipe_middle_ware"):
            self.task.pipe_middle_ware(self)

        if self.model is None:
            self.model = self.prepare_model(args)

        self.model = self._place_model(self.model)

        if "to" in self.task._opt_impl:
            self.task.to(args.device)

        self._runtime_ready = True

    def _load_state_dict_file(self, state_file: str, *, description: str) -> None:
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        target_model = (
            self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        )
        target_model.load_state_dict(state)
        self.logger.info(f"[qPipeline] loaded {description} weights from {state_file}")

    def _place_model(self, model: torch.nn.Module) -> torch.nn.Module:
        args = self.args
        if not args.distributed:
            placed_model = model.to(args.device)
            self.logger.info(f"model moved to {args.device}")
            return placed_model
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.to(args.local_rank)
            placed_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                find_unused_parameters=False,
            )
            self.logger.info(f"model with DDP wrapped to {args.local_rank}")
            return placed_model

        self.logger.info(f"model with DDP already wrapped on {next(model.parameters()).device}")
        return model

    def init_for_train(self):
        args = self.args
        self._ensure_runtime_ready()

        if self.task.has_implemented("init_train_model"):
            target_model = (
                self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            )
            self.task.init_train_model(target_model, args)
            self.logger.info("[qPipeline] `init_train_model` hook applied.")
        elif getattr(args, "init_file", None):
            self._load_state_dict_file(args.init_file, description="init")

        if getattr(args, "init_file", None) and getattr(args, "ckp_file", None):
            self.logger.warning(
                "[qPipeline] WARNING: args.init_file and args.ckp_file are both set. "
                "Checkpoint resume in runner will override the init weights."
            )

        ema_params = args.optim.ema_params or qt.qData(ema=False, ema_decay=0.99)
        ema_source_model = (
            self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        )
        self.ema_model = qEMA(ema_source_model, ema_params.ema_decay, torch.device("cpu")) if ema_params.ema else None

        self.loss_fn = self.prepare_loss(args)

    def init_for_test(self):
        args = self.args
        self._ensure_runtime_ready()

        if getattr(args, "ckp_file", None):
            self._load_state_dict_file(args.ckp_file, description="checkpoint")
        elif getattr(args, "init_file", None):
            self._load_state_dict_file(args.init_file, description="init")

    @staticmethod
    @abstractmethod
    def prepare_task(args) -> qTaskBase:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prepare_model(args):
        raise NotImplementedError

    @staticmethod
    def prepare_loss(args):
        loss_fn = entry_utils.prepare_loss(args)
        return loss_fn

    def regist_extra_ckp_caches(self, caches: dict):
        """will be saved in checkpoint"""
        self.extra_ckp_caches.update(caches)

    def regist_middleware(self, middleware):
        if not isinstance(middleware, (list, tuple)):
            middleware = [middleware]

        for m in middleware:
            m(self)

    @staticmethod
    def prepare_env(args):
        if args.ddp_detect:
            qt.qdist.init_distributed_mode(args)
        else:
            args.distributed = False
            args.rank = 0
            args.local_rank = 0

        qt.freeze_rand(args.seed)

        # after distributed init
        prepare_logdir(args)

        args.device = qt.parse_device(args.local_rank)
        if args.device.type == "cuda":
            torch.cuda.set_device(args.device)

    def fit(self, use_profiler=False):
        if self.mode != "train":
            raise RuntimeError("qPipeline.fit() is only available in train mode")

        args = self.args
        max_epochs = args.runner.max_epochs
        max_steps = args.runner.max_steps
        clip_grad = args.runner.clip_grad
        accum_grad = args.runner.get("accum_grad", None)
        distributed = args.distributed
        log_dir = args.log_dir
        print_freq = args.print_freq or 100

        extra_log_keys = None

        run_mode = args.runner.get("run_mode", "epoch")
        eval_interval = args.runner.get("eval_interval", 1)
        save_interval = args.runner.get("save_interval", None)

        return train_runner(
            model=self.model,
            task=self.task,
            loss_fn=self.loss_fn,
            args=args,
            logger=self.logger,
            max_epochs=max_epochs,
            max_steps=max_steps,
            clip_grad=clip_grad,
            distributed=distributed,
            save_dir=log_dir,
            print_freq=print_freq,
            extra_log_keys=extra_log_keys,
            extra_ckp_caches=self.extra_ckp_caches,
            use_profiler=use_profiler,
            ema_model=self.ema_model,
            run_mode=run_mode,
            eval_interval=eval_interval,
            save_interval=save_interval,
            accum_grad=accum_grad,
        )

    def infer(self, dataloader=None):
        if dataloader is None:
            warnings.warn(Warning("[qPipeline]No dataloader provided, use task.test_loader"))
            dataloader = self.task.test_loader
        return self.infer_only(dataloader)

    def infer_only(self, dataloader):
        assert dataloader is not None
        model, task = self.model, self.task
        args = self.args

        distributed = args.distributed

        return infer_runner(
            model=model,
            task=task,
            dataloader=dataloader,
            args=args,
            distributed=distributed,
            logger=self.logger,
        )

    def evaluate_once(self, dataloader=None, prefix: str = "test", return_outputs: bool = False):
        if dataloader is None:
            warnings.warn(Warning("[qPipeline]No dataloader provided, use task.test_loader"))
            dataloader = self.task.test_loader
        assert dataloader is not None

        return evaluate_runner(
            model=self.model,
            task=self.task,
            dataloader=dataloader,
            args=self.args,
            prefix=prefix,
            return_outputs=return_outputs,
            logger=self.logger,
        )
