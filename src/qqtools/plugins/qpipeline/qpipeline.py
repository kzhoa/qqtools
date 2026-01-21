import functools
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import qqtools as qt
import torch
import yaml

from . import entry_utils
from .entry_utils.qema import qEMA
from .runner import epoch_runner, infer_only_runner
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
        yaml.dump(args.to_dict(), open(args.config_file, "w"))

    elif args.ckp_file is not None:
        assert args.log_dir is not None
        args.config_file = str(Path(args["log_dir"], "config_ckprecover.yaml"))
        yaml.dump(args.to_dict(), open(args.config_file, "w"))


def rank_zero_only(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if qt.qdist.get_rank() == 0:
            return fn(*args, **kwargs)
        else:
            return None

    return wrapper


@rank_zero_only
def main_print(*args, **kwargs):
    rank = qt.qdist.get_rank()
    print(f"[qPipeline:{rank}]", *args, **kwargs)


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
        train: bool = False,
        task: Optional[qTaskBase] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.args = args
        self.train = train
        self.task = task
        self.model = model

        # convention
        if hasattr(task, "pipe_middle_ware"):
            task.pipe_middle_ware(self)

        if train:
            self.init_for_train()

    def init_for_train(self):
        args = self.args
        task = self.task
        model = self.model

        self.prepare_env(args)
        main_print("[qPipeline] use args:\n", args)

        if task is not None:
            self.task = task
        else:
            self.task = self.prepare_task(args)

        if model is None:
            model = self.prepare_model(args)

        # handle ema
        ema_params = args.optim.ema_params or qt.qData(ema=False, ema_decay=0.99)
        if ema_params.ema is True:
            ema_model = qEMA(model, ema_params.ema_decay, torch.device("cpu"))
        else:
            ema_model = None
        self.ema_model = ema_model

        if not args.distributed:
            self.model = model.to(args.device)
            main_print(f"model moved to {args.device}")
        elif not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.to(args.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                find_unused_parameters=False,
            )
            main_print(f"model with DDP wrapped to {args.local_rank}")
        else:
            self.model = model
            main_print(f"model with DDP already wrapped on {next(model.parameters()).device}")

        # convention: task device check
        if hasattr(self.task, "to"):
            self.task.to(args.device)

        # train-only
        if self.train:
            self.optimizer, self.scheduler = self.prepare_optim(args, self.model)
            self.loss_fn = self.prepare_loss(args)

        # other declarations
        self.extra_ckp_caches = {}

    @staticmethod
    @abstractmethod
    def prepare_task(args) -> qTaskBase:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prepare_model(args):
        raise NotImplementedError

    @staticmethod
    def prepare_optim(args, model):
        optimizer, scheduler = entry_utils.prepare_optim(args, model)
        return optimizer, scheduler

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
        torch.cuda.set_device(args.device)
        main_print(f"Set device to: {args.device}")

    def fit(self, use_profiler=False):
        model, task, loss_fn, optimizer, scheduler, extra_ckp_caches = (
            self.model,
            self.task,
            self.loss_fn,
            self.optimizer,
            self.scheduler,
            self.extra_ckp_caches,
        )

        # unpack
        args = self.args
        max_epochs = args.runner.epochs
        clip_grad = args.runner.clip_grad
        distributed = args.distributed
        log_dir = args.log_dir
        print_freq = args.print_freq or 100

        extra_log_keys = None

        epoch_runner(
            model,
            task,
            loss_fn,
            optimizer,
            scheduler,
            args,
            max_epochs,
            clip_grad,
            distributed,
            save_dir=log_dir,
            print_freq=print_freq,
            extra_log_keys=extra_log_keys,
            extra_ckp_caches=extra_ckp_caches,
            use_profiler=use_profiler,
            ema_model=self.ema_model,
        )

    def infer(self, dataloader=None):
        if dataloader is None:
            warnings.warn(Warning("[qPipeline]No dataloader provided, use task.test_loader"))
            dataloader = self.task.test_loader
        self.infer_only(dataloader)

    def infer_only(self, dataloader):
        assert dataloader is not None
        model, task = self.model, self.task
        args = self.args

        distributed = args.distributed
        print_freq = args.print_freq or 100
        infer_only_runner(
            model,
            task,
            dataloader,
            args,
            distributed,
            print_freq,
        )
