import gc
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.profiler import ProfilerActivity, profile

import qqtools as qt
from qqtools.config import qtime

from ..entry_utils import qWarmupScheduler
from ..entry_utils.qema import qEMA
from ..qlogger import ConsoleLogger, qLogger
from ..task.qtask import qTaskBase
from .avgbank import AvgBank
from .earlystop import singleTargetEarlyStopper
from .loglistener import LogListener

__all__ = ["epoch_runner"]


class runnerState(qt.qDict):
    epoch: int
    max_epochs: int
    best_epoch: int
    best_train_metric: float
    best_val_metric: float
    best_test_metric: float
    best_ckp_file: Optional[str]


def get_init_state() -> runnerState:
    best_epoch, best_train_metric, best_val_metric, best_test_metric = (0, None, None, None)
    state: runnerState = qt.qDict(
        {
            "epoch": -1,
            "max_epochs": -1,
            "best_epoch": best_epoch,
            "best_train_metric": best_train_metric,
            "best_val_metric": best_val_metric,
            "best_test_metric": best_test_metric,
        },
        allow_notexist=False,
    )
    return state


# @qContextProvider(qt.qDict())
class EpochAgent(object):
    """
    define task interface convention
    do not consider I/O logger events
    """

    def __init__(
        self,
        task: qTaskBase,
        model: torch.nn.Module,
        loss_fn=None,
        optimizer=None,
        clip_grad=None,
        device=None,
        # auxiliary
        ddp: bool = False,
        ema: Optional[Callable] = None,
        use_profiler: bool = False,
        val_and_test: bool = True,
        dtype: torch.dtype = None,
    ):
        self.task = task
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.device = device
        self.ddp = ddp
        self.ema = ema  # or qt.donothing
        self.use_profiler = use_profiler
        self.val_and_test = val_and_test
        self.dtype = dtype

        # listeners
        self.onEpochStartListeners = []
        self.onBatchEndListeners = []
        self.onTrainBatchEndListeners = []
        self.onEpochEndListeners = []

        # inner variables
        self.tr_result = None
        self.val_result = None
        self.te_result = None

        # input check
        if self.device is None:
            warnings.warn("device is not specified, using cpu")
            self.device = torch.device("cpu")

        if use_profiler:
            self.record_function = torch.profiler.record_function
        else:
            self.record_function = qt.nn.DoNothing

    def enumerate_one_epoch(self, epoch):
        task = self.task

        # convention
        train_loader = task.train_loader
        # val_loader = task.val_loader
        # test_loader = task.test_loader
        model: torch.nn.Module = self.model
        ema: Optional[Callable] = self.ema

        # run training
        tr_metrics = self.__run__(train_loader, model, train=True)
        train_metric = task.post_metrics_to_value(tr_metrics)  # convention

        # perform validation and testing if required
        if self.val_and_test:
            val_metric, test_metric, val_metrics, te_metrics = self._evaluate(task, model)
        else:
            val_metric, test_metric = None, None
            val_metrics, te_metrics = {}, {}

        if ema is not None and self.val_and_test:
            # use ema for validation
            orig_device = ema.device
            if orig_device != self.device:
                ema.to(self.device)
            ema_val_metric, ema_test_metric, ema_val_metrics, ema_te_metrics = self._evaluate(task, ema)
            if orig_device != self.device:
                ema.to(orig_device)
        else:
            ema_val_metric, ema_test_metric = None, None
            ema_val_metrics, ema_te_metrics = {}, {}

        # lifetime
        epoch_end_state = {
            "epoch": epoch,
            "train_metric": train_metric,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "train_loss": tr_metrics["loss"],
        }

        if ema is not None:
            epoch_end_state["ema_val_metric"] = ema_val_metric
            epoch_end_state["ema_test_metric"] = ema_test_metric

        avg_metrics = {
            "train": tr_metrics,
            "val": val_metrics,
            "test": te_metrics,
            "ema_val": ema_val_metrics,
            "ema_test": ema_te_metrics,
        }

        # TODO
        # need to support custom metrics
        # maybe add a metric interface
        for listener in self.onEpochEndListeners:
            listener(epoch_end_state, avg_metrics)
        return epoch_end_state

    def _evaluate(self, task: qTaskBase, eval_model: torch.nn.Module):
        val_loader = task.val_loader
        test_loader = task.test_loader

        val_metric, test_metric = None, None
        val_result, te_result = {}, {}

        if val_loader is not None:
            # print("===============validation===============")
            val_result = self.__run__(val_loader, eval_model, train=False)
            val_metric = task.post_metrics_to_value(val_result)

        if test_loader is not None:
            te_result = self.__run__(test_loader, eval_model, train=False)
            test_metric = task.post_metrics_to_value(te_result)

        return val_metric, test_metric, val_result, te_result

    def __run__(self, data_loader, model, train=False):
        if data_loader is None:
            return None

        # core
        task = self.task
        # model: torch.nn.Module = self.model
        ema: Optional[qEMA] = self.ema

        # common auxiliary params
        device = self.device
        ddp = self.ddp
        onEpochStartListeners = self.onEpochStartListeners
        onBatchEndListeners = self.onBatchEndListeners
        onTrainBatchEndListeners = self.onTrainBatchEndListeners
        record_function = self.record_function
        dtype = self.dtype

        # train-only params
        if train:
            loss_fn = self.loss_fn
            optimizer = self.optimizer
            clip_grad = self.clip_grad

        # eval-mode explicity
        if not train:
            _last_mode = model.training
            model.eval()

        # pre
        avgBank = AvgBank(verbose=False)
        num_batches = len(data_loader)
        losses = {}
        batch_metrics = {}
        epoch_cache = qt.qDict(default_function=list)

        # lifecycle
        for listener in onEpochStartListeners:
            listener(num_batches)

        for i, batch_data in enumerate(data_loader):
            batch_metrics.clear()
            start_time = time.perf_counter()
            batch_data = batch_data.to(device)  # convention, batch_data should offer `to(device)`
            if dtype is not None and hasattr(batch_data, "to_dtype"):  # convention
                batch_data = batch_data.to_dtype(dtype)
            batch_data = task.pre_batch_forward(batch_data)  # convention

            # forward
            out = task.batch_forward(model, batch_data)  # convention
            out = task.post_batch_forward(out, batch_data)  # convention

            # metric
            metrics = task.batch_metric(out, batch_data)  # batch_metric convention

            if train:
                with record_function("batch_loss"):
                    losses = task.batch_loss(out, batch_data, loss_fn)  # convention
                    loss_tensor, _ = losses["loss"]
                # backward
                optimizer.zero_grad()
                loss_tensor.backward()
                if clip_grad is not None:
                    clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                if ema is not None:
                    ema.update()

            # hook metric after loss
            # maybe we need to write down some loss value
            if hasattr(task, "extra_batch_metric"):
                extra_metrics = task.extra_batch_metric(out, batch_data, losses, model, epoch_cache)  # convention
                metrics.update(extra_metrics)

            # log
            for loss_k, (_tensor, _cnt) in losses.items():
                avgBank.add(loss_k, _tensor.item(), _cnt)
                batch_metrics[loss_k] = _tensor.item()

            for metric_k, (metric_v, num) in metrics.items():
                if torch.is_tensor(metric_v):
                    metric_v = metric_v.item()
                avgBank.add(metric_k, metric_v, num)
                batch_metrics[metric_k] = metric_v

            # after all metrics, record time
            elapsed_time = time.perf_counter() - start_time
            batch_metrics["BatchTime"] = elapsed_time
            avgBank.add("BatchTime", elapsed_time, 1)

            # state update
            batch_state = {
                "batch_idx": i,
                "num_batches": num_batches,
                "batch_metrics": batch_metrics,  # cur_batch
                "avgBank": avgBank.to_dict(ddp),  # smooth_average
            }
            if train:
                batch_metrics["loss"] = loss_tensor.item()
                batch_state["lr"] = optimizer.param_groups[0]["lr"]
                for listener in onTrainBatchEndListeners:
                    listener(**batch_state)

            # batch end
            for listener in onBatchEndListeners:
                listener(**batch_state)

        # epoch end
        result = avgBank.gather_average(ddp)

        # epoch end metrics
        if "epoch_metric" in task._opt_impl:
            epoch_metrics = task.epoch_metric(epoch_cache)  # convention
            result.update(epoch_metrics)

        # train-mode restore
        if not train:
            model.train(_last_mode)
        return result

    def addBatchEndListener(self, listener):
        self.onBatchEndListeners.append(listener)

    def addTrainBatchEndListener(self, listener):
        self.onTrainBatchEndListeners.append(listener)

    def addEpochStartListener(self, listener):
        self.onEpochStartListeners.append(listener)

    def addEpochEndListener(self, listener):
        self.onEpochEndListeners.append(listener)


class BestCkpManager(object):

    def __init__(self, save_dir, keep_last: bool = False):

        self.save_dir = save_dir
        self.keep_last = keep_last

        self._last_file = None

    def update_best_checkpoint(
        self,
        epoch,
        model,
        optimizer=None,
        lr_scheduler=None,
        early_stopper=None,
        state=None,
        **kwargs,
    ):
        save_dir = self.save_dir
        this_file = f"best_ckp_ep{epoch}.pt"
        _last_file = self._last_file
        keep_last = self.keep_last
        qt.save_ckp(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            early_stop=early_stopper,
            save_dir=save_dir,
            save_file=this_file,
            state=state,
            **kwargs,
        )
        if not keep_last and _last_file is not None:
            last_fp = Path(save_dir, _last_file)
            self.remove_file(last_fp)
        self._last_file = this_file

    def update_latest_checkpoint(
        self,
        model,
        optimizer=None,
        lr_scheduler=None,
        early_stopper=None,
        state=None,
        **kwargs,
    ):
        save_dir = self.save_dir
        file_name = "latest_ckp.pt"
        qt.save_ckp(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            early_stop=early_stopper,
            save_dir=save_dir,
            save_file=file_name,
            datetime=qtime.now_str(),  # datenow_str datetime.datetime.now(),
            state=state,
            **kwargs,
        )

    @staticmethod
    def remove_file(fp):
        # only save checkpoint on main process
        if qt.qdist.get_rank() != 0:
            return
        fp = Path(fp).resolve()
        if fp.exists() and fp.is_file():
            fp.unlink()


def epoch_runner(
    model,
    task: qTaskBase,
    loss_fn,
    optimizer,
    scheduler: qWarmupScheduler,
    args,
    max_epochs,
    clip_grad,
    distributed,
    save_dir,  # save both debug.log, metrics.csv, ckp.pt
    print_freq,
    # optional function
    extra_log_keys=None,
    extra_ckp_caches=None,  # extra information stored with checkpoint
    use_profiler=False,
    ema_model=None,
):
    # input check
    assert isinstance(scheduler, qWarmupScheduler)

    # interface
    _default_log_keys = ["epoch", "train_metric", "val_metric", "test_metric", "train_loss"]
    if ema_model is not None:
        _default_log_keys.extend(["ema_val_metric", "ema_test_metric"])
    log_keys = list(dict.fromkeys(_default_log_keys + (extra_log_keys or [])))
    device = args.device
    earlystop_params = args.runner.early_stop

    # profiler
    if use_profiler:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/maptr_qq"),
            record_shapes=True,
        )
        prof.start()
    else:
        prof = qt.nn.DoNothing()

    # state preparation
    start_epoch = 0
    best_record_state: runnerState = get_init_state()
    best_record_state.max_epochs = max_epochs
    best_record_state.update(extra_ckp_caches or {})

    # logger relative
    if args.ckp_file is not None:
        logger = qLogger(save_dir, columns=log_keys, console=True, recover=True)
    else:
        logger = qLogger(save_dir, columns=log_keys, console=True)

    # early stopper
    allowed_metrics = ["val_metric", "test_metric", "train_loss"]
    assert (
        earlystop_params.target in allowed_metrics
    ), f"must take one in {allowed_metrics}. other metrics not supported now"
    earlyStopper = singleTargetEarlyStopper(
        earlystop_params.target,
        earlystop_params.patience,
        earlystop_params.mode,
        earlystop_params.min_delta,
    )

    ckpManageer = BestCkpManager(save_dir, keep_last=False)

    if args.init_file is not None:
        logger.info(f"Initializing model from: {args.init_file}")
        ckp_dict = qt.recover(
            model,
            ckp_file=args.init_file,
            weights_only=False,
            strict=False,
        )

        if ema_model is not None and "ema_state_dict" in ckp_dict and ckp_dict["ema_state_dict"] is not None:
            ema_model.load_state_dict(ckp_dict["ema_state_dict"])
            logger.info(f"EMA model weights loaded from `ema_state_dict`.")

        logger.info(f"Initialization Complete.\n")

    if args.ckp_file is not None:
        logger.info(f"recovering from: {args.ckp_file}")
        ckp_dict = qt.recover(
            model,
            optimizer,
            scheduler,
            earlyStopper,
            ckp_file=args.ckp_file,
            weights_only=False,
            strict=False,
        )
        best_record_state.epoch = ckp_dict["state"].epoch
        start_epoch = best_record_state.epoch + 1
        logger.info(f'after load ckp, lr :{optimizer.param_groups[0]["lr"]}\n')

        if "lrscheduler_state_dict" in ckp_dict:
            scheduler.step()
        else:
            warnings.warn(
                "Learning rate scheduler state dict not found in checkpoint. "
                "Will proceed with initial scheduler state. "
                "Please ensure this is intentional - if resuming training, ",
                category=UserWarning,
                stacklevel=2,
            )

        if "earlystop_state_dict" in ckp_dict:
            earlyStopper.load_state_dict(ckp_dict["earlystop_state_dict"])

        if "ema_state_dict" in ckp_dict and ckp_dict["ema_state_dict"] is not None:
            ema_model.load_state_dict(ckp_dict["ema_state_dict"])

        if "task_state_dict" in ckp_dict and ckp_dict["task_state_dict"] is not None:
            if "load_state_dict" in task._opt_impl:
                task.load_state_dict(model, ckp_dict["task_state_dict"])

    """mainstream"""
    epoch_agent = EpochAgent(
        task,
        model,
        loss_fn,
        optimizer,
        clip_grad=clip_grad,
        device=device,
        ddp=distributed,
        ema=ema_model,
        use_profiler=use_profiler,
        val_and_test=True,
    )
    logLisitener = LogListener("train", logger, print_freq, epoch_state=best_record_state)

    if args.rank == 0:
        epoch_agent.addEpochStartListener(logLisitener.onEpochStart)
        epoch_agent.addEpochEndListener(logLisitener.onEpochEnd)
        epoch_agent.addBatchEndListener(logLisitener.onBatchEnd)
        epoch_agent.addTrainBatchEndListener(scheduler.onTrainBatchEnd)

    if "onEpochStart" in task._opt_impl:
        epoch_agent.addEpochStartListener(task.onEpochStart)
    if "onEpochEnd" in task._opt_impl:
        epoch_agent.addEpochEndListener(task.onEpochEnd)
    if "onBatchEnd" in task._opt_impl:
        epoch_agent.addBatchEndListener(task.onBatchEnd)

    # Main Loop
    for epoch in range(start_epoch, max_epochs):
        # pre
        best_record_state.epoch = epoch
        epoch_start_time = time.perf_counter()

        if distributed:
            task.train_loader.sampler.set_epoch(epoch)

        # core result
        epoch_result = epoch_agent.enumerate_one_epoch(epoch)
        train_metric = epoch_result["train_metric"]
        val_metric = epoch_result["val_metric"]
        test_metric = epoch_result["test_metric"]
        current_epoch_state = qt.qDict(
            {
                "train_metric": train_metric,
                "val_metric": val_metric,
                "test_metric": test_metric,
                "epoch": epoch,
            }
        )

        # handle ema
        ema_state_dict = ema_model.state_dict() if ema_model is not None else None

        # handle task
        task_state_dict = None
        if "state_dict" in task._opt_impl:
            task_state_dict = task.state_dict(model)

        gc.collect()

        # epoch end
        if scheduler is not None:
            if scheduler._is_plateau:
                scheduler.step_epoch(metrics=val_metric)
            else:
                scheduler.step_epoch()

        # update state
        if earlyStopper.is_better(earlystop_params.target, val_metric, best_record_state.best_val_metric):
            best_record_state.best_epoch = epoch
            best_record_state.best_train_metric = train_metric
            best_record_state.best_val_metric = val_metric
            best_record_state.best_test_metric = test_metric if test_metric is not None else float("nan")

            # save best
            ckpManageer.update_best_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                earlyStopper,
                state=best_record_state,
                ema_state_dict=ema_state_dict,
                task_state_dict=task_state_dict,
            )

            if "on_better_model" in task._opt_impl:
                logger.info("[Runner] better model found")
                # in case we only do inference when model is better
                task.on_better_model(epoch_agent, current_epoch_state)  # convention

        # save latest
        ckpManageer.update_latest_checkpoint(
            model,
            optimizer,
            scheduler,
            earlyStopper,
            state=best_record_state,
            ema_state_dict=ema_state_dict,
            task_state_dict=task_state_dict,
        )

        # message
        info_str = "Epoch: [{epoch}] train Metric: {train_mae:.5f}, ".format(
            epoch=epoch, train_mae=current_epoch_state.train_metric
        )
        val_metric = current_epoch_state.val_metric if current_epoch_state.val_metric is not None else float("nan")
        test_metric = current_epoch_state.test_metric if current_epoch_state.test_metric is not None else float("nan")
        info_str += "val Metric: {:.5f}, ".format(val_metric)
        info_str += "test Metric: {:.5f}, ".format(test_metric)
        info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
        info_str += " lr: {:.8f}".format(optimizer.param_groups[0]["lr"])
        logger.info(info_str)

        # state
        info_str = (
            "Best -- epoch={}, train Metric: {:.5f}, val Metric: {:.5f}, test Metric: {:.5f}\nLog_dir: {}".format(
                best_record_state.best_epoch,
                best_record_state.best_train_metric,
                best_record_state.best_val_metric,
                best_record_state.best_test_metric,
                save_dir,
            )
        )
        logger.info(info_str)

        # early stop
        stop, stop_msg, debug_msg = earlyStopper(epoch_result)
        if debug_msg is not None:
            logger.info(debug_msg)
        if stop:
            logger.info(stop_msg)
            return

    # at the very end
    prof.stop()
    if isinstance(prof, profile):
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return


def infer_only_runner(model, task, dataloader, args, ddp, print_freq):
    """"""
    device = args.device

    """logger"""
    logger = ConsoleLogger(None, logger_name="infer")
    # loglisitener = LogListener("val", logger, print_freq)

    if args.ckp_file is not None:
        logger.info(f"recovering from: {args.ckp_file}")
        qt.recover(model, ckp_file=args.ckp_file, weights_only=False, strict=False)

    """mainstream"""
    epoch_agent: EpochAgent = EpochAgent(task, model, device=device, ddp=ddp)
    # epoch_agent.addBatchEndListener(loglisitener.onBatchEnd)
    # epoch_agent.addEpochEndListener(loglisitener.onEpochEnd)

    epoch_start_time = time.perf_counter()
    # core
    # with torch.no_grad():
    result = epoch_agent.__run__(dataloader, model, train=False)

    epoch_end_time = time.perf_counter()
    elapsed_time = f"{epoch_end_time - epoch_start_time:.4f}"

    for k, v in result.items():
        print(f"{k}: {v}")
