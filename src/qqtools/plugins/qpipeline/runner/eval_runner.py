from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import qqtools as qt

from ..entry_utils.type_qconfig import qConfig
from ..qlogger import ConsoleLogger
from ..task.qtask import qTaskBase
from .runner_utils.avgbank import AvgBank
from .runner_utils.common import _getattr_or_default, move_batch_to_device
from .runner_utils.ddpdeduper import EvalDedupRuntime, unwrap_eval_batch, wrap_eval_dataloader_for_ddp_dedup
from .runner_utils.tensorbank import TensorBank

__all__ = ["evaluate_runner", "infer_runner"]


def normalize_output_dict(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return dict(out)
    return {"preds": out}


def extract_auxiliary_output_fields(batch_data: Any) -> Dict[str, Any]:
    labels = None
    if isinstance(batch_data, dict) and "y" in batch_data:
        labels = batch_data["y"]
    elif hasattr(batch_data, "y"):
        labels = batch_data.y

    return {"labels": labels} if labels is not None else {}


def _resolve_eval_dedup_meta(args: Any) -> tuple[bool, bool, tuple[str, ...]]:
    task_cfg = _getattr_or_default(args, "task")
    eval_cfg = _getattr_or_default(task_cfg, "eval")
    dedup_cfg = _getattr_or_default(eval_cfg, "ddp_dedup")

    enabled = bool(_getattr_or_default(dedup_cfg, "enabled", True))
    is_graph = bool(_getattr_or_default(dedup_cfg, "is_graph", False))
    node_aligned_output_keys = tuple(_getattr_or_default(dedup_cfg, "node_aligned_output_keys", tuple()))
    return enabled, is_graph, node_aligned_output_keys


def _prepare_eval_loader(dataloader: DataLoader, args: Any):
    distributed = bool(_getattr_or_default(args, "distributed", False))
    enabled, is_graph, node_aligned_output_keys = _resolve_eval_dedup_meta(args)
    if not distributed:
        return dataloader
    return wrap_eval_dataloader_for_ddp_dedup(
        dataloader,
        enabled=enabled,
        is_graph=is_graph,
        node_aligned_output_keys=node_aligned_output_keys,
    )


def _build_output_bank_result(
    *,
    output_bank: TensorBank,
    distributed: bool,
    device: torch.device,
    runtime: Optional[EvalDedupRuntime],
) -> Optional[Dict[str, Any]]:
    if runtime is not None:
        return runtime.gather_output_bank(output_bank, distributed, device)

    gathered_outputs = output_bank.gather(distributed, device)
    if distributed and qt.qdist.get_rank() != 0:
        return None
    return gathered_outputs


def _get_runtime(loader: DataLoader) -> Optional[EvalDedupRuntime]:
    return getattr(loader, "dedup_runtime", None)


def _build_logger(rank: int, logger):
    if logger is not None:
        return logger
    return ConsoleLogger(rank=rank) if rank == 0 else None


def evaluate_runner(
    model: nn.Module,
    task: qTaskBase,
    dataloader: DataLoader,
    args: Optional[qConfig] = None,
    prefix: str = "test",
    return_outputs: bool = False,
    allow_auto_offload: bool = False,
    logger=None,
) -> Optional[Dict[str, Any]]:
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")
    del allow_auto_offload

    dataloader = _prepare_eval_loader(dataloader, args)
    runtime = _get_runtime(dataloader)
    device = _getattr_or_default(args, "device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank = _getattr_or_default(args, "rank", 0)
    distributed = _getattr_or_default(args, "distributed", False)
    logger = _build_logger(rank, logger)

    model.to(device)
    was_training = model.training
    model.eval()

    eval_avg_bank = AvgBank()
    has_batch_cache = task.has_implemented("batch_cache")
    has_epoch_metric = task.has_implemented("epoch_metric")
    use_tensor_bank_for_eval = has_batch_cache and has_epoch_metric
    eval_tensor_bank = TensorBank(logger=logger) if use_tensor_bank_for_eval else None
    eval_output_bank = TensorBank(logger=logger) if return_outputs else None

    try:
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(dataloader):
                batch_data, control = unwrap_eval_batch(raw_batch)
                batch_data = move_batch_to_device(batch_data, device)
                batch_data = task.pre_batch_forward(batch_data)
                out = task.batch_forward(model, batch_data)
                out = task.post_batch_forward(out, batch_data)

                if control.all_duplicate:
                    continue

                raw_batch_metrics = task.batch_metric(out, batch_data)
                eval_avg_bank.update_from_dict(raw_batch_metrics)
                if eval_tensor_bank:
                    eval_tensor_bank.add(task.batch_cache(out, batch_data))
                if eval_output_bank is not None:
                    batch_outputs = normalize_output_dict(out)
                    batch_outputs.update(extract_auxiliary_output_fields(batch_data))
                    eval_output_bank.add(batch_outputs)

                if logger and (batch_idx + 1) % max(_getattr_or_default(args, "print_freq", 10), 1) == 0:
                    logger.info(f"Evaluation progress: {batch_idx + 1} batches processed")

        if runtime is not None:
            avg_metrics = runtime.gather_avg_bank(eval_avg_bank, distributed, device)
        else:
            avg_metrics = eval_avg_bank.gather_average(distributed)
        if use_tensor_bank_for_eval and eval_tensor_bank:
            if runtime is not None:
                gathered_cache = runtime.gather_tensor_bank(eval_tensor_bank, distributed, device)
            else:
                gathered_cache = eval_tensor_bank.gather(distributed, device)
            task_epoch_metrics = task.epoch_metric(gathered_cache)
            if task_epoch_metrics:
                avg_metrics.update(task_epoch_metrics)

        prefixed_metrics = {f"{prefix}_{key}": value for key, value in avg_metrics.items()}
        prefixed_metrics[f"{prefix}_metric"] = task.post_metrics_to_value(avg_metrics)
        if eval_output_bank is None:
            return prefixed_metrics

        gathered_outputs = _build_output_bank_result(
            output_bank=eval_output_bank,
            distributed=distributed,
            device=device,
            runtime=runtime,
        )
        if gathered_outputs is None:
            return None
        prefixed_metrics.update(gathered_outputs)
        return prefixed_metrics
    finally:
        model.train(was_training)


def infer_runner(
    model: nn.Module,
    task: qTaskBase,
    dataloader: DataLoader,
    args: Optional[Any] = None,
    distributed: bool = False,
    logger=None,
) -> Optional[Dict[str, Any]]:
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")

    dataloader = _prepare_eval_loader(dataloader, args)
    runtime = _get_runtime(dataloader)
    device = _getattr_or_default(args, "device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank = _getattr_or_default(args, "rank", 0)
    distributed = bool(distributed or _getattr_or_default(args, "distributed", False))
    render_type = _getattr_or_default(args, "render_type", "auto")
    logger = _build_logger(rank, logger)

    model.to(device)
    model.eval()
    output_bank = TensorBank(logger=logger)

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    use_tqdm = has_tqdm and render_type != "plain"
    use_simple_logging = render_type != "plain" and not use_tqdm
    progress_iter = tqdm(dataloader, desc="Inference", dynamic_ncols=True) if use_tqdm else dataloader

    try:
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(progress_iter):
                batch_data, control = unwrap_eval_batch(raw_batch)
                batch_data = move_batch_to_device(batch_data, device)
                batch_data = task.pre_batch_forward(batch_data)
                out = task.batch_forward(model, batch_data)
                out = task.post_batch_forward(out, batch_data)

                if control.all_duplicate:
                    continue

                batch_results = normalize_output_dict(out)
                batch_results.update(extract_auxiliary_output_fields(batch_data))
                output_bank.add(batch_results)

                if use_simple_logging and (batch_idx + 1) % 10 == 0 and logger:
                    logger.info(f"Inference progress: {batch_idx + 1} batches processed")
    except Exception as e:
        if logger:
            logger.error(f"Error during inference: {e}")
        raise

    return _build_output_bank_result(
        output_bank=output_bank,
        distributed=distributed,
        device=device,
        runtime=runtime,
    )
