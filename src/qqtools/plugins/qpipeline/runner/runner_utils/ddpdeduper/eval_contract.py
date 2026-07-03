from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist

import qqtools as qt


def _all_gather_object(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]

    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _infer_tail_shape_and_dtype(tensor_list: list[torch.Tensor]) -> tuple[tuple[int, ...], str]:
    local_cat = torch.cat(tensor_list, dim=0)
    return tuple(local_cat.shape[1:]), str(local_cat.dtype)


@dataclass(frozen=True)
class EvalBatchControl:
    all_duplicate: bool = False
    real_logical_sample_ids: Sequence[Any] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalBatch:
    payload: Any
    control: EvalBatchControl


@dataclass(frozen=True)
class EvalDedupRuntime:
    enabled: bool = False
    is_graph: bool = False
    node_aligned_output_keys: Sequence[str] = field(default_factory=tuple)
    expected_real_sample_count: Optional[int] = None

    def gather_avg_bank(self, avg_bank: Any, distributed: bool, device: torch.device) -> dict[str, Any]:
        if not distributed or not qt.qdist.is_dist_available_and_initialized():
            return avg_bank.gather_average(False)

        local_stats = {
            key: {
                "sum": float(meter.sum),
                "count": float(meter.count),
            }
            for key, meter in avg_bank.avgMeters.items()
        }
        gathered_keys = _all_gather_object(sorted(local_stats.keys()))
        global_keys = sorted({key for keys in gathered_keys for key in keys})

        result: dict[str, Any] = {}
        for key in global_keys:
            local_entry = local_stats.get(key, {"sum": 0.0, "count": 0.0})
            local_tensor = torch.tensor(
                [local_entry["sum"], local_entry["count"]],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
            global_sum, global_count = local_tensor.tolist()
            result[key] = (global_sum / global_count) if global_count > 0 else 0.0
        return result

    def gather_tensor_bank(
        self,
        tensor_bank: Any,
        distributed: bool,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        local_bank = tensor_bank.bank
        if not distributed or not qt.qdist.is_dist_available_and_initialized():
            return tensor_bank.gather(False, device)

        local_schema = {}
        local_cat_map = {}
        for key, tensor_list in local_bank.items():
            if not tensor_list:
                continue
            local_cat = torch.cat(tensor_list, dim=0)
            local_cat_map[key] = local_cat
            local_schema[key] = {
                "tail_shape": tuple(local_cat.shape[1:]),
                "dtype": str(local_cat.dtype),
            }

        gathered_schema = _all_gather_object(local_schema)
        global_keys = sorted({key for schema in gathered_schema for key in schema.keys()})
        result: dict[str, torch.Tensor] = {}

        for key in global_keys:
            schema_candidates = [schema[key] for schema in gathered_schema if key in schema]
            if not schema_candidates:
                continue
            reference = schema_candidates[0]
            for candidate in schema_candidates[1:]:
                if candidate != reference:
                    raise ValueError(
                        f"Dedup gather schema mismatch for key '{key}': "
                        f"{reference} vs {candidate}"
                    )

            if key in local_cat_map:
                local_tensor = local_cat_map[key]
            else:
                local_tensor = torch.empty(
                    (0, *reference["tail_shape"]),
                    dtype=getattr(torch, reference["dtype"].split(".")[-1]),
                )

            result[key] = qt.qdist.all_gather_tensor(local_tensor, device).cpu()

        tensor_bank.reset()
        return result

    def gather_output_bank(
        self,
        output_bank: Any,
        distributed: bool,
        device: torch.device,
    ) -> Optional[dict[str, torch.Tensor]]:
        gathered_outputs = self.gather_tensor_bank(output_bank, distributed, device)
        if distributed and qt.qdist.get_rank() != 0:
            return None
        self.validate_outputs(gathered_outputs)
        return gathered_outputs

    def validate_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        if not self.enabled or not self.is_graph:
            return
        if self.expected_real_sample_count is None:
            raise ValueError("Graph eval output validation requires expected_real_sample_count.")

        expected_count = int(self.expected_real_sample_count)
        node_aligned_keys = set(self.node_aligned_output_keys)
        for key, value in outputs.items():
            if key in node_aligned_keys:
                continue
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Graph eval output '{key}' must be a torch.Tensor, got {type(value).__name__}.")
            if value.dim() == 0:
                raise ValueError(
                    f"Graph eval output '{key}' is undeclared and scalar-shaped; "
                    "undeclared graph outputs must be sample-aligned tensors."
                )
            if value.shape[0] != expected_count:
                raise ValueError(
                    f"Graph eval output '{key}' is undeclared but has leading dimension {value.shape[0]}, "
                    f"expected sample-aligned leading dimension {expected_count}. "
                    "Declare node-aligned outputs explicitly in task.eval.ddp_dedup.node_aligned_output_keys."
                )


def unwrap_eval_batch(batch_data: Any) -> tuple[Any, EvalBatchControl]:
    if isinstance(batch_data, EvalBatch):
        return batch_data.payload, batch_data.control
    return batch_data, EvalBatchControl()
