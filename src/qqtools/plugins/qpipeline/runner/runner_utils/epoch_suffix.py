"""Epoch-suffix auto-compute resolver for qpipeline step-mode fields.

Converts config values like "0.5epoch" or "5epoch" into concrete optimizer step counts
based on len(train_loader) and accum_grad.
"""

import re
import warnings as _warnings
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

__all__ = ["EpochSuffixResolver", "standardize_epoch_suffixes"]

_EPOCH_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)epoch$")

_SCHEDULER_FIELDS = ("T_max", "step_size", "T_0")
_SCHEDULER_LIST_FIELDS = ("milestones",)
_RUNNER_FIELDS = ("eval_interval", "save_interval")


def _parse_epoch_suffix(value: Any) -> Optional[float]:
    """Extract the numeric coefficient from an epoch-suffix string.

    Returns None if the value does not use epoch-suffix syntax.
    """
    if not isinstance(value, str):
        return None
    m = _EPOCH_PATTERN.match(value.strip())
    if m is None:
        return None
    return float(m.group(1))


class EpochSuffixResolver:
    """One-shot resolver that converts epoch-suffix config values to optimizer step counts.

    Lifecycle: instantiate before scheduler construction, call resolve() once,
    then discard. Does not participate in the training loop.
    """

    def __init__(
        self,
        run_mode: str,
        step_on: Optional[str],
        train_loader_length: int,
        accum_grad: Optional[int] = None,
        distributed: bool = False,
        scheduler_name: Optional[str] = None,
    ):
        """
        Args:
            run_mode: "epoch" or "step".
            step_on: Scheduler stepping trigger. None means unresolved (will be
                     inferred from scheduler type by downstream code). For the
                     purpose of epoch-suffix validation, None is treated as
                     "optimizer_step" (the default for non-plateau schedulers).
            train_loader_length: len(task.train_loader) for the current rank.
            accum_grad: Gradient accumulation factor. None or <1 falls back to 1.
            distributed: Whether DDP is active.
            scheduler_name: Scheduler type name (e.g. "cosine", "plateau").
                           Used to infer step_on when step_on is None.
        """
        if not isinstance(train_loader_length, int) or train_loader_length <= 0:
            raise ValueError(
                "epoch-suffix auto-compute requires len(task.train_loader) to be a positive integer, "
                f"got {train_loader_length!r}. Provide explicit integer values or ensure "
                "train_loader supports __len__."
            )

        self.run_mode = run_mode.lower() if isinstance(run_mode, str) else run_mode
        self.step_on = self._resolve_effective_step_on(step_on, scheduler_name)
        self.train_loader_length = train_loader_length
        self.effective_accum_grad = accum_grad if accum_grad and accum_grad >= 1 else 1
        self.distributed = distributed

        self.steps_per_epoch = ceil(train_loader_length / self.effective_accum_grad)

        self._resolved: Dict[str, Tuple[str, int]] = {}
        self._warmup_resolved = False
        self._scheduler_resolved_fields: List[str] = []
        self._warnings: List[str] = []
        self._logs: List[str] = []

    @staticmethod
    def _resolve_effective_step_on(step_on: Optional[str], scheduler_name: Optional[str]) -> str:
        if step_on is not None:
            return step_on
        if scheduler_name and scheduler_name.lower() == "plateau":
            return "valid_end"
        return "optimizer_step"

    def resolve(self, args) -> "ResolveResult":
        """Parse all epoch-suffix fields in args, rewrite to integers, return result."""
        self._check_ddp_consistency()
        self._resolve_scheduler_params(args)
        self._resolve_warmup(args)
        self._resolve_runner_fields(args)
        self._check_warmup_coupling()
        self._build_summary_log()
        return ResolveResult(
            logs=self._logs,
            warnings=self._warnings,
            resolved_fields=dict(self._resolved),
            steps_per_epoch=self.steps_per_epoch,
        )

    # ─── DDP consistency ───────────────────────────────────────────────

    def _check_ddp_consistency(self) -> None:
        if not self.distributed:
            return
        if not dist.is_initialized():
            return

        local_length = torch.tensor([self.train_loader_length], dtype=torch.int64)
        if torch.cuda.is_available():
            local_length = local_length.cuda()

        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(local_length) for _ in range(world_size)]
        dist.all_gather(gathered, local_length)

        lengths = [int(t.item()) for t in gathered]
        if len(set(lengths)) > 1:
            rank_details = "\n".join(f"  rank {i}: len(train_loader) = {l}" for i, l in enumerate(lengths))
            raise RuntimeError(
                f"[qpipeline] epoch-suffix auto-compute requires all ranks to have "
                f"identical len(train_loader), but detected inconsistency:\n"
                f"{rank_details}\n"
                f"Because epoch-suffix syntax (e.g. \"T_max: 0.5epoch\") converts to optimizer steps "
                f"based on len(train_loader), inconsistent values across ranks would cause each rank "
                f"to compute different step counts, leading to divergent training behavior.\n"
                f"To fix: ensure your dataset + sampler produce equal batch counts on all ranks, "
                f"or use explicit integer values instead of epoch-suffix syntax."
            )

    # ─── Scheduler params ──────────────────────────────────────────────

    def _resolve_scheduler_params(self, args) -> None:
        scheduler_params = self._get_scheduler_params(args)
        if scheduler_params is None:
            return

        for field in _SCHEDULER_FIELDS:
            raw = self._get_field(scheduler_params, field)
            if raw is None:
                continue
            coeff = _parse_epoch_suffix(raw)
            if coeff is None:
                continue
            resolved = self._convert_scheduler_field(field, coeff)
            self._set_field(scheduler_params, field, resolved)
            self._record(f"scheduler_params.{field}", raw, resolved)
            self._scheduler_resolved_fields.append(field)

        for field in _SCHEDULER_LIST_FIELDS:
            raw_list = self._get_field(scheduler_params, field)
            if raw_list is None or not isinstance(raw_list, (list, tuple)):
                continue
            new_list = []
            any_resolved = False
            for i, item in enumerate(raw_list):
                coeff = _parse_epoch_suffix(item)
                if coeff is not None:
                    resolved = self._convert_scheduler_field(f"{field}[{i}]", coeff)
                    new_list.append(resolved)
                    any_resolved = True
                else:
                    new_list.append(item if not isinstance(item, str) else int(item))
            if any_resolved:
                self._set_field(scheduler_params, field, new_list)
                self._record(f"scheduler_params.{field}", raw_list, new_list)
                self._scheduler_resolved_fields.append(field)

    def _convert_scheduler_field(self, field_name: str, coeff: float) -> int:
        self._validate_coeff(coeff, f"scheduler_params.{field_name}")

        if self.step_on == "valid_end":
            raise ValueError(
                f"epoch-suffix auto-compute is not supported for scheduler_params.{field_name} "
                f"when step_on='valid_end'. In this mode scheduler parameters represent the number "
                f"of validation triggers, not optimizer steps. Please provide an explicit integer value."
            )

        if self.run_mode == "epoch":
            return self._epoch_mode_scheduler_convert(coeff, f"scheduler_params.{field_name}")

        return max(1, int(coeff * self.steps_per_epoch))

    def _epoch_mode_scheduler_convert(self, coeff: float, field_path: str) -> int:
        """In epoch mode, scheduler fields with epoch suffix are still converted to steps
        (since scheduler always counts optimizer steps), but only when step_on=optimizer_step.
        """
        # step_on=valid_end is already rejected upstream
        return max(1, int(coeff * self.steps_per_epoch))

    # ─── Warmup ────────────────────────────────────────────────────────

    def _resolve_warmup(self, args) -> None:
        warmup_params = self._get_warmup_params(args)
        if warmup_params is None:
            return

        raw = self._get_field(warmup_params, "warmup_steps")
        if raw is None:
            return
        coeff = _parse_epoch_suffix(raw)
        if coeff is None:
            return

        self._validate_coeff(coeff, "warmup.warmup_steps")
        resolved = max(1, int(coeff * self.steps_per_epoch))
        self._set_field(warmup_params, "warmup_steps", resolved)
        self._record("warmup.warmup_steps", raw, resolved)
        self._warmup_resolved = True

    # ─── Runner fields ─────────────────────────────────────────────────

    def _resolve_runner_fields(self, args) -> None:
        runner = self._get_runner(args)
        if runner is None:
            return

        for field in _RUNNER_FIELDS:
            raw = self._get_field(runner, field)
            if raw is None:
                continue
            coeff = _parse_epoch_suffix(raw)
            if coeff is None:
                continue
            resolved = self._convert_runner_field(field, coeff)
            self._set_field(runner, field, resolved)
            self._record(f"runner.{field}", raw, resolved)

    def _convert_runner_field(self, field_name: str, coeff: float) -> int:
        self._validate_coeff(coeff, field_name)

        if self.run_mode == "epoch":
            raise ValueError(
                f"epoch-suffix is not supported for {field_name} when run_mode='epoch'. "
                f"These fields already use epoch as their unit. "
                f"Please provide a plain integer value."
            )

        return max(1, int(coeff * self.steps_per_epoch))

    # ─── Warmup coupling check ────────────────────────────────────────

    def _check_warmup_coupling(self) -> None:
        if not self._warmup_resolved:
            return
        if not self._scheduler_resolved_fields:
            return

        affected = ", ".join(self._scheduler_resolved_fields)
        self._warnings.append(
            f"[qpipeline] Warning: Both warmup_steps and scheduler_params ({affected}) "
            f"use epoch-suffix. Note that scheduler parameter counting begins AFTER warmup "
            f"completes (main scheduler is not stepped during warmup). "
            f"Effective main-scheduler budget = total_steps - warmup_steps. "
            f"If you intend the scheduler period to cover the entire post-warmup phase, "
            f"subtract your warmup duration from the scheduler parameter."
        )

    # ─── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_coeff(coeff: float, field_path: str) -> None:
        if coeff <= 0:
            raise ValueError(
                f"epoch-suffix coefficient must be positive for {field_path}, got {coeff}."
            )

    def _record(self, field_path: str, raw: Any, resolved: int) -> None:
        self._resolved[field_path] = (str(raw), resolved)

    def _build_summary_log(self) -> None:
        if not self._resolved:
            return
        header = (
            f"[epoch-suffix] len_loader={self.train_loader_length}, "
            f"accum_grad={self.effective_accum_grad}, "
            f"steps_per_epoch={self.steps_per_epoch}"
        )
        lines = [header]
        for field_path, (raw, resolved) in self._resolved.items():
            lines.append(f"  {field_path}: {raw} -> {resolved}")
        self._logs.append("\n".join(lines))

    @staticmethod
    def _get_scheduler_params(args) -> Optional[Any]:
        optim = getattr(args, "optim", None)
        if optim is None:
            return None
        return getattr(optim, "scheduler_params", None)

    @staticmethod
    def _get_warmup_params(args) -> Optional[Any]:
        optim = getattr(args, "optim", None)
        if optim is None:
            return None
        return getattr(optim, "warmup_params", None)

    @staticmethod
    def _get_runner(args) -> Optional[Any]:
        return getattr(args, "runner", None)

    @staticmethod
    def _get_field(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @staticmethod
    def _set_field(obj: Any, key: str, value: Any) -> None:
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)


class ResolveResult:
    """Immutable result returned by EpochSuffixResolver.resolve()."""

    __slots__ = ("logs", "warnings", "resolved_fields", "steps_per_epoch")

    def __init__(
        self,
        logs: List[str],
        warnings: List[str],
        resolved_fields: Dict[str, Tuple[str, int]],
        steps_per_epoch: int,
    ):
        self.logs = logs
        self.warnings = warnings
        self.resolved_fields = resolved_fields
        self.steps_per_epoch = steps_per_epoch

    @property
    def has_resolved(self) -> bool:
        return len(self.resolved_fields) > 0


def standardize_epoch_suffixes(
    args: Any,
    task: Any,
    accum_grad: Optional[int],
    distributed: bool,
    logger: Any,
) -> None:
    """Resolve all epoch-suffix config values in args as part of config standardization.

    This is the public entry point called by train_runner. It extracts the necessary
    context from args and task, constructs an EpochSuffixResolver, and applies
    the resolution. Logs are emitted to the provided logger.

    Args:
        args: The full config object (qDict or Namespace).
        task: Task instance with a train_loader attribute.
        accum_grad: Gradient accumulation factor.
        distributed: Whether DDP is active.
        logger: Logger instance supporting .info() and .warning().
    """
    try:
        train_loader_length = len(task.train_loader)
    except (AttributeError, TypeError):
        train_loader_length = None

    if train_loader_length is None or train_loader_length <= 0:
        return

    optim_cfg = getattr(args, "optim", None)
    scheduler_params = getattr(optim_cfg, "scheduler_params", None) if optim_cfg else None
    step_on = None
    scheduler_name = None
    if scheduler_params is not None:
        if isinstance(scheduler_params, dict):
            step_on = scheduler_params.get("step_on")
        else:
            step_on = getattr(scheduler_params, "step_on", None)
    if optim_cfg is not None:
        scheduler_name = getattr(optim_cfg, "scheduler", None)

    runner_cfg = getattr(args, "runner", None)
    if runner_cfg is None:
        run_mode = "epoch"
    elif hasattr(runner_cfg, "get"):
        run_mode = runner_cfg.get("run_mode", "epoch")
    else:
        run_mode = getattr(runner_cfg, "run_mode", "epoch")

    resolver = EpochSuffixResolver(
        run_mode=run_mode,
        step_on=step_on,
        train_loader_length=train_loader_length,
        accum_grad=accum_grad,
        distributed=distributed,
        scheduler_name=scheduler_name,
    )
    result = resolver.resolve(args)

    for log_line in result.logs:
        logger.info(log_line)
    for warning_line in result.warnings:
        _warnings.warn(warning_line, stacklevel=2)
