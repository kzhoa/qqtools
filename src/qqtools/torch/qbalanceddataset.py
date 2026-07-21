from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..data.qbalance import compute_global_even_sort_order, validate_balance_strategy
from ..qimport import LazyImport
from .qdataset import qDictDataset

tqdm = LazyImport("tqdm", "tqdm")


class _BalancedDatasetProvider:
    """Manage reusable balance metadata for a host dataset."""

    META_FILE_NAME = "balance_meta.npz"
    ORDER_FILE_NAME = "balance_order.npy"

    def __init__(
        self,
        host: qDictDataset,
        *,
        enabled: bool,
        is_graph: bool,
        balance_seed: int,
        balance_strategy: str,
        get_sample_cost: Callable[[Any], int | float] | None = None,
    ) -> None:
        if not isinstance(host, qDictDataset):
            raise TypeError(
                f"Balanced dataset host must be a qDictDataset, got {type(host).__name__}"
            )

        self.host = host
        self.enabled = bool(enabled)
        self.is_graph = bool(is_graph)
        self.balance_seed = int(balance_seed)
        self.balance_strategy = validate_balance_strategy(str(balance_strategy))
        self._get_sample_cost = self._resolve_get_sample_cost(get_sample_cost)
        self._meta_cache: np.ndarray | None = None
        self._order_cache: np.ndarray | None = None

    def processed_file_names(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        return (self.META_FILE_NAME, self.ORDER_FILE_NAME)

    def process(self) -> None:
        if self.enabled:
            self.rebuild_balance_assets(force=False)

    def rebuild_balance_assets(self, force: bool = False) -> None:
        self._require_enabled()
        meta_was_built = False
        if force or not self.meta_path.exists():
            self._build_meta()
            meta_was_built = True
        if force or meta_was_built or not self.order_path.exists():
            self._build_order()

    def sample_costs(self) -> np.ndarray:
        self._require_enabled()
        if self._meta_cache is None:
            self._meta_cache = self._load_meta()
        return self._meta_cache

    def sample_order(self) -> np.ndarray:
        self._require_enabled()
        if self._order_cache is None:
            self._order_cache = self._load_order()
        return self._order_cache

    @property
    def meta_path(self) -> Path:
        return self._processed_dir / self.META_FILE_NAME

    @property
    def order_path(self) -> Path:
        return self._processed_dir / self.ORDER_FILE_NAME

    @property
    def _processed_dir(self) -> Path:
        processed_dir = self.host.processed_dir
        if processed_dir is None:
            raise RuntimeError("Balance assets require the host dataset to define `root`.")
        return Path(processed_dir)

    def _resolve_get_sample_cost(
        self,
        get_sample_cost: Callable[[Any], int | float] | None,
    ) -> Callable[[Any], int | float] | None:
        if not self.enabled:
            return None
        if get_sample_cost is not None:
            return get_sample_cost
        host_hook = getattr(self.host, "get_sample_cost", None)
        if host_hook is None or not callable(host_hook):
            raise RuntimeError(
                "Balance is enabled but `get_sample_cost` is unavailable. "
                "Implement `get_sample_cost(sample)` on the dataset subclass."
            )
        return host_hook

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Balance assets are disabled. Set `enable_balance=True` or "
                "`enable_rewrite=True` when constructing the dataset."
            )

    def _build_meta(self) -> None:
        total = self.host.len()
        costs = np.empty(total, dtype=np.float64)
        try:
            indices = tqdm(
                range(total),
                total=total,
                desc="Build balance metadata",
                unit="sample",
                disable=None,
            )
            for idx in indices:
                costs[idx] = self._normalize_cost(
                    self._get_sample_cost(self.host.get(idx)),
                    idx,
                )
        finally:
            close = getattr(self.host, "close", None)
            if callable(close):
                close()

        sample_indices = np.arange(total, dtype=np.int64)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_save_npz(
            self.meta_path,
            sample_costs=costs,
            sample_indices=sample_indices,
        )
        costs.setflags(write=False)
        self._meta_cache = costs
        self._order_cache = None

    def _build_order(self) -> None:
        costs = self.sample_costs()
        order = compute_global_even_sort_order(
            costs,
            seed=self.balance_seed,
            strategy=self.balance_strategy,
        )
        self._atomic_save_npy(self.order_path, order)
        order.setflags(write=False)
        self._order_cache = order

    def _load_meta(self) -> np.ndarray:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing balance metadata: {self.meta_path}")
        try:
            with np.load(self.meta_path, allow_pickle=False) as meta:
                costs = np.asarray(meta["sample_costs"], dtype=np.float64)
                sample_indices = np.asarray(meta["sample_indices"], dtype=np.int64)
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load balance metadata: {self.meta_path}") from exc

        self._validate_costs(costs)
        expected_indices = np.arange(costs.shape[0], dtype=np.int64)
        if not np.array_equal(sample_indices, expected_indices):
            raise RuntimeError(
                f"Invalid sample_indices in balance metadata: {self.meta_path}"
            )
        host_total = self.host.len()
        if costs.shape[0] != host_total:
            raise RuntimeError(
                "Balance metadata length does not match the dataset: "
                f"{costs.shape[0]} != {host_total}"
            )
        costs = np.ascontiguousarray(costs)
        costs.setflags(write=False)
        return costs

    def _load_order(self) -> np.ndarray:
        if not self.order_path.exists():
            raise FileNotFoundError(f"Missing balance order: {self.order_path}")
        try:
            order = np.asarray(np.load(self.order_path, allow_pickle=False), dtype=np.int64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load balance order: {self.order_path}") from exc

        total = int(self.sample_costs().shape[0])
        if order.shape != (total,):
            raise RuntimeError(
                f"Balance order has shape {order.shape}, expected {(total,)}: {self.order_path}"
            )
        if total > 0:
            if len(np.unique(order)) != total:
                raise RuntimeError(f"Balance order contains duplicates: {self.order_path}")
            if order.min(initial=0) != 0 or order.max(initial=-1) != total - 1:
                raise RuntimeError(
                    f"Balance order does not cover [0, {total}): {self.order_path}"
                )
        order = np.ascontiguousarray(order)
        order.setflags(write=False)
        return order

    @staticmethod
    def _normalize_cost(value: int | float, idx: int) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Sample cost at index {idx} must be numeric, got bool")
        try:
            cost = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Sample cost at index {idx} must be a real number, got {value!r}"
            ) from exc
        if not np.isfinite(cost):
            raise ValueError(f"Sample cost at index {idx} must be finite, got {cost}")
        if cost < 0:
            raise ValueError(f"Sample cost at index {idx} must be non-negative, got {cost}")
        return cost

    @staticmethod
    def _validate_costs(costs: np.ndarray) -> None:
        if costs.ndim != 1:
            raise RuntimeError(f"sample_costs must be 1D, got shape {costs.shape}")
        if not np.all(np.isfinite(costs)):
            raise RuntimeError("sample_costs must contain only finite values")
        if np.any(costs < 0):
            raise RuntimeError("sample_costs must be non-negative")

    @staticmethod
    def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
