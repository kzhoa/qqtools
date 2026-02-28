import copy
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.utils
from torch import Tensor

import qqtools as qt

from ..qimport import LazyImport
from ..utils.warning import QDataWarning
from .qsplit import get_data_splits

lmdb = LazyImport("lmdb")


class qData(qt.qDict):
    """A simple data container adheres to torch dataloader"""

    def __init__(self, d=None, /, **kwargs):
        if d is not None and kwargs:
            raise ValueError(
                "Conflicting arguments: "
                "Either pass a dictionary `d` OR keyword arguments, "
                "but not both.\n"
                "Example:\n"
                "  qData({'key': value}) \n"
                "  qData(key=value)      "
            )
        d = d if d is not None else kwargs
        super().__init__(d, allow_notexist=False, recursive=False)

    def to(self, target):
        """inplace operation"""
        if isinstance(target, torch.dtype):
            for k, v in self.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if target.is_floating_point and (not v.dtype.is_floating_point):
                    warnings.warn(
                        f"[qData] Converting tensor '{k}' ({v.dtype}) to {target}. "
                        f"Integer-to-float conversion may lose precision. "
                        f"Use qData.float()/double() to only convert floating-point tensors.",
                        QDataWarning,
                    )
                self[k] = v.to(target)
        else:
            # assume the device
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    self[k] = v.to(target)
        return self

    def to_dtype(self, target):
        """
        safely handle floating tensors
        """
        assert isinstance(target, torch.dtype), f"expect torch.dtype but got {type(target)}"
        if target.is_floating_point:
            self._convert_floating_tensors(target=target)

        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if target.is_floating_point and (not v.dtype.is_floating_point):
                    warnings.warn(
                        f"[qData] Converting tensor '{k}' ({v.dtype}) to {target}. "
                        f"Integer-to-float conversion may lose precision. "
                        f"Use qData.float()/double() to only convert floating-point tensors.",
                        QDataWarning,
                    )
                self[k] = v.to(target)

    def _convert_floating_tensors(self, target: torch.dtype):
        for k, v in self.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                self[k] = v.to(target)

    def float(self):
        self._convert_floating_tensors(torch.float32)

    def double(
        self,
    ):
        self._convert_floating_tensors(torch.float64)

    def __copy__(self):
        """return new instance"""
        _d = self.__class__.__new__(self.__class__)
        _d.__init__(self)
        return _d

    @staticmethod
    def get_splits(
        total_num,
        sizes=None,
        ratios=None,
        seed=1,
    ):
        return get_data_splits(total_num, sizes, ratios, seed)


# class qBatchList(qDataList):
#     def to(self, device):
#         """
#         This is a no-operation method that allows qBatchList to be used
#         in pipelines expecting PyTorch's `.to(device)` interface while
#         preserving the list-of-dicts structure without tensor conversion.
#         """
#         return self


def smart_combine(value_list: List, prefer_stack=False):
    """Intelligently combines a list of values using either stack or concatenation.

    Args:
        value_list (List[Union[torch.Tensor, float, int, np.ndarray, str]]):
            List of values to combine. All elements must be of the same type.
        prefer_stack (bool, optional):
            If True, prioritizes stacking (adding new dimension).
            If False, prioritizes concatenation (along existing dimension).
            Defaults to False.

            Examples:
                - prefer_stack=True:  [(3,3), (3,3)] -> stack -> (2,3,3)
                - prefer_stack=False: [(3,3), (3,3)] -> cat  -> (6,3)

    Returns:
        Union[torch.Tensor, List[str]]:
            Combined result. Returns original list if input contains strings.
    """
    v = value_list[0]
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:  # scala
            res = torch.stack(value_list)  # have to use torch.stack
        elif prefer_stack:
            res = torch.stack(value_list)  # (bz, *)
        else:
            res = torch.cat(value_list, dim=0)
    elif isinstance(v, (float, int)):
        res = torch.tensor(value_list)  #
    elif isinstance(v, (np.ndarray, np.generic)):
        if prefer_stack:
            val = np.stack(value_list)  # (bz, *)
        else:
            val = np.concatenate(value_list, dim=0)
        res = torch.from_numpy(val)
    elif isinstance(v, str):
        res = value_list
    elif isinstance(v, (list, tuple)):
        res = value_list
    else:
        raise TypeError(f"Unsupported type: {type(v)}")

    return res


def collate_dict_samples(batch_list: List[dict]):
    """
    Merge a list of dicts into a batch, supporting various data types.

    Args:
        batch: List of samples (dicts), each with the same keys

    Returns:
        A dict where each key corresponds to a merged batch of values
    """
    if not batch_list:
        return {}

    # Verify all samples have the same keys
    first_keys = set(batch_list[0].keys())
    for i, sample in enumerate(batch_list[1:], 1):
        sample_keys = set(sample.keys())
        if sample_keys != first_keys:
            missing = first_keys - sample_keys
            extra = sample_keys - first_keys
            raise AssertionError(f"Sample {i} has inconsistent keys. Missing: {missing}, Extra: {extra}")

    merged = qt.qData()
    for key in batch_list[0].keys():
        values = [sample[key] for sample in batch_list]
        v = values[0]

        try:
            # Handle different data types
            merged[key] = smart_combine(values, prefer_stack=True)
        except Exception as e:
            raise RuntimeError(f"Failed to collate key '{key}': {str(e)}") from e

    return merged


def determine_graph_key_types(batch_list):
    """
    Determines the type (node, edge, or graph attribute) for each key in the graph samples.

    Args:
        batch_list: A list of graph sample dictionaries. Assumes all samples have consistent keys.


    Returns:
        A dictionary mapping attribute type ('node', 'edge', 'graph') to a set of keys.
        Returns None if batch_list is empty or invalid.
    """

    reserved_keys = ["num_nodes", "edge_index", "batch"]

    if not batch_list:
        return None, None, None  # Return None if batch is empty

    # ================= Determine Key Types =================
    sample0 = batch_list[0]
    _keys = list(sample0.keys())
    assert all(k in sample for sample in batch_list for k in _keys), "Not all keys are the same in the batch"

    attr_keys = set(_keys) - set(reserved_keys)
    has_edge_index = "edge_index" in _keys

    num_nodes_by_sample = []
    num_edges_by_sample = []
    for sample in batch_list:
        num_nodes = sample.get("num_nodes", None)
        num_edges = None
        if has_edge_index:
            edge_index = sample["edge_index"]
            num_edges = edge_index.shape[1]
            if num_nodes is None and edge_index.numel() > 0:
                num_nodes = edge_index.max().item() + 1

        num_nodes_by_sample.append(num_nodes)
        num_edges_by_sample.append(num_edges)

    edge_attr_keys = set()
    graph_attr_keys = set()
    node_attr_keys = set()
    pending_attr_keys = set()
    key_length_signature = {}

    for k in attr_keys:
        value = sample0[k]

        if not isinstance(value, (torch.Tensor, np.ndarray)) or value.ndim < 1:
            continue

        lengths = []
        for sample in batch_list:
            sample_value = sample[k]
            if not isinstance(sample_value, (torch.Tensor, np.ndarray)) or sample_value.ndim < 1:
                lengths = []
                break
            lengths.append(sample_value.shape[0])

        if not lengths:
            continue

        key_length_signature[k] = tuple(lengths)
        is_shape_constant = all(length == lengths[0] for length in lengths)

        node_match_all = all(
            num_nodes is not None and length == num_nodes for length, num_nodes in zip(lengths, num_nodes_by_sample)
        )
        edge_match_all = has_edge_index and all(
            num_edges is not None and length == num_edges for length, num_edges in zip(lengths, num_edges_by_sample)
        )

        if edge_match_all and (not node_match_all):
            edge_attr_keys.add(k)
            continue

        if node_match_all and (not edge_match_all):
            node_attr_keys.add(k)
            continue

        # Ambiguous case: length can match both node and edge counts.
        # Keep backward-compatible behavior for constant shapes by preferring
        # node attributes; leave varying shapes for second-pass disambiguation.
        if node_match_all and edge_match_all:
            if is_shape_constant:
                node_attr_keys.add(k)
            else:
                pending_attr_keys.add(k)
            continue

        if not is_shape_constant:
            pending_attr_keys.add(k)
            continue

        # 3. Graph Attribute Identification
        # Treat attributes with constant shape across samples as graph attributes.
        # This handles unknown/constant num_nodes. Misclassified node attributes
        # can be recovered later if needed.
        # If a constant-shape node attribute is misclassified as a graph attribute,
        # its stacked form [B, C, ...] can be later reshaped to [B*C, ...]
        # to function correctly as a node attribute [nA, ...].
        graph_attr_keys.add(k)

    # second pass: resolve pending keys by comparing their per-sample shape signature
    # with already confirmed node/edge keys
    node_signatures = {key_length_signature[k] for k in node_attr_keys if k in key_length_signature}
    edge_signatures = {key_length_signature[k] for k in edge_attr_keys if k in key_length_signature}

    unresolved_pending_keys = []
    for k in sorted(pending_attr_keys):
        signature = key_length_signature.get(k)
        is_node_signature = signature in node_signatures
        is_edge_signature = signature in edge_signatures

        if is_node_signature and (not is_edge_signature):
            node_attr_keys.add(k)
        elif is_edge_signature and (not is_node_signature):
            edge_attr_keys.add(k)
        else:
            unresolved_pending_keys.append(k)

    if unresolved_pending_keys:
        reason_text_map = {
            "no_anchor_signature": "no confirmed node/edge anchor signatures available",
            "ambiguous_signature_matches_both_node_and_edge": "signature matches both confirmed node and edge signatures",
            "no_matching_anchor_signature": "signature does not match any confirmed node/edge signature",
        }
        unresolved_details = []
        for k in unresolved_pending_keys:
            signature = key_length_signature.get(k)
            if (not node_signatures) and (not edge_signatures):
                reason = "no_anchor_signature"
            elif (signature in node_signatures) and (signature in edge_signatures):
                reason = "ambiguous_signature_matches_both_node_and_edge"
            else:
                reason = "no_matching_anchor_signature"
            reason_text = reason_text_map[reason]
            unresolved_details.append(f"{k} (signature={signature}, reason={reason}: {reason_text})")

        unresolved_details_text = "; ".join(unresolved_details)

        raise ValueError(
            "Cannot determine attribute types for keys: "
            f"{unresolved_pending_keys}. "
            "Reason: these keys vary across samples and remain unresolved after second-pass signature matching. "
            f"Details: {unresolved_details_text}. "
            "Please provide `num_nodes` or `edge_index` fields to enable automatic inference, "
            "or pass explicit `key_types`."
        )

    if len(node_attr_keys) == 0:
        # If edge_index exists, num_nodes can be inferred later, so okay to proceed.
        if (not has_edge_index) and all(num_nodes is None for num_nodes in num_nodes_by_sample):
            raise ValueError(
                "No node attributes identified. Please check your data format. "
                "If your graph has no node attributes, please add a dummy node attribute with shape (num_nodes, 1) to avoid ambiguity."
            )

    key_types = {
        "node": node_attr_keys,
        "edge": edge_attr_keys,
        "graph": graph_attr_keys,
    }
    return key_types


def collate_graph_samples(batch_list, key_types=None):
    """
    Collates a list of graph samples into a single batch.

    Reserved keys:
        - 'num_nodes': int
        - 'edge_index': Tensor[2,E]

    Args:
        batch_list: List of dictionaries, each representing a graph sample.
                   Each sample should have consistent keys.
                   Expected keys may include "edge_index" and others.

    Returns:
        A dictionary containing the batched data with:
        - All node features concatenated
        - Edge indices adjusted with offsets
        - Batch indices indicating which sample each node belongs to
    """
    if not batch_list:
        return {}  # Return empty dict for empty batch

    reserved_keys = ["num_nodes", "edge_index", "batch"]
    batch_indices = []
    node_count = 0
    graph_data = defaultdict(list)
    edge_index_list = []

    _keys = list(batch_list[0].keys())
    assert all(k in sample for sample in batch_list for k in _keys), "Not all keys are the same in the batch"

    attr_keys = set(_keys) - set(reserved_keys)
    has_edge_index = "edge_index" in _keys

    # Naming-convention based specialized keys.
    # Only keys matching these patterns use the custom path:
    #   - edge_*_index: treated like edge_index (shape [2, E]), with node-index offset during collation
    #   - edge_*_attr : treated like edge attributes (shape [E] or [E, ...]), concatenated on dim=0
    # All other keys continue to use the original inference + combine flow.
    named_edge_index_keys = {
        key for key in attr_keys if key.startswith("edge_") and key.endswith("_index") and key != "edge_index"
    }
    named_edge_attr_keys = {key for key in attr_keys if key.startswith("edge_") and key.endswith("_attr")}
    specialized_edge_keys = named_edge_index_keys.union(named_edge_attr_keys)

    # Determine key types if not provided, or convert input format to sets
    if key_types is None:
        # Specialized keys are handled by naming convention and should not be sent to
        # the generic key-type inference to avoid false unresolved-key errors.
        if specialized_edge_keys:
            batch_list_for_infer = [
                {k: v for k, v in sample.items() if k not in specialized_edge_keys} for sample in batch_list
            ]
            infer_attr_keys = set(batch_list_for_infer[0].keys()) - set(reserved_keys)
            if infer_attr_keys:
                has_num_nodes_hint = any(sample.get("num_nodes", None) is not None for sample in batch_list_for_infer)
                has_edge_index_hint = "edge_index" in batch_list_for_infer[0]
                if has_num_nodes_hint or has_edge_index_hint:
                    determined_key_types = determine_graph_key_types(batch_list_for_infer)
                else:
                    # Without node/edge hints, remaining non-special attrs are treated as graph attrs.
                    determined_key_types = {"node": set(), "edge": set(), "graph": set(infer_attr_keys)}
            else:
                determined_key_types = {"node": set(), "edge": set(), "graph": set()}
        else:
            determined_key_types = determine_graph_key_types(batch_list)

        if determined_key_types is None:  # Handle case where determine_key_types returns None
            return {}
        node_attr_keys = determined_key_types["node"]
        edge_attr_keys = determined_key_types["edge"]
        graph_attr_keys = determined_key_types["graph"]
    else:
        # Convert input types to sets if they aren't already
        node_attr_keys = set(key_types.get("node", set()))
        edge_attr_keys = set(key_types.get("edge", set()))
        graph_attr_keys = set(key_types.get("graph", set()))
        # Basic validation: Ensure provided keys are actually present attributes
        all_provided_attrs = node_attr_keys.union(edge_attr_keys).union(graph_attr_keys)
        unclassified_attrs = (attr_keys - specialized_edge_keys) - all_provided_attrs
        if unclassified_attrs:
            # Raise an error instead of a warning if attributes are unclassified.
            # This enforces that every attribute must be assigned a type.
            raise ValueError(
                f"The following attributes were found in the batch but not classified "
                f"as node, edge, or graph attributes: {unclassified_attrs}. "
                f"Please ensure they are correctly identified or included in key_types."
            )

    specialized_edge_index_data = defaultdict(list)
    specialized_edge_attr_data = defaultdict(list)

    # ================= Handle Loop =================
    for i, sample in enumerate(batch_list):
        num_nodes = None if "num_nodes" not in sample else sample["num_nodes"]
        for k in node_attr_keys:
            value = sample[k]
            if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim >= 1:
                if num_nodes is None:
                    num_nodes = value.shape[0]
                else:
                    assert (
                        num_nodes == value.shape[0]
                    ), f"Node count of key `{k}` mismatch for sample {i}, got {num_nodes} and {value.shape[0]}"

        num_edges = None
        if has_edge_index:
            edge_index = sample["edge_index"]
            num_edges = edge_index.shape[1]
            if num_nodes is None:
                # infer num_nodes from edge_index if no node attributes
                num_nodes = edge_index.max().item() + 1
            # adjust edge indices with cumulative offset
            adjusted_edge_index = edge_index.clone()
            adjusted_edge_index += node_count
            edge_index_list.append(adjusted_edge_index)

        # edge_*_index specialized path:
        # treat as index tensor [2, E], apply node-index offset exactly like edge_index.
        named_edge_index_values = {}
        max_index_from_named_edge_index = None
        for key in named_edge_index_keys:
            raw_value = sample[key]
            tensor_value = torch.as_tensor(raw_value)
            if tensor_value.ndim != 2 or tensor_value.shape[0] != 2:
                raise ValueError(
                    f"Key '{key}' is matched as edge_*_index and must have shape [2, E], "
                    f"but got {tuple(tensor_value.shape)} in sample {i}."
                )
            if tensor_value.dtype == torch.bool or torch.is_floating_point(tensor_value):
                raise TypeError(
                    f"Key '{key}' is matched as edge_*_index and must be an integer tensor, "
                    f"but got dtype={tensor_value.dtype} in sample {i}."
                )

            named_edge_index_values[key] = tensor_value
            if tensor_value.numel() > 0:
                current_max = int(tensor_value.max().item())
                if max_index_from_named_edge_index is None or current_max > max_index_from_named_edge_index:
                    max_index_from_named_edge_index = current_max

        # If num_nodes is still unknown, infer from edge_*_index keys.
        if num_nodes is None and max_index_from_named_edge_index is not None:
            num_nodes = max_index_from_named_edge_index + 1

        for key, tensor_value in named_edge_index_values.items():
            adjusted_value = tensor_value.clone()
            adjusted_value += node_count
            specialized_edge_index_data[key].append(adjusted_value)

        # edge_*_attr specialized path:
        # concatenate on dim=0, with no index offset.
        for key in named_edge_attr_keys:
            tensor_value = torch.as_tensor(sample[key])
            if tensor_value.ndim < 1:
                raise ValueError(
                    f"Key '{key}' is matched as edge_*_attr and must have at least 1 dimension, "
                    f"but got {tuple(tensor_value.shape)} in sample {i}."
                )
            specialized_edge_attr_data[key].append(tensor_value)

        # store all data
        for key in attr_keys:
            if key in specialized_edge_keys:
                continue
            value = sample[key]
            if key not in reserved_keys:
                graph_data[key].append(value)

        # create batch indices
        # consider situation that only graph attributes provided
        if num_nodes is not None:
            batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
            node_count += num_nodes

    # ================= Concatenate Data =================
    for key, value_list in graph_data.items():
        if key in graph_attr_keys:
            graph_data[key] = smart_combine(value_list, prefer_stack=True)
        elif key in node_attr_keys or key in edge_attr_keys:
            graph_data[key] = smart_combine(value_list, prefer_stack=False)
        else:
            # Should not happen if classification is correct
            raise KeyError(f"Attribute '{key}' was not classified as node, edge, or graph attribute.")

    for key, value_list in specialized_edge_attr_data.items():
        graph_data[key] = torch.cat(value_list, dim=0)

    for key, value_list in specialized_edge_index_data.items():
        graph_data[key] = torch.cat(value_list, dim=1)

    if len(batch_indices) > 0:
        batch_combined = torch.cat(batch_indices, dim=0)
    else:
        batch_combined = torch.empty(0, dtype=torch.long)  # keep consistent type even if empty

    result = qt.qData({"batch": batch_combined, **graph_data})
    if has_edge_index:
        result["edge_index"] = torch.cat(edge_index_list, dim=1)

    # TODO maybe if has_num_nodes?
    return result


def qdict_pad_collate_fn(batch_list: List[dict], padding: dict, target_keys):
    """
    maybe need multi type support
    """
    output = qData(default_function=list)
    for p in batch_list:
        for k, v in p.items():
            if target_keys is not None and k not in target_keys:
                continue
            if isinstance(v, (list, np.ndarray, np.generic, torch.Tensor)):
                output[k].append(torch.as_tensor(v))
            elif isinstance(v, str):
                continue
            else:
                raise TypeError(f"{type(v)}")
    for k, v in output.items():
        if isinstance(v[0], torch.Tensor):
            if v[0].dim() == 0:
                output[k] = torch.stack(v)  # (bz,)
            else:
                output[k] = torch.nn.utils.rnn.pad_sequence(v, True, padding[k])

            # TODO remove... tempory fix
            if output[k].dtype == torch.uint8:
                output[k] = output[k].type(torch.int64)

    return output


def has_override(parent, obj, method_name):
    """ """
    if not hasattr(obj, method_name):
        return False

    if not hasattr(parent, method_name):
        return True

    obj_method = getattr(obj, method_name)
    parent_method = getattr(parent, method_name)

    obj_func = getattr(obj_method, "__func__", obj_method)
    parent_func = getattr(parent_method, "__func__", parent_method)

    return obj_func != parent_func


class qDictDataset(torch.utils.data.Dataset, ABC):
    """A dataset class that works with series of dictionaries.


    This class accepts 3 usage patterns:

    1. Naive:
        Simply input a datalist: qDictDataset(data_list=[{}])
        Override `get()` and `len()` to customize the dataset

    2. Advanced:
        Initialize with qDictDataset(root='/path/to/root')
        Override `self.processed_file_names` and `self.process`
        We employ the same filepath convention with the pyg package

    3. Custom:
        Initialize with empty input: qDictDataset()
    """

    def __init__(self, data_list=None, root=None):

        self.data_list: List[dict] = []
        self._indices = None
        self.root = root

        # naive init
        if data_list is not None:
            self.data_list = data_list

        # advanced init
        if self.root is not None:
            self.processed_dir.mkdir(exist_ok=True, parents=True)
            self.maybe_process()

    @property
    def raw_file_names(self):
        raise []

    @property
    def processed_file_names(self):
        raise []

    @property
    def raw_dir(self):
        if self.root is None:
            return None
        return Path(self.root).joinpath("raw").absolute()

    @property
    def processed_dir(self):
        if self.root is None:
            return None
        return Path(self.root).joinpath("processed").absolute()

    @property
    def raw_paths(self):
        if self.root is None:
            return None
        return [str(self.raw_dir / fn) for fn in self.raw_file_names]

    @property
    def processed_paths(self):
        if self.root is None:
            return None
        return [str(self.processed_dir / fn) for fn in self.processed_file_names]

    def maybe_process(self):
        if not has_override(qDictDataset, self, "processed_file_names"):
            return

        if not self.processed_files_exist():
            self._process()

    def _process(self):
        if hasattr(self, "process"):
            self.process()

    def __getitem__(self, idx) -> Union[dict, torch.utils.data.Dataset]:
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self._indices is not None:
                idx = self._indices[idx]
            return self.get(idx)
        else:
            return self.index_select(idx)

    def get(self, true_idx):
        return self.data_list[true_idx]

    def len(self):
        return len(self.data_list)

    def __len__(self) -> int:
        return len(self.indices())

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    def __iter__(self):
        for idx in self.indices():
            yield self.__getitem__(idx)

    def index_select(self, idx: Union[slice, Tensor, np.ndarray, Sequence]) -> torch.utils.data.Dataset:
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def processed_files_exist(self):
        return all([Path(f).exists() for f in self.processed_paths])

    def raw_files_exist(self):
        return all([Path(f).exists() for f in self.raw_paths])

    def shuffle(
        self,
    ) -> "torch.utils.data.Dataset":
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return dataset

    def collate(self, batch_size, target_keys=None, padding: dict = None):
        """prepare fixed batches. Only recommended for small dataset."""
        if padding is None:
            padding = self.padding if hasattr(self, "padding") else defaultdict(lambda: 0)

        if target_keys is None:
            target_keys = list(self.data_list[0].keys()) if len(self.data_list) > 0 else []

        batches = []
        current_batch = []
        for data in self.data_list:
            current_batch.append(data)
            if len(current_batch) == batch_size:
                batches.append(qdict_pad_collate_fn(current_batch, padding, target_keys))
                current_batch = []
        if current_batch:  # last batch
            batches.append(qdict_pad_collate_fn(current_batch, padding, target_keys))
        return batches

    def get_norm_factor(self, target):
        vs = [self.data_list[i][target] for i in self.indices()]
        val = smart_combine(vs)
        mean = torch.mean(val).item()
        std = torch.std(val).item()
        return (mean, std)

    def get_splits(self, ratios=[0.8, 0.1, 0.1], seed=None):
        return get_data_splits(total_num=self.__len__(), ratios=ratios, seed=seed)

    @staticmethod
    def collate_dict_samples(*args, **kwargs):
        return collate_dict_samples(*args, **kwargs)

    @staticmethod
    def collate_graph_samples(*args, **kwargs):
        return collate_graph_samples(*args, **kwargs)

    @staticmethod
    def collate_keep_list_of_dict(batch_list):
        """
        Custom collate function that preserves list of dictionaries structure.

        qq:
        In NLP scenarios where each sample is a dictionary (e.g., containing tokens,
        labels, etc.), this function keeps the original list-of-dicts structure
        instead of batching into tensors. This enables the model to handle padding
        and tensor conversion individually for each sample on the model side.

        Example:
            >>> batch = [{'tokens': ['hello', 'world'], 'labels': [0, 1]},
                {'tokens': ['test'], 'labels': [1]}]
            >>> collated = collate_keep_list_of_dict(batch)
            >>> collated.to(device)  # Compatible with device transfer

        Args:
            batch_list: List of dictionaries, where each dict represents one sample

        Returns:
            qList: The input list of dictionaries, unchanged. Compatible with batch_data.to(device) usage。
        """
        return batch_list

    def to_list_dataloader(self, batch_size, **kwargs):
        return qDictDataloader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=qDictDataset.collate_keep_list_of_dict,
            **kwargs,
        )

    def to_graph_dataloader(self, batch_size, **kwargs):
        return qDictDataloader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=qDictDataset.collate_graph_samples,
            **kwargs,
        )


class qDictDataloader(torch.utils.data.DataLoader):
    """
    A specialized DataLoader for handling dictionary-based datasets with automatic collation.

    Examples:
        >>> # For regular dictionary data
        >>> loader = qDictDataloader(dataset, batch_size=32, shuffle=True)

        >>> # For graph data (automatically uses graph collation)
        >>> loader = qDictDataloader(graph_dataset, batch_size=16, is_graph=True)

        >>> # With custom collation function
        >>> loader = qDictDataloader(dataset, batch_size=8, collate_fn=my_collate_fn)
    """

    def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None, is_graph=False, **kwargs):
        """
        Initialize the qDictDataloader.

        Args:
            dataset (Dataset): Dataset from which to load the data. Should return dictionaries
                             when indexed. For graph data, dictionaries should contain
                             'edge_index' and node/edge attributes.
            batch_size (int): Number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the data at every epoch.
                                    Default: False.
            collate_fn (callable, optional): Custom function to collate samples into batches.
                                           If None, automatically selects based on is_graph.
            is_graph (bool, optional): Whether the dataset contains graph data. If True and
                                     no collate_fn is provided, uses graph-specific collation
                                     that handles edge index offsets and batch indices.
                                     Default: False.
            **kwargs: Additional keyword arguments passed to the parent DataLoader.

        Note:
            When is_graph=True, the collation function will:
            - Adjust edge indices with cumulative node offsets
            - Create batch indices indicating sample origin for each node
            - Concatenate node features along the node dimension
            - Handle both node-level and edge-level attributes appropriately
        """

        _collate_fn_to_use = collate_fn

        if collate_fn is None:
            if is_graph:
                # Create the stateful collate fn that caches key type inference results for efficiency
                _collate_fn_to_use = self._create_stateful_graph_collate()
                print(f"qDictDataloader: is_graph=True. Using stateful graph collator: {_collate_fn_to_use.__name__}")
            else:
                _collate_fn_to_use = collate_dict_samples
                print("qDictDataloader: is_graph=False. Using default collate_dict_samples.")

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn_to_use, **kwargs)

    def _create_stateful_graph_collate(self):
        self._cached_key_types = None

        def stateful_graph_collate(batch_list):
            if self._cached_key_types is None:
                inferred_types = determine_graph_key_types(batch_list)
                self._cached_key_types = inferred_types
            return collate_graph_samples(batch_list, key_types=self._cached_key_types)

        stateful_graph_collate.__name__ = "_stateful_graph_collate_internal"
        return stateful_graph_collate


class qLmdbDataset(qDictDataset):
    """
    LMDB dataset that acts like a dictionary.
    A consumer of qDictDataset, keep the storage behavior the same.

    Default Behavior:
        - Dataset Length: The number of entries in the dataset (`len(self)`)
          is automatically determined by querying `self.env.stat()["entries"]`.
          If this default behavior is unsuitable, users can provide a custom
          implementation by overriding the `def len(self) -> int` method.

        - Data Caching Strategy: For optimized repeated access, data retrieved
          from LMDB is cached in an internal list (`self.data_list`). Please be
          aware that this strategy can lead to increased memory usage, especially
          with large datasets. If memory efficiency is a priority and caching is
          undesired, override the `get(self, idx)` method to fetch and return
          data directly without storing it internally.

    Implementation Requirements:
        Subclasses must implement the following methods:
        - `get_key(self, idx)->bytes`: C
          Converts a given index `idx` into the corresponding key (as bytes)
          to be used for retrieving data from LMDB.
        - `parse_value(self, v)`:
          Parses the raw value (bytes) retrieved from
          LMDB into a usable Python object.

    Example Usage:
        class MyLmdbDataset(qLmdbDataset):
            def get_key(self, idx):
                return f"image_{idx}".encode()

            def parse_value(self, v):
                return torch.load(io.BytesIO(v))
    """

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        is_subdir=False,
        max_readers=128,
        map_size=1024 * 1024 * 1024 * 100,  # 100GB
    ):

        super().__init__(root=None)

        # we only save the params here
        self.lmdb_path = str(lmdb_path)
        self.is_subdir = is_subdir
        self.max_readers = max_readers
        self.map_size = map_size

        self.env = None
        self.txn = None

        self._initialized = False
        self._num_entries = None
        self._pre_load()

    @abstractmethod
    def get_key(self, idx) -> bytes:
        pass

    @abstractmethod
    def parse_value(self, v):
        pass

    def len(self) -> int:
        return self._num_entries

    def _pre_load(
        self,
    ):
        with lmdb.open(
            self.lmdb_path,
            readonly=True,
            subdir=self.is_subdir,
            max_readers=self.max_readers,
            map_size=self.map_size,
            lock=False,
            readahead=False,
            meminit=False,
        ) as env:
            self._num_entries = env.stat()["entries"]

        self.data_list = [None] * self._num_entries

    def _init_db(self):
        if not self._initialized:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                subdir=self.is_subdir,
                max_readers=self.max_readers,
                map_size=self.map_size,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)
            self._initialized = True

    def close(self):
        if self._initialized:
            self.txn.close()
            self.env.close()
            self._initialized = False

    def __del__(self):
        if hasattr(self, "_initialized"):
            self.close()

    def __getitem__(self, idx):
        if not self._initialized:
            self._init_db()

        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            if self._indices is not None:
                idx = self._indices[idx]
            return self.get(idx)
        else:
            return self.index_select(idx)

    def get(self, idx):
        # behavior: cache the data list
        if self.data_list[idx] is not None:
            return self.data_list[idx]

        key = self.get_key(idx)
        v = self.txn.get(key)
        v = self.parse_value(v)

        self.data_list[idx] = v
        return v
