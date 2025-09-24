import copy
import functools
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.utils
from torch import Tensor

import qqtools as qt

from .qsplit import get_data_splits


class qData(qt.qDict):

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self

    @staticmethod
    def get_splits(
        total_num,
        sizes=None,
        ratios=None,
        seed=1,
    ):
        return get_data_splits(total_num, sizes, ratios, seed)


def naive_values_collate(ls_values):
    """no pad, assume all data have same length"""
    v = ls_values[0]
    if isinstance(v, torch.Tensor):
        res = torch.stack(ls_values)  # (bz, *)
    elif isinstance(v, (float, int)):
        res = torch.stack([torch.tensor(val) for val in ls_values], dim=0)  # (bz,)
    elif isinstance(v, (np.ndarray, np.generic)):
        res = torch.from_numpy(np.stack(ls_values))  # (bz, *)
    elif isinstance(v, str):
        res = ls_values
    else:
        raise TypeError(f"type {type(v)}")
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

    merged = qt.qDict()
    for key in batch_list[0].keys():
        values = [sample[key] for sample in batch_list]
        v = values[0]

        try:
            # Handle different data types
            if isinstance(v, torch.Tensor):
                merged[key] = torch.stack(values)  # (bz, *)
            elif isinstance(v, (float, int)):
                merged[key] = torch.tensor(values)  # (bz,)
            elif isinstance(v, (np.ndarray, np.generic)):
                merged[key] = torch.from_numpy(np.stack(values))  # (bz, *)
            elif isinstance(v, str):
                merged[key] = values  # Keep as list of strings
            else:
                raise TypeError(f"Unsupported type {type(v)} for key '{key}'")
        except Exception as e:
            raise RuntimeError(f"Failed to collate key '{key}': {str(e)}") from e

    return merged


def collate_graph_samples(batch_list):
    """
    Collates a list of graph samples into a single batch.

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
    reserved_keys = ["edge_index", "batch"]
    batch_indices = []
    node_count = 0
    graph_data = defaultdict(list)
    edge_index_list = []

    _keys = list(batch_list[0].keys())
    assert all(k in sample for sample in batch_list for k in _keys), "Not all keys are the same in the batch"

    attr_keys = set(_keys) - set(reserved_keys)
    has_edge_index = "edge_index" in _keys

    #  classify attribute types based on first sample
    sample0 = batch_list[0]
    num_edges0 = sample0["edge_index"].shape[1] if has_edge_index else 0
    edge_attr_keys = set()
    for k in attr_keys:
        value = sample0[k]
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim >= 1:
            if has_edge_index and value.shape[0] == num_edges0:
                edge_attr_keys.add(k)

    # handle
    for i, sample in enumerate(batch_list):
        num_nodes = None
        for k in attr_keys:
            if k not in edge_attr_keys:  # skip edge attributes
                value = sample[k]
                # single node graph is not considered
                if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim >= 1 and value.shape[0] > 1:
                    if num_nodes is None:
                        num_nodes = value.shape[0]
                    else:
                        assert (
                            num_nodes == value.shape[0]
                        ), f"Node count of key {k} mismatch for sample {i}, got {num_nodes} and {value.shape[0]}"

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

        # store all data
        for key in attr_keys:
            value = sample[key]
            if key not in reserved_keys:
                graph_data[key].append(value)

        # create batch indices
        # consider situation that only graph attributes provided
        if num_nodes is not None:
            batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
            node_count += num_nodes

    # Concatenate all data
    for key, value_list in graph_data.items():
        if isinstance(value_list[0], np.ndarray):
            graph_data[key] = torch.cat([torch.from_numpy(v) for v in value_list], dim=0)
        elif isinstance(value_list[0], torch.Tensor):
            graph_data[key] = torch.cat(value_list, dim=0)
        elif isinstance(value_list[0], (int, float)):
            graph_data[key] = torch.tensor(value_list)
        else:
            graph_data[key] = value_list  # Strings and other types

    batch_combined = torch.cat(batch_indices, dim=0)
    result = qt.qDict({"batch": batch_combined, **graph_data})
    if has_edge_index:
        result["edge_index"] = torch.cat(edge_index_list, dim=1)

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

    def processed_files_exists(self):
        return all([Path(f).exists() for f in self.processed_paths])

    def raw_files_exists(self):
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
        val = naive_values_collate(vs)
        mean = torch.mean(val).item()
        std = torch.std(val).item()
        return (mean, std)

    def get_splits(self, ratios=[0.8, 0.1, 0.1], seed=None):
        return get_data_splits(total_num=self.__len__(), ratios=ratios, seed=seed)


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
        # If no collate_fn is provided, use the default behavior
        if collate_fn is None:
            collate_fn = collate_graph_samples if is_graph else collate_dict_samples
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
