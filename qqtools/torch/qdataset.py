import copy
import functools
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Union

import h5py
import numpy as np
import qqtools as qt
import torch
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..qdict import qDict


class qData(qDict):

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


def qbatch_collate(data_list):
    """no pad, assume all data have same length"""
    v = data_list[0]
    if isinstance(v, torch.Tensor):
        res = torch.stack(data_list)  # (bz,)
    elif isinstance(v, (np.ndarray, np.generic)):
        res = np.stack(data_list)  # (bz, *)
        res = torch.as_tensor(res)
    else:
        raise TypeError(f"type {type(v)}")
    return res


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


class qDictDataset(torch.utils.data.Dataset, ABC):
    """
    self.datalist : List[dict]

    We employ the same filepath convention with the pyg package.
    """

    def __init__(self, root):
        self.root = root
        self.data_list: List[dict] = []
        self._indices = None
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        if not self.processed_files_exists():
            with qt.Timer("Process raw files", prefix="[qDataset]"):
                self._process()

    @property
    @abstractmethod
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def processed_file_names(self):
        raise NotImplementedError()

    @property
    def raw_dir(self):
        return Path(self.root).joinpath("raw").absolute()

    @property
    def processed_dir(self):
        return Path(self.root).joinpath("processed").absolute()

    @property
    def raw_paths(self):
        return [str(self.raw_dir / fn) for fn in self.raw_file_names]

    @property
    def processed_paths(self):
        return [str(self.processed_dir / fn) for fn in self.processed_file_names]

    def __getitem__(self, idx) -> Union[dict, torch.utils.data.Dataset]:
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            return self.data_list[idx]
        else:
            return self.index_select(idx)

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

    def _process(self):
        if hasattr(self, "process"):
            self.process()

    def shuffle(
        self,
    ) -> "torch.utils.data.Dataset":
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return dataset

    def collate(self, batch_size, target_keys=None, padding: dict = None):
        """prepare"""
        if padding is None:
            padding = self.padding if hasattr(self, "padding") else defaultdict(lambda: 0)

        def yield_batch_data(iterable_sequence: List[dict], batch_size, target_keys, padding):
            batch = []
            i = 0
            for d in iterable_sequence:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield qdict_pad_collate_fn(batch, padding, target_keys)
                    batch = []

            if len(batch) > 0:
                yield qdict_pad_collate_fn(batch, padding, target_keys)

        yield_fn = functools.partial(
            yield_batch_data,
            batch_size=batch_size,
            target_keys=target_keys,
            padding=padding,
        )
        return qDictDataloader(self, batch_size, yield_fn)

    def get_norm_factor(self, target):
        vs = [self.data_list[i][target] for i in self.indices()]
        val = qbatch_collate(vs)
        mean = torch.mean(val).item()
        std = torch.std(val).item()
        return (mean, std)


class qDictDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, yield_fn=None, shuffle=False, **kwargs):
        self.yield_fn = yield_fn
        self.batch_list = None
        super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=qdict_pad_collate_fn, **kwargs)

    def __iter__(self):
        if self.batch_list is not None:
            return iter(self.batch_list)
        elif self.yield_fn is not None:
            return self.yield_fn(iter(self.dataset))
        else:
            return super().__iter__()

    def cache(self):
        assert self.yield_fn is not None
        self.batch_list = list(self.yield_fn(iter(self.dataset)))
        return self
