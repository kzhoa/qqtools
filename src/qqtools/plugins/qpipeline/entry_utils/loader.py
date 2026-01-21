import qqtools as qt
import torch
from torch.utils.data.dataloader import DataLoader
from torch_geometric.loader.dataloader import Collater

__all__ = ["prepare_dataloder"]


def build_loader_trival(
    ds,
    batch_size,
    num_workers=1,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    collate_fn=None,
):

    if not collate_fn:
        # use pyg as default
        collate_fn = Collater(ds)

    loader = DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def build_loader_ddp(
    ds,
    batch_size,
    num_workers=1,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    collate_fn=None,
):

    if not collate_fn:
        # use pyg as default
        collate_fn = Collater(ds)

    sampler = torch.utils.data.DistributedSampler(
        ds,
        num_replicas=qt.qdist.get_world_size(),
        rank=qt.qdist.get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


def build_loader(ds, distributed=False, **meta):
    if ds is None:
        return None
    if distributed:
        return build_loader_ddp(ds, **meta)
    else:
        return build_loader_trival(ds, **meta)


def prepare_dataloder(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    eval_batch_size,
    num_workers,
    pin_memory=True,
    distributed=False,
    collate_fn=None,
):
    """
    if `collate_fn` is nor provided, pyg Collater will be used as default.
    """
    meta = {"num_workers": num_workers, "pin_memory": pin_memory, "collate_fn": collate_fn}
    tr_meta = {**meta, "batch_size": batch_size, "shuffle": True, "drop_last": True}  # qq: python>=3.5
    ev_meta = {**meta, "batch_size": eval_batch_size, "shuffle": False, "drop_last": False}
    tr_loader = build_loader(train_dataset, distributed, **tr_meta)
    val_loader = build_loader(val_dataset, distributed, **ev_meta)
    te_loader = build_loader(test_dataset, distributed, **ev_meta)
    return tr_loader, val_loader, te_loader
