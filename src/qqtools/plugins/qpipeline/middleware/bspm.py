import qqtools as qt
import torch

from ..qpipeline import qPipeline

__all__ = ["middleware_bspm_support"]


def qatoms_collate_bspm(data_list):
    """
    qq:
    some llm-devoted fans prefer the format like (batch, seq_len), padded along with a mask.
    called `bspm` format here.
    """
    max_natoms = max(data.z.shape[0] for data in data_list)
    batch_size = len(data_list)

    # declare vars
    z_padded = torch.zeros(batch_size, max_natoms, dtype=torch.long)  # zero-pad
    pos_padded = torch.zeros(batch_size, max_natoms, 3, dtype=torch.float32)  # zero-pad
    mask = torch.zeros(batch_size, max_natoms, dtype=torch.bool)

    # adapt qm9
    has_y = True if "y" in data_list[0] else False
    if has_y:
        ydim = data_list[0].y.shape[1]
        y_padded = torch.zeros(batch_size, ydim, dtype=torch.float32)

    # fill in vars
    for i, data in enumerate(data_list):
        natoms = data.z.shape[0]
        z_padded[i, :natoms] = data.z
        pos_padded[i, :natoms] = data.pos
        mask[i, :natoms] = True

        if has_y:
            y_padded[i, :] = data.y.view(-1)

    batch_data = qt.qData({"z": z_padded, "pos": pos_padded, "y": y_padded, "mask": mask})
    return batch_data


def qatoms_batchforward_bpsm(model, batch_data):
    z, pos, mask = batch_data.z, batch_data.pos, batch_data.mask
    return model(z, pos, mask=mask)


def middleware_bspm_support(pipe: qPipeline):
    """adapt input format for `bpsm` models"""
    task = pipe.task
    train_loader = task.train_loader
    train_loader.collate_fn = staticmethod(qatoms_collate_bspm)
    task.batch_forward = staticmethod(qatoms_batchforward_bpsm)
