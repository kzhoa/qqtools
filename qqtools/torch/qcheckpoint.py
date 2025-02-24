"""
utils for save and recover ckp
friendly adapted with ddp

"save_ckp", "recover" should be used together
"""

import datetime
import re
from pathlib import Path

import torch

from . import qdist

__all__ = ["save_ckp", "recover"]


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_ckp(model, optimizer, lr_scheduler=None, save_dir=None, save_file=None, **other_params):
    """save checkpoint on rank 0"""
    # only save checkpoint on main process
    if qdist.get_rank() != 0:
        return

    # null check and path join
    assert not (save_dir is None and save_file is None)
    save_dir = "" if save_dir is None else save_dir
    save_file = f"checkpoint_{now_str()}.pt" if save_file is None else save_file
    save_path = Path(save_dir, save_file)
    print(f"Saving checkpoint to: {save_path} ...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        checkpoint["lrscheduler_state_dict"] = lr_scheduler.state_dict()
    checkpoint.update(other_params)
    torch.save(checkpoint, save_path)
    print("Saving checkpoint Done.")


def recover(
    model: torch.nn.Module,
    optimizer=None,
    restore_file: str = None,
    strict=True,
    exclude=[],
):
    """recover a model from checkpoint
    Returns
    -------
    dict
        checkpoint dict
    """
    assert restore_file is not None
    if restore_file == "" or not Path(restore_file).is_file():
        # `is_file` also return False when file not exist
        raise FileExistsError(f"file: `{restore_file}`  not exist or is a directory")

    # recover
    checkpoint = torch.load(restore_file, map_location=torch.device("cpu"), weights_only=True)

    # add ddp - state dict convert
    if qdist.is_dist_available_and_initialized():
        k = list(checkpoint["model_state_dict"].keys())[0]
        if not k.startswith("module."):
            # add prefix
            model_state_dict = {"module." + k: v for k, v in checkpoint["model_state_dict"].items()}
            checkpoint["model_state_dict"] = model_state_dict
    else:
        k = list(checkpoint["model_state_dict"].keys())[0]
        if k.startswith("module."):
            pattern = r"module.([\s\S]*)"  # noqa
            # remove prefix
            model_state_dict = {re.findall(pattern, k)[0]: v for k, v in checkpoint["model_state_dict"].items()}
            checkpoint["model_state_dict"] = model_state_dict

    # delete weights that not welcomed
    for key in exclude:
        if key in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"][key]

    # load weihgts considering ddp
    with qdist.qBarrier():
        # model
        res = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if strict is False and qdist.get_rank() == 0:
            if len(res.unexpected_keys) > 0:
                print(
                    f"{len(res.unexpected_keys)} Unexpected key(s) in state_dict: { ','.join(res.unexpected_keys) }. "
                )
            if len(res.missing_keys) > 0:
                print(f"{len(res.missing_keys)} Missing key(s) in state_dict: { ','.join(res.missing_keys) }. ")
        # optimizer
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                rank = qdist.get_rank()
                print(f"rank_{rank}: error occurs when load optimizer state dict.")
                print(repr(e))
        return checkpoint
