import functools
import os
from typing import Callable

import torch
import torch.distributed as dist

REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "mean": dist.ReduceOp.AVG,
    "avg": dist.ReduceOp.AVG,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """minior different from torch.dist
    if not in ddp status, `torch.dist.get_rank` returns -1,
    while we return 0.
    """
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args, verbose=True):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        if verbose:
            print(f"{ '*' * 40}\nNot using distributed mode\n{'*' * 40}")
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    args.dist_url = "env://"

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        device_id=args.local_rank,
    )
    torch.distributed.barrier()


def all_reduce(value, device, reduceOp="sum"):
    """Perform an all-reduce operation across distributed processes.

    Args:
        value: Input tensor or value to be reduced.
        device: Target device for the operation (e.g., 'cuda:0' or 'cpu').
        reduceOp: Reduction operation to apply. Must be one of:
            - 'sum': Sum of all values (default)
            - 'mean': Average of all values
            - 'min': Minimum value
            - 'max': Maximum value
    """
    op = REDUCE_OPS[reduceOp]
    is_orig_tensor = torch.is_tensor(value)
    if not is_orig_tensor:
        value = torch.Tensor([value])
    value = value.to(device, non_blocking=True)
    dist.all_reduce(value, op, async_op=False)
    return value if is_orig_tensor else value.item()


class qBarrier(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, trace_tb):
        if exc_type is not None:
            print(trace_tb)
            raise RuntimeError(exc_value)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def ddpCall(fn, /, *args, **kwargs):
    if not is_dist_available_and_initialized():
        return fn(*args, **kwargs)
    rank = get_rank()
    # ddp
    with qBarrier():
        if rank == 0:
            res = fn(*args, **kwargs)
    with qBarrier():
        if rank != 0:
            res = fn(*args, **kwargs)
    return res


def ddp_safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_dist_available_and_initialized():
            rank = get_rank()
            if rank == 0:
                result = fn(*args, **kwargs)
            dist.barrier()
            if rank != 0:
                result = fn(*args, **kwargs)
            dist.barrier()
            return result
        else:
            return fn(*args, **kwargs)

    return wrapper


def rank_zero_only(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return fn(*args, **kwargs)
        else:
            return None

    return wrapper


@rank_zero_only
def main_print(*args, **kwargs):
    print(*args, **kwargs)
