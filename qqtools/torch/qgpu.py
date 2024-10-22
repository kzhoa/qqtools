import traceback
from typing import Union

import torch


def parse_device(dev: Union[str, int]):
    """
    :param dev:
    :1. cuda
    :2. cuda:number
    :3. number
    :4. cuda:0,1,3,4,5
    """
    if isinstance(dev, torch.device):
        return dev

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if isinstance(dev, str) and "cpu" in dev:
        device = torch.device("cpu")
    elif isinstance(dev, str) and "cuda" in dev:
        try:
            device = torch.device(dev)  # Parse 'cuda:1'
        except ValueError:
            print(traceback.print_exc())
    elif isinstance(dev, int) or str.isnumeric(str(dev)):
        idx = int(dev)
        device = torch.device("cuda", idx)
    elif dev is None:
        raise ValueError("input device is None, you must assign a device")
    else:
        raise NotImplementedError(f"Unsupported device: {dev}")
    return device


def gpu(show=True):
    import sys

    import gpustat

    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
    except Exception as e:
        sys.stderr.write("Error on querying NVIDIA devices." " Use --debug flag for details\n")
        try:
            import traceback

            traceback.print_exc(file=sys.stderr)
        except Exception:
            # NVMLError can't be processed by traceback:
            #   https://bugs.python.org/issue28603
            # as a workaround, simply re-throw the exception
            raise e
        sys.exit(1)

    if show:
        gpu_stats.print_formatted()

    return gpu_stats
