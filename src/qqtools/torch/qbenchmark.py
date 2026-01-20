import torch
import qqtools as qt

__all__ = ["get_duration", "get_duration_by_sync"]


def get_duration(n_epochs, fn, *args, device="cuda", warm_up=True, **kwargs):
    # from https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    # set context since torch.cuda.Event cannot specify a GPU index.
    with torch.cuda.device(device):
        # Warmup
        if warm_up:
            for _ in range(5):
                fn(*args, **kwargs)
        # Timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_epochs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_epochs)]
        torch.cuda.synchronize()
        for i in range(n_epochs):
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    duration = sum(times)
    return duration / 1000  # change to seconds


def get_duration_by_sync(n_epochs, fn, *args, **kwargs):
    """simply cuda.sync() when enter and exit"""
    with qt.Timer(cuda=True, verbose=False) as t:
        for _ in range(n_epochs):
            fn(*args, **kwargs)
    return t.duration


def get_duration_by_device_interface(n_epochs, fn, *args, device="cuda", warm_up=True, **kwargs):
    """extend API to specify GPU device"""
    di = torch._dynamo.device_interface.get_interface_for_device(device)
    # Warmup
    if warm_up:
        for _ in range(5):
            fn(*args, **kwargs)
    # Timing
    start_events = [di.Event(enable_timing=True) for _ in range(n_epochs)]
    end_events = [di.Event(enable_timing=True) for _ in range(n_epochs)]
    di.synchronize()
    for i in range(n_epochs):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    duration = sum(times)
    return duration / 1000  # change to seconds
