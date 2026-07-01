import warnings

import torch.nn as nn

__all__ = ["collect_no_decay_params", "build_param_groups"]


def collect_no_decay_params(model: nn.Module) -> set:
    """Walk the module tree and collect fully-qualified parameter names
    that should be exempt from weight decay.

    Convention methods on nn.Module submodules:
    - no_decay_deep() -> List[str]: stops recursion into subtree
    - no_decay() -> List[str]: continues recursion into children
    """
    return _collect_no_decay(model, prefix="")


def _collect_no_decay(module: nn.Module, prefix: str) -> set:
    no_decay_names = set()

    if hasattr(module, "no_decay_deep"):
        for name in module.no_decay_deep():
            no_decay_names.add(f"{prefix}{name}" if prefix else name)
        return no_decay_names

    if hasattr(module, "no_decay"):
        for name in module.no_decay():
            no_decay_names.add(f"{prefix}{name}" if prefix else name)

    for child_name, child in module._modules.items():
        if child is None:
            continue
        child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
        no_decay_names |= _collect_no_decay(child, child_prefix)

    return no_decay_names


def build_param_groups(model: nn.Module, optimizer_params: dict, no_decay_names: set = None) -> list:
    """Build optimizer param groups with no-weight-decay splitting.

    Returns a list of param group dicts. If weight_decay is 0 or no
    no-decay parameters are discovered, returns a single group.

    If no_decay_names is provided, skips internal collection.
    """
    weight_decay = optimizer_params.get("weight_decay", 0.0)

    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}

    if weight_decay == 0.0 or not trainable:
        return [{"params": list(trainable.values()), **optimizer_params}]

    if no_decay_names is None:
        no_decay_names = collect_no_decay_params(model)

    valid_no_decay_names = set()
    all_param_names = dict(model.named_parameters())
    for name in no_decay_names:
        if name not in all_param_names:
            warnings.warn(
                f"[qPipeline] no_decay name '{name}' does not correspond to "
                f"any model parameter — skipping."
            )
        elif not all_param_names[name].requires_grad:
            pass
        else:
            valid_no_decay_names.add(name)

    if not valid_no_decay_names:
        return [{"params": list(trainable.values()), **optimizer_params}]

    decay_params = []
    no_decay_params = []
    for name, param in trainable.items():
        if name in valid_no_decay_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    shared_params = {k: v for k, v in optimizer_params.items() if k != "weight_decay"}
    return [
        {"params": decay_params, **optimizer_params},
        {"params": no_decay_params, "weight_decay": 0.0, **shared_params},
    ]
