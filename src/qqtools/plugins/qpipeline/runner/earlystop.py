""" """

from collections import OrderedDict
from copy import deepcopy
from typing import Mapping, Tuple


class EarlyStopper:
    """
    Early stop conditions

    Two stopping conditions:
    1.  Value hasn't improved (min/max) for x epochs within delta range
    2.  Value crosses lower/upper bounds

    Args:
        patiences (dict): defined the x epochs for condition 1
        min_delta (dict): defined the delta range for condition 1. defaults are 0.0
        mode (dict):  "min" or "max" (default "min")
        lower_bounds (dict): define the key and lower bound for condition 2
        upper_bounds (dict): define the key and lower bound for condition 2

    Example:
        >>> stopper = EarlyStopper(
                    patiences = {'val_mae': 10},
                    mode = {'val_mae': 'min' },
                    min_delta = {'val_mae': 1e-6},
        )
        >>> stop, stop_args, debug_args = stopper.step({"val_mae": 0.02})
        >>> if stop:
        >>>     print(stop_args)

        >>> # we also provide a small helper function `singleTargetEarlyStopper` for convenience
        >>> singleTargetEarlyStopper
    """

    def __init__(
        self,
        patiences: dict = {},
        mode: dict = {},
        min_delta: dict = {},
        lower_bounds: dict = {},
        upper_bounds: dict = {},
    ):

        self.patiences = deepcopy(patiences)
        self.modes = deepcopy(mode)
        self.min_delta = deepcopy(min_delta)
        self.lower_bounds = deepcopy(lower_bounds)
        self.upper_bounds = deepcopy(upper_bounds)

        self.counters = {}
        self.extremums = {}
        for key, pat in self.patiences.items():
            if key not in self.modes:
                self.modes[key] = "min"
            if self.modes[key] not in ["min", "max"]:
                raise ValueError(f"Invalid mode for {key}, must be 'min' or 'max'")
            if pat < 1:
                raise ValueError(f"Patience for {key} must be positive integer")
            if self.min_delta[key] < 0:
                raise ValueError("Delta must be non-negative")

            self.patiences[key] = int(pat)
            self.counters[key] = 0
            self.extremums[key] = None
            self.min_delta[key] = min_delta.get(key, 0.0)

        for key in self.min_delta:
            if key not in self.patiences:
                raise ValueError(f"patience for {key} should be defined")

    def step(self, metrics):
        return self.__call__(metrics)

    def __call__(self, metrics) -> Tuple[bool, str, str]:

        stop = False
        stop_args = "Early stopping:"
        debug_args = None

        # check whether key in metrics hasn't reduced for x epochs
        for key, pat in self.patiences.items():

            value = metrics[key]
            extremum = self.extremums[key]
            min_delta = self.min_delta[key]
            mode = self.modes[key]

            if extremum is None:
                self.extremums[key] = value
                continue

            if mode == "min":
                # is_better = value < extremum
                is_improved = value < (extremum - min_delta)
            else:  # max
                # is_better = value > extremum
                is_improved = value > (extremum + min_delta)

            if not is_improved:
                self.counters[key] += 1
                debug_args = f"EarlyStopping: {self.counters[key]}/{pat}"
                if self.counters[key] >= pat:
                    stop_args += f" {key} no improvement for {pat} epochs (mode={mode})"
                    stop = True
            else:
                self.extremums[key] = value
                self.counters[key] = 0

        for key, bound in self.lower_bounds.items():
            if metrics[key] < bound:
                stop_args += f" {key} is smaller than {bound}"
                stop = True

        for key, bound in self.upper_bounds.items():
            if metrics[key] > bound:
                stop_args += f" {key} is larger than {bound}"
                stop = True

        return stop, stop_args, debug_args

    def is_better(self, target, current_val, history_val):
        if history_val is None:
            return True

        mode = self.modes[target]
        if mode == "min":
            return current_val < history_val
        else:
            return current_val > history_val

    def state_dict(self) -> "OrderedDict[dict, dict]":
        return OrderedDict([("counters", self.counters), ("extremums", self.extremums)])

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.counters = state_dict["counters"]
        self.extremums = state_dict["extremums"]


def singleTargetEarlyStopper(
    target: str,
    patience: int,
    mode: str = "min",
    min_delta: float = 0.0,
    lower_bound: float = None,
    upper_bound: float = None,
) -> "EarlyStopper":
    """
    Helper to create an EarlyStopping instance for a single target metric.

    Args:
        target (str): The name of the metric to monitor.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        mode (str): One of {'min', 'max'}. In 'min' mode, training will stop if the metric stops decreasing.
                    In 'max' mode, training will stop if the metric stops increasing. Default is 'min'.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.0.
        lower_bound (float, optional): If provided, training will stop if the metric goes below this value.
        upper_bound (float, optional): If provided, training will stop if the metric goes above this value.

    Returns:
        EarlyStopping: A configured EarlyStopping instance.

    Example:
        >>> stopper = singleTargetEarlyStopper(
            targe='fmae',
            patience=10,
            mode='min',
            min_delta=1e-6,
        )
        >>>
    """
    patiences = {target: patience}
    modes = {target: mode}
    min_deltas = {target: min_delta}
    lower_bounds = {}
    upper_bounds = {}

    if lower_bound is not None:
        lower_bounds[target] = lower_bound
    if upper_bound is not None:
        upper_bounds[target] = upper_bound

    return EarlyStopper(
        patiences=patiences,
        mode=modes,
        min_delta=min_deltas,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
