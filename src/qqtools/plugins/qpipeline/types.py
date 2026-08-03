from enum import Enum

__all__ = ["Stage"]


class Stage(str, Enum):
    """Execution stage supplied by qpipeline runtime callers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
