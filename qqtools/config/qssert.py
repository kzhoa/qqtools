from typing import Dict, List, Optional, Sequence, Tuple, Union
import qqtools as qt


def batch_assert_type(inpts, class_or_tuple):
    for obj in inpts:
        assert isinstance(obj, class_or_tuple), f" type of {type(obj)} inconsistent with {class_or_tuple}"
