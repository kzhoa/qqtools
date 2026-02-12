from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

from .plot_utils import plot_dict_distribution


class qScalaDict:
    """
    maps a key to scala
    """

    def __init__(self, d):
        assert all([isinstance(v, (int, float)) for v in d.values()])
        self.d = {k: v for k, v in d.items()}

    def __gt__(self, other):
        """Override '>' for comparison."""
        if isinstance(other, (int, float)):
            return [k for k, v in self.d.items() if v > other]
        return NotImplemented

    def __lt__(self, other):
        """Override '<' for comparison."""
        if isinstance(other, (int, float)):
            return [k for k, v in self.d.items() if v < other]
        return NotImplemented

    def __ge__(self, other):
        """Override '>=' for comparison."""
        if isinstance(other, (int, float)):
            return [k for k, v in self.d.items() if v >= other]
        return NotImplemented

    def __le__(self, other):
        """Override '<=' for comparison."""
        if isinstance(other, (int, float)):
            return [k for k, v in self.d.items() if v <= other]
        return NotImplemented

    def gt(self, other):
        if isinstance(other, (int, float)):
            return qScalaDict({k: v for k, v in self.d.items() if v > other})
        return NotImplemented

    def lt(self, other):
        if isinstance(other, (int, float)):
            return qScalaDict({k: v for k, v in self.d.items() if v < other})
        return NotImplemented

    def ge(self, other):
        if isinstance(other, (int, float)):
            return qScalaDict({k: v for k, v in self.d.items() if v >= other})
        return NotImplemented

    def le(self, other):
        if isinstance(other, (int, float)):
            return qScalaDict({k: v for k, v in self.d.items() if v <= other})
        return NotImplemented

    def loc(self, keys: List[str]):
        """Return a new qScalaDict based on condition."""
        return qScalaDict({k: self.d[k] for k in keys})

    def __len__(self):
        return len(self.d)

    def __repr__(self):
        return f"qScalaDict({self.d})"

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def items(self):
        return self.d.items()

    def to_dict(self):
        return self.d

    def plot(self, terminal_width=80, sort_by="value", show_percentage=True, show_bar=True):
        """Plot the qScalaDict using a smart distribution chart."""
        plot_dict_distribution(
            self.d,
            terminal_width=terminal_width,
            sort_by=sort_by,
            show_percentage=show_percentage,
            show_bar=show_bar,
        )

    def plot_bar(self, *args, **kwargs):
        """Plot the qScalaDict using matplotlib."""

        keys = list(self.d.keys())
        values = list(self.d.values())
        plt.bar(keys, values, *args, **kwargs)
        plt.xlabel("Keys")
        plt.ylabel("Values")
        plt.tight_layout()
        plt.show()
