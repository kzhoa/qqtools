import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt

from .qscaladict import qScalaDict


class qList(list):
    def __gt__(self, other):
        """Greater than: returns a boolean list where each element in the list is greater than the corresponding element in other."""
        if isinstance(other, qList):
            # Compare each element in self with the corresponding element in other
            return [a > b for a, b in zip(self, other)]
        elif isinstance(other, (int, float)):  # If comparing with a scalar
            return [a > other for a in self]
        else:
            raise TypeError("Comparison not supported between instances of 'MyList' and '{}'".format(type(other)))

    def __lt__(self, other):
        """Less than: returns a boolean list where each element in the list is less than the corresponding element in other."""
        if isinstance(other, qList):
            # Compare each element in self with the corresponding element in other
            return [a < b for a, b in zip(self, other)]
        elif isinstance(other, (int, float)):  # If comparing with a scalar
            return [a < other for a in self]
        else:
            raise TypeError("Comparison not supported between instances of 'MyList' and '{}'".format(type(other)))

    def __eq__(self, other):
        """Equal to: returns a boolean list where each element in the list is equal to the corresponding element in other."""
        if isinstance(other, qList):
            # Compare each element in self with the corresponding element in other
            return [a == b for a, b in zip(self, other)]
        elif isinstance(other, (int, float)):  # If comparing with a scalar
            return [a == other for a in self]
        else:
            raise TypeError("Comparison not supported between instances of 'MyList' and '{}'".format(type(other)))

    def __ge__(self, other):
        """Greater than or equal to: returns a boolean list where each element in the list is greater than or equal to the corresponding element in other."""
        if isinstance(other, qList):
            # Compare each element in self with the corresponding element in other
            return [a >= b for a, b in zip(self, other)]
        elif isinstance(other, (int, float)):  # If comparing with a scalar
            return [a >= other for a in self]
        else:
            raise TypeError("Comparison not supported between instances of 'MyList' and '{}'".format(type(other)))

    def __le__(self, other):
        """Less than or equal to: returns a boolean list where each element in the list is less than or equal to the corresponding element in other."""
        if isinstance(other, qList):
            # Compare each element in self with the corresponding element in other
            return [a <= b for a, b in zip(self, other)]
        elif isinstance(other, (int, float)):  # If comparing with a scalar
            return [a <= other for a in self]
        else:
            raise TypeError("Comparison not supported between instances of 'MyList' and '{}'".format(type(other)))


class qListData:
    """
    List of dict:

    basic APIs:
    - counts: Returns the count of values associated with a given key in each dictionary.
    - count_distribution(key, skip_none=False): Returns the distribution (count) of values associated with the specified key in each dictionary.
    - get_list(key): Returns a list of values for the given key from each dictionary in the list.
    - get_set(key): Returns a set of unique values for the given key from each dictionary in the list.
    - get_map(key1, key2): Returns a dictionary mapping key1 to key2 for each dictionary in the list.
    - shuffle(seed): Shuffles the list of dictionaries. If seed is provided, the shuffle will be deterministic.

    Attributes:
    - data_list: A list of dictionaries.
    """

    def __init__(self, data_list):
        """
        Initialize qListData with a list of dictionaries.

        Args:
        - data_list (list): The list of dictionaries to store in the object.
        """
        self.data_list = data_list

        self.cnt_cache = {}

    def counts(self, key, skip_none=False):
        """
        Computes and returns the distribution (count) of values associated with the specified key in each dictionary.

        Args:
        - key (str): The key for which the distribution is calculated.
        - skip_none (bool): Whether to skip 'None' values. If True, 'None' values will not be counted.

        Returns:
        - qScalaDict: A dictionary where the keys are the unique values corresponding to the specified key,
                    and the values are the counts of these values across all dictionaries.
        """
        if not self.data_list:
            return {}

        if key in self.cnt_cache:
            return self.cnt_cache[key]

        if skip_none:
            counts = qScalaDict(Counter(d.get(key) for d in self.data_list if key in d))
        else:
            counts = qScalaDict(Counter(d.get(key) for d in self.data_list))

        # Cache the result
        self.cnt_cache[key] = counts
        return counts

    def count_distribution(self, key, skip_none=False):
        """
        Computes and returns the distribution of the counts of values for the specified key.

        Args:
        - key (str): The key for which the distribution of counts is calculated.
        - skip_none (bool): Whether to skip 'None' values. If True, 'None' values will not be counted.

        Returns:
        - qScalaDict: A dictionary where the keys are the count values and the values are the frequencies
                      of these counts across the datasets.
        """
        cnts = self.counts(key, skip_none)
        cnt_dist = qScalaDict(Counter(v for v in cnts.values()))
        return cnt_dist

    def get_map(self, key1, key2) -> dict:
        """
        Returns a dictionary mapping values from key1 to corresponding values from key2.
        """
        return {d[key1]: d[key2] for d in self.data_list}

    def get_list(self, key) -> list:
        return [d[key] for d in self.data_list]

    def get_set(self, key) -> set:
        return set([d[key] for d in self.data_list])

    def shuffle(self, seed):
        """
        Shuffles the list of dictionaries in place. If a seed is provided, the shuffle will be deterministic.

        Args:
        - seed (int or None): If provided, this will ensure that the shuffle is deterministic.
                               If None, the shuffle will be random.

        Returns:
        - list: A shuffled version of the data_list.
        """
        if seed:
            rng = random.Random(42)
        else:
            rng = random.Random()
        shuffled_list = self.data_list[:]  # shallow copy
        rng.shuffle(shuffled_list)
        return shuffled_list

    def plot_count_distribution(self, key):
        cnt_dist = self.count_distribution(key)
        names = list(cnt_dist.keys())
        counts = list(cnt_dist.values())

        plt.bar(names, counts)

        plt.title("Distribution of Counts")
        plt.xlabel("Cout Values")
        plt.ylabel("Count")

        plt.show()

    def loc(self, indices: Sequence[int | bool]):
        """
        Returns a new qList based on indices, which can be either:
        - A list of integers: Directly returns the elements at the corresponding indices.
        - A list of booleans: Returns elements where the corresponding boolean value is True.

        Args:
        - indices (Sequence[int | bool]): A sequence of indices or boolean values.

        Returns:
        - qList: A new qList containing the selected elements.
        """
        if all(isinstance(i, int) for i in indices):
            # If all indices are integers, select by index
            return qListData([self.data_list[i] for i in indices])
        elif all(isinstance(i, bool) for i in indices):
            # If all indices are booleans, select by boolean mask
            return qListData([item for item, include in zip(self.data_list, indices) if include])
        else:
            raise ValueError("Indices must be either all integers or all booleans.")

    def __getitem__(self, key):
        return qList([d[key] for d in self.data_list])
