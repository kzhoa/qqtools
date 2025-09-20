"""
handle list of dict datas
"""

from typing import Dict, List, Optional, Tuple, Union


def filter(iterable: List[Dict], key, value) -> List[Dict]:
    res = []
    for d in iterable:
        if d[key] == value:
            res.append(d)
    return res if len(res) > 0 else None


def find(iterable: List[Dict], key, value) -> Dict:
    for d in iterable:
        if d[key] == value:
            return d
