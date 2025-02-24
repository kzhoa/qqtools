"""
General Log Reader
read lines under construction of ReadRule
use callback(lines)->value to define the parse method.
"""

import io
import pathlib
import re
import warnings
from typing import Iterable, List, Union


def extract_int(text) -> List[int]:
    matches = re.findall(r"[-+]?\d+", text)
    matches = [int(v) for v in matches]
    return matches


def extract_float(text) -> List[float]:
    """return [] if not matched"""
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    matches = [float(v) for v in matches]
    return matches


def is_open_file_handler(var):
    return isinstance(var, (io.TextIOWrapper, io.BufferedReader)) and not var.closed


# TODO
# end pattern for non-fixed nlines , to be implemented
class ReadRule(object):
    def __init__(
        self,
        name: str,
        pattern: str,
        nlines: int,
        skip_when_meet=0,
        callback=None,
        end_pattern=None,
    ):
        """__init__ _summary_

        Args:
            name (str): Defaults to None.
            pattern (str): pattern string.
            nlines (int or str): read nlines when meet pattern string. If str given, use eval().
            skip_when_meet (int): Defaults to 0.
            callback (optional): callback(read_lines)->parsedValue. Defaults to None.
        """
        self.pattern = pattern
        self.nlines = nlines  # -1 means read til end
        self.skip_when_meet = skip_when_meet
        self.callback = callback
        self.name = name
        if isinstance(nlines, int):
            assert nlines >= -1
        else:
            assert isinstance(nlines, str)

    def ensureNLines(self, read_results):
        if isinstance(self.nlines, str):
            self.nlines = eval(self.nlines, read_results)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def todict(self):
        return {
            "name": self.name,
            "pattern": self.pattern,
            "nlines": self.nlines,
            "skip_when_meet": self.skip_when_meet,
        }


class GeneralLogReader(object):
    """GeneralLogReader"""

    def __init__(self, rules: List[dict]):
        self.rules = [ReadRule(**rule) for rule in rules]
        self.cur_rule_idx = 0

    def get_rule(self, idx):
        if idx < len(self.rules):
            return self.rules[idx]
        else:
            return ReadRule(None, None, 0)  # null rule

    @property
    def cur_rule(self):
        return self.get_rule(self.cur_rule_idx)

    @property
    def next_rule(self):
        return self.get_rule(self.cur_rule_idx + 1)

    def read_file(self, file):
        if isinstance(file, (str, pathlib.PosixPath)):
            with open(file, "r") as f:
                return self.read_lines(f)
        elif is_open_file_handler(file):
            return self.read_lines(file)
        else:
            raise TypeError(f"Input TypeError: {type(file)}")

    def read_lines(self, lines: Union[list, Iterable]):
        results = {}
        self.cur_rule_idx = 0
        cur_rule = self.cur_rule

        has_meet = False
        skip_count = 0
        read_count = 0
        read_lines = []
        for i, line in enumerate(lines):
            if cur_rule.pattern is None or cur_rule.nlines == 0:
                # empty rule
                if len(self.rules) != 0 and self.cur_rule_idx != len(self.rules):
                    warnings.warn("Unexpected None or empty rule", UserWarning)
                break

            if cur_rule.pattern in line:
                has_meet = True

            # meet check
            if not has_meet:
                continue

            # skip check
            if skip_count < cur_rule.skip_when_meet:
                skip_count += 1
                continue

            # start read
            if read_count < cur_rule.nlines:
                read_lines.append(line.strip())
                read_count += 1

            # stop read, change to next rule
            if read_count == cur_rule.nlines:
                # callback and store result
                if cur_rule.callback is not None:
                    res = cur_rule.callback(read_lines)
                else:
                    res = read_lines
                results[cur_rule.name] = res

                # next rule
                self.cur_rule_idx += 1
                cur_rule = self.cur_rule
                # eval when change to next rule, copy()
                # cannot be omitted here to prevent from `eval() -> '__builtins__'` pollution
                cur_rule.ensureNLines(results.copy())
                has_meet = False
                skip_count = 0
                read_count = 0
                read_lines = []
        return results
