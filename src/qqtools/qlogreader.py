"""
General Log Reader
read lines under construction of ReadRule


callback take up to 2 parameters
    - cb(lines) -> value ,  then results[rule.name] = value
    - cb(lines, results) -> results

skip_when_meet:  how many lines to skip, the pattern line stands for 1;
"""

# TODO
# support functional pattern check, to be implemented


import inspect
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


class ReadRule(object):
    def __init__(
        self,
        name: str,
        pattern: str,
        nlines: int = -1,
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
        self._nlines = nlines  # original pattern, -1 means read til end
        self.nlines = nlines  # -1 means read til end
        self.skip_when_meet = skip_when_meet
        self.callback = callback
        self.end_pattern = end_pattern
        self.name = name
        if isinstance(nlines, int):
            assert nlines >= -1
        else:
            assert isinstance(nlines, str)

        self.check_legal()

    def activateNLines(self, read_results):
        """activate nlines from string by formerly read variable"""
        if isinstance(self._nlines, int):
            return

        if isinstance(self._nlines, str):
            # qq: copy() cannot be omitted here to prevent from `eval() -> '__builtins__'` pollution
            self.nlines = eval(self._nlines, read_results.copy())
        else:
            raise TypeError(f"Unsupported type for nlines: {type(self._nlines)}")

    def evoke_callback(self, lines: List[str], results: dict):
        if self.callback is None:
            return lines
        sig = inspect.signature(self.callback)
        num_params = len(sig.parameters)
        if num_params == 1:
            res = self.callback(lines)
            results[self.name] = res
            return results
        elif num_params == 2:
            results = self.callback(lines, results)
            return results
        else:
            raise ValueError(f"callback function only take <=2 parameters, found {num_params}")

    def check_legal(self):
        # only welcome one stop method
        if self.end_pattern is not None:
            assert self.nlines == -1
        else:
            assert (
                self.nlines >= 0
            ), f"expect non-negative value, but found {self.nlines} in {self.name}"  # 0 means some temp Transit Station

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_dict(self):
        return {
            "name": self.name,
            "pattern": self.pattern,
            "nlines": self._nlines,
            "skip_when_meet": self.skip_when_meet,
            "end_pattern": self.end_pattern,
        }

    def __repr__(self):
        return str(self.to_dict())


class GeneralLogReader(object):
    """GeneralLogReader"""

    def __init__(self, rules: List[dict]):
        self.rules = [ReadRule(**rule) for rule in rules]
        self.cur_rule_idx = -1

    def get_rule(self, idx):
        if idx < len(self.rules):
            return self.rules[idx]
        else:
            return ReadRule(None, None, 0)  # null rule

    @property
    def cur_rule(self) -> ReadRule:
        return self.get_rule(self.cur_rule_idx)

    @property
    def next_rule(self) -> ReadRule:
        return self.get_rule(self.cur_rule_idx + 1)

    def to_next_rule(self, results):
        self.cur_rule_idx += 1
        self.cur_rule.activateNLines(results)
        return self.cur_rule

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
        self.cur_rule_idx = -1
        cur_rule = self.to_next_rule({})  # activate first rule

        has_meet = False
        skip_count = 0
        read_count = 0
        read_lines = []
        for i, line in enumerate(lines):
            if cur_rule.pattern is None or cur_rule.nlines == 0:
                # exit door
                if self.cur_rule_idx == len(self.rules):
                    break
                # empty rule
                if cur_rule.callback is None:
                    warnings.warn("Unexpected None or empty rule", UserWarning)

            if not has_meet and cur_rule.pattern in line:
                has_meet = True

            # meet check
            if not has_meet:
                continue

            # skip check
            if skip_count < cur_rule.skip_when_meet:
                skip_count += 1
                continue

            # end pattern check
            if cur_rule.end_pattern is not None:
                # then we donot use line count to control ending
                if cur_rule.end_pattern not in line:
                    read_lines.append(line.strip())
                    read_count += 1
                    continue

                # End pattern matched
                # callback and update result
                results = cur_rule.evoke_callback(read_lines, results)

                # turn to next rule
                cur_rule = self.to_next_rule(results)

                # reset states
                has_meet = False
                skip_count = 0
                read_count = 0
                read_lines = []
                continue

            # start read
            if read_count < cur_rule.nlines:
                read_lines.append(line.strip())
                read_count += 1

            # stop read, change to next rule
            if read_count == cur_rule.nlines:
                # callback and update result
                results = cur_rule.evoke_callback(read_lines, results)

                # turn to next rule
                cur_rule = self.to_next_rule(results)

                # reset states
                has_meet = False
                skip_count = 0
                read_count = 0
                read_lines = []
        return results
