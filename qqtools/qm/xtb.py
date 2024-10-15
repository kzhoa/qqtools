"""
Usage example:
from qqtools.qm.xtb import xtbLogReader, xtbEnGradReader

results = xtbEnGradReader.read_file('xtb.log')
print(results)
# {'nAtoms': 12,
 'totalEnergy': -19.419013714798,
 'grad': array([[ 0.01263001,  0.00018626, -0.00228997],
       [ 0.0080979 , -0.00087593,  0.00555754],
       [-0.00628179, -0.004597  , -0.00335408],
       [-0.00084677, -0.00395188, -0.0011892 ],
       [-0.00029505,  0.00380362,  0.00133404],
       [-0.00628608,  0.00513616,  0.00075018],
       [ 0.0051048 , -0.00352793,  0.00040646],
       [-0.00573095, -0.00213501, -0.00210943]])
}
"""

import re

import numpy as np

from ..qlogreader import GeneralLogReader, extract_float

__all__ = ["xtbLogReader", "xtbEnGradReader"]


xtb_log_rules = [
    {
        "name": "totalEnergy",
        "pattern": ":: total energy",
        "nlines": 1,
        "skip_when_meet": 0,
        "callback": lambda lines: extract_float(lines[0])[0],
    },
    {
        "name": "dispersion",
        "pattern": ":: -> dispersion",
        "nlines": 1,
        "skip_when_meet": 0,
        "callback": lambda lines: extract_float(lines[0])[0],
    },
]

xtb_engrad_rules = [
    {
        "name": "nAtoms",
        "pattern": "Number of atoms",
        "nlines": 1,
        "skip_when_meet": 2,
        "callback": lambda lines: int(lines[0]),
    },
    {
        "name": "totalEnergy",
        "pattern": "The current total energy in Eh",
        "nlines": 1,
        "skip_when_meet": 2,
        "callback": lambda lines: float(lines[0]),
    },
    {
        "name": "grad",
        "pattern": "The current gradient in Eh/bohr",
        "nlines": "3*nAtoms",
        "skip_when_meet": 2,
        "callback": lambda lines: np.array(lines, dtype=float).reshape(-1, 3),
    },
]


xtbLogReader = GeneralLogReader(xtb_log_rules)
xtbEnGradReader = GeneralLogReader(xtb_engrad_rules)
