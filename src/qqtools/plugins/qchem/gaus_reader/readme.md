# Gauss Reader
A Python module for parsing Gaussian 16 output files.

>maybe g09 is also supported?

# Currently Supported Calculation Types

qq:

This module has been tested with the following task types:
- Single Point (SP):`#SP theory/basis force`  
- Optimization + Single Point: 
  - `#P theory/basis opt freq` 
  - We only return the last structure after optimization is completed.

Note on Electronic State:
- All calculations assume closed-shell systems​ (multiplicity = 1, singlet state).
- This applies to both neutral and charged systems where all electrons are paired.
- For open-shell systems (multiplicity ≥ 2), additional customized handling would be required.


# Quick Start

For Single Point Calculations Only
```python
from gaus_reader import create_g16_reader

g16Reader = create_g16_reader(opt=False)
```

For Optimization + Single Point Calculations
```python
from gaus_reader import create_g16_reader

g16Reader = create_g16_reader(opt=True)
```


# Harmonic frequencies

Both `create_g16_reader(opt=False)` and `create_g16_reader(opt=True)` now
include an optional `freq` rule:

```python
from qqtools.plugins.qchem import create_g16_reader

result = create_g16_reader(opt=False).read_file("output.log")
frequencies = result.get("freq")  # list[float], cm^-1; None if not returned
```

The rule reads from `Harmonic frequencies` through `Thermochemistry` and
extracts `Frequencies --` (ordinary, up to three modes per group) and
`Frequencies ---` (`Freq=HPModes`, up to five modes per group), preserving
negative values and mode order.
Fortran D/d and E/e exponents are supported. `Low frequencies ---` diagnostics
are not included. Mode numbers must be consecutive starting at 1; malformed or
non-finite values raise `ValueError`. Numbering is checked independently for
each format. When both formats occur in the same block, their mode counts and
values must agree within the rounding intervals implied by their printed
decimal precision; the high-precision list is returned once, without duplicates.
Either printing order is supported. Repeated numbering within a single format
or conflicting representations raises `ValueError`.
The terminating Thermochemistry line is
consumed, leaving subsequent zero-point correction, thermal corrections and
forces available to their existing rules.

The rule is optional: logs without a harmonic block retain the existing fields
without a `freq` key. A block truncated before Thermochemistry likewise does
not return `freq`; absence is not an empty or validated zero-mode spectrum.
Consumers that require frequencies must check for this key.

This sequential reader selects the harmonic block encountered at that point in
its existing rule sequence; it does not combine or select among multiple jobs
in concatenated logs. Successful extraction alone does not verify normal
termination, the expected 3N-6/3N-5 mode count, archive NImag consistency, or
correspondence with external coordinates. Dataset-specific validation must
perform those checks separately. Values retain Gaussian's printed precision.

# Contributing

We adopt a rule-driven design philosophy for the reader class, where each computational task corresponds to a set of rules.

To add support for more calculation types, you can extend the module by defining a custom set of parsing rules. 

Refer to the existing source code for examples. 

