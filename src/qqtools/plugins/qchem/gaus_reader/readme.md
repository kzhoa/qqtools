# Gauss Reader
A Python module for parsing Gaussian 16 output files.

>maybe g09 is also supported?

# Currently Supported Calculation Types

qq:

This module has been tested with the following task types:
- Single Point (SP):`#SP theory/basis force`  
- Optimization + Single Point: `#P theory/basis opt freq` 

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


# Contributing

We adopt a rule-driven design philosophy for the reader class, where each computational task corresponds to a set of rules.

To add support for more calculation types, you can extend the module by defining a custom set of parsing rules. 

Refer to the existing source code for examples. 

