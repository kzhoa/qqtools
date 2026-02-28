# Unit Test TODOLIST

Scope: identify modules not covered by `tests/unit` under local source (`PYTHONPATH=src`),
excluding `src/qqtools/plugins` from this list.

## Classification Rule

- **full**: modules under `src/qqtools/plugins/` (not listed in this file)
- **standard**: modules that depend on optional std packages in `pyproject.toml` (`lmdb`, `tqdm`, `requests`)
- **base**: modules that are not `full` and do not depend on the std optional packages above

## Base modules not covered (`0%` in `tests/unit`)

- None
## Standard modules not covered (`0%` in `tests/unit`)

- None


## Full Modules not covered

- `src/qqtools/cli/qcgen.py`
- `src/qqtools/cli/qread.py`
  - `src/qqtools/qlogreader.py`

- `src/qqtools/qm/ase.py`
- `src/qqtools/qm/rdkit.py`
- `src/qqtools/qm/units.py`
- `src/qqtools/qm/utils.py`
- 
## Notes

- Current `standard` modules detected in source are:
  - `src/qqtools/config/fetch/gdown.py`
  - `src/qqtools/config/qlmdb.py`
  - `src/qqtools/qimport.py`
  - `src/qqtools/torch/qdataset.py`
- They are not `0%` covered, so not included in the standard TODO list above.

## Latest Coverage Update (tests/unit)

- Updated by: `PYTHONPATH=src python -m pytest tests/unit --cov=qqtools --cov-report=json:coverage_unit_local.json -q`
- Base modules moved out of `0%` after new tests:
  - `src/qqtools/config/qtime.py` -> `100%`
  - `src/qqtools/torch/metrics/__init__.py` -> `100%`
  - `src/qqtools/torch/metrics/binary.py` -> `91%`
  - `src/qqtools/torch/nn/functional.py` -> `93%`
  - `src/qqtools/torch/nn/utils.py` -> `100%`
  - `src/qqtools/torch/qbenchmark.py` -> `95%`
  - `src/qqtools/utils/qlist.py` -> `100%`
- Additional progress:
  - `src/qqtools/torch/qdataset.py` -> `62%` (improved from ~45%)

## Latest Coverage Update (unit + qpipeline functional)

- Updated by: `PYTHONPATH=src python -m pytest tests/unit tests/functional/test_qpipeline --cov=qqtools --cov-report=json:coverage_all_local.json -q`
- Selected test suites: `84 passed`
- Total coverage: `38%`
- qpipeline key modules:
  - `src/qqtools/plugins/qpipeline/runner/runner.py` -> `81%`
  - `src/qqtools/plugins/qpipeline/entry_utils/scheduler.py` -> `59%`
  - `src/qqtools/plugins/qpipeline/entry.py` -> `56%`
