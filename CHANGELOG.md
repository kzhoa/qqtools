# History

## v1.2.5

- breaking: qpipeline non-plateau schedulers now default to stepping on completed optimizer updates instead of `on_validation_end`
- feat: add `optim.scheduler_params.step_on` to control non-plateau scheduler stepping with `optimizer_step` or `valid_end`; plateau remains validation-driven
- fix: align qpipeline train-event LR reporting with post-step scheduler state so batch/progress listeners observe the current learning rate
- docs: update qConfig docs and add an ADR for scheduler stepping semantics and `step_on`
- test: add functional coverage for scheduler policy validation/defaulting, qcgen scheduler prompts, and LR event timing under accumulation

## v1.2.4

- feat: add `runner.accum_grad` support across qpipeline runtime, qConfig schema, and qcgen prompts
- fix: align step-mode eval/save/global-step semantics with completed optimizer updates under gradient accumulation
- fix: weight accumulated gradients by loss sample counts so uneven micro-batches match grouped-batch updates
- refactor: enforce mutual-exclusive runner boundaries so epoch mode uses `max_epochs` and step mode uses `max_steps`
- refactor: remove unused `qoptim` module exports from package surface
- docs: document accum-grad semantics, run-boundary policy, and add ADR for optimizer-step semantics
- test: add functional coverage for accum-grad scheduling, checkpoint/eval triggers, qcgen schema rules, and qpipeline forwarding

## v1.2.3

- feat: add `qbalanced_partition` as a data helper for fast NumPy-based balanced bucket assignment
- feat: add `qtriplets` as a formal pure-PyTorch graph utility under `qqtools.torch`
- feat: expose new data and torch helpers through package exports
- fix: remove invalid Rich progress-bar f-string syntax in runner progress rendering
- refactor: refresh qpipeline runner internals and related utility modules
- refactor: update yaml/config loader internals and package wiring
- test: add benchmark coverage for balanced partition strategies
- test: add CUDA benchmark coverage for `qtriplets`
- test: add unit coverage for `qtriplets`
- test: refresh functional runner / qpipeline / demo test suites and benchmark scripts

## v1.2.2

- feat: add eval summary formatter

## v1.2.1

- feat: add sheet log listener
- feat: auto offload model when evaluation
- feat: progress bar slide to bottom
- refactor: max epochs and steps remove from runstate while preserve in config
- refactor: qema update fn
- fix: separate progress tick with update table to keep real batch idx as completed
- tests: cover ckp and evaluation trigger

## v1.2.0

- refactor: handly refresh live progresstracker
- fix: evaluation now trigger correctly with eval_interval under epoch mode

## v1.1.34

- feat: add qconfiggen and cli command
- fix: qdict copy now correctly pass defaultfunction
- refactor: change jit.script to compile for qscatter
- refactor: update runner package to support step mode
- docs: qconfig schema to qcgen
- docs: qconfig format under docs/

## v1.1.33

- feat: add lmdb env proxy
- feat: enhance qlist with plot
- feat: separate key types inference from graph collate fn
- feat: allow qdictdataloader to cached key types for graph data
- feat: add gdown into fetch module
- fix: LazyImport support subscription operator for typing annotations

## v1.1.32

- feat: add calc refe fast implementation
- fix: qtimer cuda mode record end before sync
- test: add speed test for scatter mean operator
- test: add speed test for calc refe implementations

## v1.1.29

- feat: add to_graph_dataloader for qdictdataset
- feat: getitem support for lazyimport
- feat: add qLmdbDataset

## v1.1.28

- feat: add qread script under cli, support g16 mode
- feat: lazyimporterror for rich
- feat: g16 support electro properties
- fix: g16 reader now support jump table with optional rules
- fix: g16 reader now add end line to read lines

## v1.1.27

- feat: add plugin qchem
- feat: add hyperconnect
- fix: pipeline task opt impl check

## v1.1.26

- feat: add attr-related functions is_override
- feat: under test: add plugin qpipeline

## v1.1.25

- chore: switch to pyproject.toml build system

## v1.1.22

- feat: to device support for qData, automatically adapt single/double precision
- doc: qDataset init

## v1.1.21

- feat: add fn binary_metrics for a set of useful indicators
- fix: qDataList check bool before int

## v1.1.20

- feat: add BatchList class

## v1.1.19

- feat: rank zero only
- fix: qdict init for kv tuple list

## v1.1.17

- feat: qdatalist plot_counts, support iter
- feat: add smart_combine for qDictDataset's collate fn
- refactor: data.qdatalist filename
- refactor: add staticmethod collate_dict_samples & graph in qDictDataset for convenience
- fix: qimport for matplotlib.pyplot

## v1.1.16

- fix: qtyping np.bool*
- fix: qData initialize for backward compatibility
- fix: qData init signature inconsist with parent which causes `copy.copy()` problem

## v1.1.12

- add: numpy type annotation in qtyping
- refactor: change qData `__init__` into kwargs style
- refactor: change qDictDataset interject name to raw_files_exist

## v1.1.10

- feat: qlistdata
- feat: avgbank
- feat: more qimports
- feat: set seed argument to naive split implementations
- feat: naive initialization for qdictdataset
- feat: naive & graph collate fn for qdictdataloader
- refactor: get_data_split moved to qsplit module

## v1.1.9

- add: numeric check and convert from str input
- add: alias for scatter mean&add
- add: staticmethod `qData.get*data_splits`
- fix: get nonlinearity discriminate str and callable activation

## v1.1.7

- add: more import common packages
- fix: qdict `_copy*` compatible with `copy.copy()`

## v1.1.6

- add: ensure numpy
- fix: unfreeze import relationship

## v1.1.5

- add: qdict fetch
- add: unfreeze module

## v1.1.4

- add: qmlp
- fix: qdict deepcopy memo

## v1.1.3

- fix: qdataset fetch logics

## v1.1.2

- feat: add functional donothing
- feat: recover support lr scheduler
- feat: add get_data_splits tool function
- feat: add freeze_module
- fix: qcontextprovider apply to general object instance
- refactor: find root return Path by default

## v1.1.1

- feat: global import nn.funtional as F
- feat: add nn submdule
- feat: add nn.DoNothing
- feat: add qcontextprovider

## v1.1.0

- refactor: module organization

## v1.0.15

- add: early stop support to ckp save&load
- add: pickle i/o
- add: lazyimport for class or function & transfer import common to lazy style

## v1.0.14

- add: find_root optional return type
- change: inherit loader to $base

## v1.0.13

- add: qDict from_args
- style: typing alias
- fix: qDict subclass copy bug with dataloader while pin_memory is True

## v1.0.12

- add: yaml inherit loader
- fix: find*root function

## v1.0.11

- add: qtyping

## v1.0.10

- add: find root

## v1.0.9

- donot sort keys by default when dump yaml

## v1.0.8

- ddp safe & time format

## v1.0.7

- add freeze random seed

## v1.0.6

- enhance timer

## v1.0.5

- refactor torch ops

## v1.0.3

- add assert

## v1.0.2

- add cuda synchronization for qtimer

## v1.0.0

- first version
- feat: import common support custom global
- feat: save ckp allow verbose control


# TODO

## Feature

- plugin/mol
  - is there a need to retain a unified intermediate form class?

## Tests

- tests/unit/timer
- tests/unit/lazyimporterrorproxy
- tests/functional/qlmdbdataset
