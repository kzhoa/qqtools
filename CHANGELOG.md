# History

## v1.2.28

- chore: squash dev/qq history (v1.2.8 through v1.2.27) onto main as a single commit for cleaner git lineage

## v1.2.27

- refactor: replace `LoopSignal.stop_message` with structured `stop_reasons: List[StopReason]` — each stop event now records source and message independently, preventing multiple abort reasons from overwriting each other
- feat: add `LoopSignal.request_stop(source, message)` as the single entry point for signaling training stop
- style: change `qDict.__repr__` indentation from tab to 2 spaces
- docs: add §3.1 "Default Monitored Target Contract" to qConfig spec — formalize `val_metric` as the canonical default monitored target and document per-component `target` as optional override
- refactor: replace `print` calls in `create_pipeline` with `pipe.logger.info` for consistent logging

## v1.2.26

- breaking: remove `QPipeline.prepare_optim` — optimizer and scheduler creation is now exclusively owned by `train_runner`
- breaking: `QPipeline` no longer holds `self.optimizer` or `self.scheduler` attributes
- feat: `train_runner` is now fully self-contained — when called independently (without QPipeline), it auto-creates optimizer, scheduler, and resolves epoch-suffix config from `args`
- feat: add `logger` parameter to `train_runner`, `evaluate_runner`, and `infer_runner` — QPipeline passes its logger so all diagnostics flow into the same `debug.log`
- feat: QPipeline creates `qLogger` in `_ensure_runtime_ready` (after `prepare_env` determines `log_dir`), replacing all `main_print` calls with structured logger output
- feat: add no-weight-decay parameter group auto-discovery — models can implement `no_decay()` or `no_decay_deep()` convention methods on `nn.Module` submodules to automatically split optimizer param groups into decay and no-decay groups
- feat: auto-reset training metrics (avg_bank) after warmup phase completes, using self-removing event listener on `on_train_batch_end`
- refactor: move epoch-suffix resolution from QPipeline into `train_runner` config standardization phase, exposed as `standardize_epoch_suffixes` in `runner_utils.epoch_suffix`
- refactor: `train_runner` validates illegal `scheduler` without `optimizer` combination
- refactor: add `remove_listener` to `EventDispatcher` and `RunningAgent` for symmetric listener lifecycle management
- refactor: remove dead-code forwarding methods `_run_evaluation` and `_run_evaluation_loop` from `RunningAgent`
- refactor: remove `main_print` and `rank_zero_only` from qpipeline module surface
- test: add 22 functional tests covering no-decay traversal algorithm, param group construction, edge cases (frozen params, non-existent names, zero weight_decay no-op, None children), and end-to-end optimizer integration
- test: add functional coverage for warmup avg_bank reset (listener registration, self-removal, reset behavior, warmup-period persistence)
- docs: document `no_decay` / `no_decay_deep` convention methods in qpipeline readme

## v1.2.25

- feat: add epoch-suffix auto-compute for qpipeline step-mode fields — config values like `T_max: 0.5epoch` or `eval_interval: 1epoch` are automatically converted to optimizer step counts based on `len(train_loader)` and `accum_grad`
- feat: implement `EpochSuffixResolver` with full validation for `step_on`/`run_mode` constraint matrix, warmup coupling warning, and DDP `all_gather` consistency check
- test: add 43 functional tests covering epoch-suffix resolution, `step_on=valid_end` rejection, `run_mode=epoch` field restrictions, warmup coupling detection, and plateau scheduler inference
- docs: add pitch for epoch-suffix auto-compute design and resume silent-override detection

## v1.2.24

- feat: add framework-owned qpipeline dotted CLI config overrides with strict validation, type-aware coercion, bool-safe flag-only handling, and post-YAML highest-precedence application
- docs: document qpipeline dotted override usage and CLI constraints in plugin readme and root README
- test: add functional coverage for dotted override parsing, reserved-key rejection, path/type safety, negative-value handling, and patch-parser interaction

## v1.2.23

- refactor: unify qPipeline into mode-based API — replace boolean `train` flag with explicit `mode` parameter (`"train"` / `"test"`) and rename `general_train` to `create_pipeline`
- refactor: introduce lazy `_ensure_runtime_ready` initialization in qPipeline for deferred env/task/model setup
- refactor: extract `_place_model` helper and add `evaluate_runner` entry point
- refactor: introduce `Stage` enum (`TRAIN`, `VAL`, `TEST`) in qpipeline runner to replace raw string `stage` literals and remove the redundant `stage="eval"` from lifecycle events (`eval_start`, `eval_end`, `validation_end`)
- feat: add `init_train_model` task hook for custom model weight initialization in qpipeline, with precedence over `args.init_file` and double-init warning when `ckp_file` is also set
- fix: correct `pipe_middle_ware` invocation to use `self.task` with `has_implemented` guard instead of raw `_opt_impl` dict access
- fix: correct qtriplets unit test expectations that were wrong since initial commit — fix periodic offset setup (combined offset must be non-zero), fix manual graph expected values, and fix no-chain test to use a truly disconnected graph
- fix: replace mock-based eval listener in `test_runner_accum_grad` with direct context capture for `on_eval_start`
- fix: remove `sys.modules` purge in `test_progress_render_mode` that caused cross-test pollution

## v1.2.22

- fix: keep qpipeline step-mode epoch-end summaries logging `[train]` metrics even when no evaluation fires at the boundary by normalizing epoch train aggregates into `train_*` results
- test: add functional coverage for step-mode epoch summaries with oversized `eval_interval` and no boundary evaluation

## v1.2.21

- breaking: formalize qpipeline event-context state handling so `context.runner.run_state` is exposed directly as the live `RunningState`, while `signal` remains the default write-back channel
- feat: split qpipeline runner events into typed public/internal contexts under `runner.events` and add explicit contracts for validation, checkpoint, progress, and loss-computed payloads
- refactor: remove legacy `EventContext` / `EventType` plumbing, precompile event emitters at dispatcher init, and drop runtime frozen-state wrapping from the qpipeline event path
- docs: document qpipeline event context access rules, internal `on_epoch_start_internal` usage, and the intended freedom to mutate `run_state` and other payload objects when callers explicitly choose to
- test: add functional coverage for typed event contracts, live `run_state` behavior, pre-backward loss hooks, and updated lifecycle bridge semantics

## v1.2.20

- fix: return stable graph-batch `num_graphs` metadata for graph-only batches and reserve the field from sample payload collision

## v1.2.19

- feat: formalize supported qpipeline task lifecycle hooks with explicit snake_case hook contracts bridged through runner listener registration
- docs: document the official qpipeline task lifecycle hook surface and supported runner listener events
- test: add functional coverage for task lifecycle hook bridging, early-stop callbacks, and unsupported legacy hook names

## v1.2.18

- fix: suppress rich/tqdm progress rendering on non-rank-0 DDP processes to prevent garbled console output in multi-GPU training

## v1.2.17

- feat: add qexp task `working_dir` support across submit/resubmit flows, batch manifests, subprocess execution, and tmux window startup directories
- fix: allow qpipeline step mode to infer `max_steps` from `max_epochs` and loader length, including accumulated-gradient runs, while preserving secondary epoch stopping boundaries

## v1.2.16

- feat: add `use_ctx` decorator for scoped runtime context mutation
- docs: document the `qt.ctx` mutation contract in README
- test: add unit coverage for decorator-based context mutation flows

## v1.2.15

- feat: expose `qt.ctx` as a first-class package instance for scoped runtime context access
- fix: remove `qcontext` package-init circular import and isolate fresh `ContextVar` contexts from shared default-state pollution
- test: add unit coverage for qcontext initialization, scoped merging, reset, and fresh-context isolation

## v1.2.14

- feat: restore qpipeline step mode secondary epoch boundary support so `max_epochs` can again act as an optional secondary stop limit while `max_steps` remains required
- feat: restructure qpipeline loss specs to support RMSE alongside the refreshed loss-entry configuration flow

## v1.2.13

- feat: allow `run_mode=step` to treat `max_epochs` as an optional secondary stopping boundary while keeping `max_steps` required
- fix: harden qexp machine GPU live view with safer agent/observer handling and add regression coverage for the live machine status path

## v1.2.12

- docs: archive completed qexp truth-domain and derived-index pitch documents under `docs/archive/pitch`
- docs: keep unfinished qexp machine-gpu-view-repair and PTY stdio contract proposals active under `docs/pitch`

## v1.2.11

- breaking: remove qexp v1 compatibility shims and the transitional `qqtools.plugins.qexp.v2` package surface; only the unified shared-root qexp API/CLI remains
- feat: add `qexp resubmit` to replace one terminal non-batch task in place with the same `task_id`, persist resubmit operation truth, and surface unfinished replacement state through `inspect`
- feat: add `qexp doctor repair` to converge unfinished resubmit operations and repair batch metadata inconsistencies in one recovery entrypoint
- fix: reconcile qexp batch commit-state truth so incomplete `preparing` batches are repaired to committed or aborted with refreshed batch summaries and indexes
- feat: fail qpipeline runs with `reason=nan_detected` when periodic eval/save boundaries observe NaN training loss
- feat: synchronize NaN-failure signals across DDP ranks and report source ranks from rank0 logs before unified failed exit
- docs: sync qexp README/manual/specs with the unified shared-root package surface, project-root `.qexp` contract, `group` semantics, `resubmit`, and `doctor repair`
- docs: add qpipeline timing manual and update log-format docs for periodic NaN interception behavior
- test: add qexp coverage for unified package surface, doctor integrity/repair flows, batch commit-state reconciliation, and resubmit transaction semantics
- test: add functional coverage for NaN terminal events, periodic interception, and distributed NaN-failure signal synchronization

## v1.2.10

- breaking: qpipeline standard config field `runner.keep_latest_ckp` is replaced by `runner.checkpoint.regular_latest_only`
- feat: qpipeline regular checkpoints now default to keeping only the latest file unless `runner.checkpoint.regular_latest_only: false`
- feat: upgrade qexp agent lifecycle to machine-scoped `active` / `draining` / `idle` semantics with `state/agent.json` as the single lifecycle source of truth
- feat: add formal qexp lifecycle contracts for machine worksets, agent snapshots, and machine summary caches
- fix: prevent on-demand qexp agents from auto-exiting while the current machine still owns `running` task responsibility
- fix: stop cross-machine qexp status and orphan repair flows from misclassifying healthy remote agents via local PID probes
- docs: update qConfig docs and standard config generator output for `runner.checkpoint.regular_latest_only`
- docs: sync qexp manual and runtime/product specs with the final lifecycle contract and lifecycle-state observability rules
- test: add coverage for regular checkpoint latest-only rotation and config parsing
- test: add qexp coverage for lifecycle state transitions, cross-machine observer behavior, orphan repair, and contract serialization

## v1.2.9

- feat: add `qexp use` command for CLI context persistence so `--shared-root`, `--machine`, and `--runtime-root` no longer need to be repeated on every command
- feat: `qexp init` now auto-saves CLI context after successful initialization
- feat: `_resolve_cfg` fallback chain extended to flags → env vars → context file (`~/.qqtools/qexp-context.json`) → error
- feat: log learning rate in qpipeline eval summary blocks and summary-table headers
- docs: refresh qpipeline log-format examples to show eval-summary learning rate output
- test: add eval-summary formatter coverage for learning-rate rendering

## v1.2.8

- fix: handle libtmux `ObjectDoesNotExist` exception in tmux session/window lookup to support newer libtmux versions where `.get()` raises instead of returning `None`

## v1.2.7

- breaking: qexp now uses the shared-root engine as the only supported runtime
- feat: qexp — shared-root multi-machine experiment queue with explicit machine identity, CAS-based concurrency, batch support, retry lineage, on-demand agent, and scheduling events
- feat: qexp subcommands: `init`, `submit`, `cancel`, `retry`, `batch-submit`, `batch-retry-failed`, `batch-retry-cancelled`, `list`, `inspect`, `top`, `batches`, `batch`, `machines`, `logs`, `clean`, `agent start/stop/status`, `doctor verify/rebuild-index/repair-orphans/cleanup-locks`
- breaking: qexp v1 single-machine engine has been removed
- docs: add qexp user manual and release workflow checks that publish changelog-backed release notes

## v1.2.6

- breaking: `train_runner().early_stopped` is now strictly derived from `terminal_event.reason == "early_stop"`; user interruption is exposed as `terminal_event.status=\"stopped\"` and `reason=\"user_interrupt\"`
- feat: add explicit `terminal_event` and `TrainRunnerResult` typed return contracts for qpipeline runner exits
- feat: add explicit epoch-result metric provenance in qpipeline logs with `source=current_eval|latest_eval_reuse|missing`
- fix: emit one unified terminal event across normal finish, early stop, user interrupt, OOM, and generic exception paths
- docs: add qpipeline log-format documentation and phase planning notes for terminal-event and epoch-result semantics
- test: add functional coverage for terminal-event classification and epoch-result provenance reporting

## v1.2.5

- breaking: `train_runner().early_stopped` now means only `terminal_event.reason == "early_stop"`; user interrupts must be consumed from `terminal_event.status/reason`
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
