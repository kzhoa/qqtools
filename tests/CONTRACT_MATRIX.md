# Public Behavior Contract Matrix

This matrix records only cross-layer and architecture-level behavior whose ownership is not
obvious from the test directory. Every linked test must exist; `Pending` is an explicit gap.

| Contract | Lowest sufficient evidence | Delivery boundary | Status |
| --- | --- | --- | --- |
| qpipeline epoch-suffix configuration | [YAML and runner Integration](integration/qpipeline/runner/test_epoch_suffix_config_flow.py) | source preflight | covered |
| qexp isolated scheduler-authority algorithm | [Integration](integration/qexp/test_resource_isolation.py) | local preflight | covered |
| qexp production default host authority | [installed CLI E2E](e2e/qexp/test_host_authority.py) | serial clean CI runner | covered |
| qexp durable-write crash boundaries | [Integration](integration/qexp/test_store_crash_boundaries.py) | local preflight | covered |
| qexp machine registration generations, eligibility, and logical-name reuse | [Integration](integration/qexp/test_machine_runtime.py), [CLI contract](integration/qexp/test_cli_contract.py) | qexp-integration / installed E2E | covered |
| qexp machine fresh-start, Project inventory/enrollment, image detachment, and global readiness | [Integration](integration/qexp/test_machine_enrollment.py) | [installed protected workflows](e2e/qexp/test_compatibility_contract.py) / artifact E2E | covered |
| qexp multi-participant claim, CAS, fencing, and cancel races | [machine lab](integration/qexp/test_machine_lab.py) | local preflight | covered |
| qexp deterministic recovery decisions | [Unit](unit/qexp/test_architecture_primitives.py) | local preflight | covered |
| qexp expanded seed and crash-point matrix | [Unit](unit/qexp/test_architecture_simulation.py) | local preflight | covered |
| qexp protected installed workflows | [installed E2E](e2e/qexp/test_compatibility_contract.py) | artifact E2E / release | covered |
| qexp agent lifecycle independence and restart reconciliation | [real-process Integration](integration/qexp/test_agent_lifecycle_independence.py), [installed workflow](e2e/qexp/test_agent_lifecycle_independence.py) | qexp-integration / installed E2E | covered: latest acceptance evidence in [qexp_lifecycle_acceptance.md](../docs/spec/qexp_lifecycle_acceptance.md) |
| qexp machine-local dynamic GPU admission policy, draining, and reservation linearization | [Policy Unit](unit/qexp/test_gpu_policy.py), [CLI and runtime Integration](integration/qexp/test_dynamic_gpu_policy.py) | unit / qexp-integration | covered |
| qexp Attempt-scoped live progress, reporting policy, and transparent qPipeline integration | [Policy Unit](unit/qexp/test_progress_policy.py), [Projection Unit](unit/qexp/test_progress_projector.py), [real-process Integration](integration/qexp/test_live_progress.py), [qPipeline real training](integration/functional/test_qpipeline/test_qexp_live_progress.py), [CLI contract](integration/qexp/test_output_format.py) | unit / qexp-integration / local preflight / release full training | covered |
| qexp read-only continuous Task/progress/log observation and optional Attempt-bound tmux plain logs | [Viewer Unit](unit/qexp/test_continuous_observation.py), [policy Unit](unit/qexp/test_tmux_policy.py), [submission policy Integration](integration/qexp/test_tmux_observation_policy.py), [CLI contract](integration/qexp/test_continuous_observation_cli.py), [real viewer processes](integration/qexp/test_continuous_observation_processes.py), [real enabled tmux pane](integration/qexp/test_tmux_plain_log_observer.py) | [installed CLI](e2e/qexp/test_cli.py) / artifact E2E | covered |
| qexp finite CLI JSON/human output boundary and bounded presentation | [output architecture](unit/qexp/output/test_architecture.py), [family renderers](unit/qexp/output/test_families.py), [CLI dispatch](integration/qexp/test_output_format.py) | unit / qexp-integration | covered |
| qexp unified submission modes, read-only preview, idempotent transaction, and provisional Group visibility | [submission contracts and crash recovery](integration/qexp/test_submission_contracts_and_transactions.py), [transaction protocol](integration/qexp/test_submission_protocol.py) | qexp-integration / installed CLI | covered |
| qexp bounded group-ready-member publication failure diagnostics across projection, Submission, and CLI surfaces | [diagnostic model Unit](unit/qexp/runtime/test_group_member_failure_diagnostics.py), [failure-path Integration](integration/qexp/test_submission_failure_diagnostics.py) | unit / qexp-integration | covered |
| qexp live indexed Task pagination and publication recovery | [Integration](integration/qexp/test_task_observation.py), [bounded index access](integration/qexp/test_task_observation_tree.py), [CLI contract](integration/qexp/test_task_observation_cli.py) | qexp-integration | covered |
| qexp consolidated resource/action CLI, typed configuration, daily Task/Group operations, bounded status, durable operation lookup, and Project selection precedence | [service Unit](unit/qexp/test_cli_consolidation_services.py), [Task operation Unit](unit/qexp/test_cli_daily_task_operations.py), [bounded status Unit](unit/qexp/test_cli_status_operations.py), [Group lifecycle Integration](integration/qexp/test_cli_consolidation_group_operations.py), [Project selection Integration](integration/qexp/test_cli_project_selection.py), [CLI tree Unit](unit/qexp/test_cli_consolidated_tree.py) | unit / qexp-integration / local preflight | covered |

## Maintenance rule

Update this file only when a listed contract, its lowest sufficient test layer, its delivery
boundary, or an explicit `Pending` gap changes. Ordinary local test additions do not belong here.
`scripts/checks/check_contract_matrix.py` verifies local test links and rejects retired lane terms.
