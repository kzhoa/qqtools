# Public Behavior Contract Matrix

This matrix records only cross-layer and architecture-level behavior whose ownership is not
obvious from the test directory. Every linked test must exist; `Pending` is an explicit gap.

| Contract | Lowest sufficient evidence | Delivery boundary | Status |
| --- | --- | --- | --- |
| qpipeline epoch-suffix configuration | [Integration](integration/qpipeline/runner/test_epoch_suffix_config_flow.py) | [E2E](e2e/qpipeline/test_step_epoch_suffix_training.py) | covered |
| qexp isolated scheduler-authority algorithm | [Integration](integration/qexp/test_resource_isolation.py) | local preflight | covered |
| qexp production default host authority | [installed CLI E2E](e2e/qexp/test_host_authority.py) | serial clean CI runner | covered |
| qexp durable-write crash boundaries | [Integration](integration/qexp/test_store_crash_boundaries.py) | local preflight | covered |
| qexp machine registration generations, eligibility, and logical-name reuse | [Integration](integration/qexp/test_machine_runtime.py), [CLI contract](integration/qexp/test_cli_contract.py) | qexp-integration / installed E2E | covered |
| qexp multi-participant claim, CAS, fencing, and cancel races | [machine lab](integration/qexp/test_machine_lab.py) | local preflight | covered |
| qexp deterministic recovery decisions | [Unit](unit/qexp/test_architecture_primitives.py) | local preflight | covered |
| qexp expanded seed and crash-point matrix | [Unit](unit/qexp/test_architecture_stress.py) | local preflight | covered |
| qexp protected installed workflows | [installed E2E](e2e/qexp/test_compatibility_contract.py) | artifact E2E / release | covered |
| qexp agent lifecycle independence and restart reconciliation | [real-process Integration](integration/qexp/test_agent_lifecycle_independence.py), [installed workflow](e2e/qexp/test_agent_lifecycle_independence.py) | source preflight / qexp-integration / installed E2E | Pending: final requirement audit and latest-change verification; see [acceptance evidence](../docs/spec/qexp_lifecycle_acceptance.md) |

## Maintenance rule

Update this file only when a listed contract, its lowest sufficient test layer, its delivery
boundary, or an explicit `Pending` gap changes. Ordinary local test additions do not belong here.
`scripts/checks/check_contract_matrix.py` verifies local test links and rejects retired lane terms.
