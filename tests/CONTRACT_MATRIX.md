# Public Behavior Contract Matrix

This matrix records behavior protection, not coverage percentage. Every `Yes`
cell links to the responsible test; `Pending` is an explicit coverage gap.

| Public behavior | Unit | Integration | E2E | Release E2E | Notes |
| --- | --- | --- | --- | --- | --- |
| epoch-suffix parser | [Yes](unit/plugins/qpipeline/test_epoch_suffix_resolver.py) | N/A | N/A | N/A | syntax and conversion boundaries |
| scheduler epoch suffix | [Yes](unit/plugins/qpipeline/test_epoch_suffix_resolver.py) | Pending | Optional | N/A | public scheduler configuration |
| runner eval/save epoch suffix | [Yes](unit/plugins/qpipeline/test_epoch_suffix_resolver.py) | [Yes](integration/qpipeline/runner/test_epoch_suffix_config_flow.py) | [Yes](e2e/qpipeline/test_step_epoch_suffix_training.py) | N/A | step-mode configuration flow |
| step-mode max-step inference | [Yes](functional/test_qpipeline/test_qpipeline_train_flow.py) | Pending | Optional | N/A | migration target remains functional |
| checkpoint resume | [Yes](functional/test_qpipeline/test_qpipeline_train_flow.py) | Pending | Pending | N/A | public-entry scenario still needed |
| installed `qexp` CLI | N/A | [Yes](integration/qexp/test_cli_contract.py) | N/A | [Yes](e2e/qexp/test_cli.py) | validates installed wheel; the prior unit-test link did not exist |

## qexp runtime verification matrix

Single-host tests prove the qexp protocol between independent local processes on one POSIX
filesystem. They do not certify NFS, Lustre, or any other cross-host filesystem implementation.
`Pending` is an intentional evidence gap, not a passing substitute.

| Requirement | Decision evidence | Local protocol evidence | Profile / lane | Owner |
| --- | --- | --- | --- | --- |
| machine-local authority isolation | [test_architecture_primitives.py](unit/qexp/test_architecture_primitives.py) | [test_test_architecture.py](integration/qexp/test_test_architecture.py) | hermetic / qexp-fast | qexp maintainers |
| revisioned CAS conflict preservation | [test_architecture_primitives.py](unit/qexp/test_architecture_primitives.py) | Pending | hermetic / qexp-fast | qexp maintainers |
| protocol fault, explicit interleaving, and replay envelope | [test_architecture_primitives.py](unit/qexp/test_architecture_primitives.py) | Pending | hermetic / qexp-fast | qexp maintainers |
| reference claim, fencing, and launch-authority invariants | [test_architecture_primitives.py](unit/qexp/test_architecture_primitives.py) | Pending | hermetic / qexp-fast | qexp maintainers |
| machine dispatch order, primary-probe admission, revision, and cursor plan | [test_machine_dispatch_plan.py](unit/qexp/test_machine_dispatch_plan.py) | [test_ready_index_dispatch.py](integration/qexp/test_ready_index_dispatch.py) | hermetic / qexp-fast | qexp maintainers |
| independent participant environment, checkpoint, and restart | N/A | [test_test_architecture.py](integration/qexp/test_test_architecture.py) | machine-lab / merge | qexp maintainers |
| CW-01 Submission Operation creation | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-02 Operation before Task staging | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-03 Partial Task staging | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-04 Worker addition before commit | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-05 Reservation before claim | Pending | [test_machine_dispatch.py](integration/qexp/test_machine_dispatch.py) | hermetic / qexp-fast | qexp maintainers |
| CW-06 Claim before Attempt | Pending | [test_test_architecture.py](integration/qexp/test_test_architecture.py) | machine-lab / merge | qexp maintainers |
| CW-07 Attempt before launch gate | Pending | [test_runner_pipeline.py](integration/qexp/test_runner_pipeline.py) | machine-lab / merge | qexp maintainers |
| CW-08 Launch gate before spawn | Pending | [test_runner_pipeline.py](integration/qexp/test_runner_pipeline.py) | OS / merge | qexp maintainers |
| CW-09 Process before metadata | Pending | [test_runner_pipeline.py](integration/qexp/test_runner_pipeline.py) | OS / merge | qexp maintainers |
| CW-10 Process exit before terminal publish | Pending | Pending | OS / merge | qexp maintainers |
| CW-11 Lease expiry before authorization | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-12 Lease expiry after authorization | Pending | Pending | hermetic / qexp-fast | qexp maintainers |
| CW-13 Orphan-token recovery | Pending | Pending | machine-lab / merge | qexp maintainers |
| CW-14 Successor fencing rejects old writer | Pending | [test_cross_host_claims.py](integration/qexp/test_cross_host_claims.py) | machine-lab / merge | qexp maintainers |
| CW-15 Worker removal race | Pending | Pending | machine-lab / merge | qexp maintainers |
| CW-16 Pause/cancel launch race | Pending | Pending | machine-lab / merge | qexp maintainers |
| CW-17 Recovery drain/remove race | Pending | Pending | machine-lab / merge | qexp maintainers |
| CW-18 Cancellation restart | Pending | Pending | machine-lab / merge | qexp maintainers |
| Protected workflows from installed wheel | N/A | N/A | [release-e2e](e2e/qexp/test_compatibility_contract.py) | qexp maintainers |
