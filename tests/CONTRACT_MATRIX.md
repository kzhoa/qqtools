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
| installed `qexp` CLI | [Yes](unit/qexp/test_cli_contract.py) | Pending | N/A | [Yes](e2e/qexp/test_cli.py) | validates installed wheel |
