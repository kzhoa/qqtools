---
doc_type: spec
status: drafting
updated_at: 2026-09-09
archived_at:
---

# Lifecycle acceptance evidence

## Prospective acceptance limits

The earlier exploratory candidate runs preceded a complete gate-runtime budget declaration;
they cannot retroactively satisfy the pitch's ordering requirement. For subsequent acceptance
runs, freeze these limits: 15 seconds for return-to-convergence of four healthy-storage Attempts,
90 seconds for the representative lifecycle pytest call, 600 seconds for the complete qexp
Integration pytest call, and 90 seconds for the installed lifecycle pytest call. Build and
dependency installation time is reported separately. Exceeding a limit fails that acceptance run;
do not raise it to accept the observation. These prospective limits do not erase the earlier
process deviation.

Measured local launch/registration/process/exit evidence was approximately 9.8 KB total for four
short commands in the probe. This is a workload observation, not a cap: command and path lengths
affect size, and logs are separate. Retain unresolved records without an agent-downtime deadline;
remove them only after durable terminal truth and required claim/accounting convergence. Verified
local absence permits capacity release while keeping those evidence records.

Delivery remains pending. This record distinguishes executed checks from full acceptance of
[the lifecycle pitch](../pitch/qexp-agent-lifecycle-independence.md).

## Latest frozen-code verification

During these runs no runtime or test code was edited:

- `tox run -e qexp-integration`: 525 passed in 377.62 seconds (382.47 seconds with tox),
  below the fixed 600-second gate budget.
- `tox run -e artifact-e2e -- tests/e2e/qexp/test_agent_lifecycle_independence.py -xq`:
  1 passed in 27.38 seconds (105.62 seconds including build/install), below the workflow budget.

The global preflight Unit failures reproduced on the baseline remain unresolved. Acceptance
also requires acknowledging or otherwise resolving the budget-ordering deviation above; later
successful runs cannot change when the original limits were declared.

## Executed candidate checks

## Repeated baseline/candidate probe

The tracked reproducer is `tests/fixtures/qexp/lifecycle_probe.py`. It accepts explicit source
and isolated work roots, launches four real runner/guardian commands, waits for every durable
exit observation with the agent stopped, restarts, and measures all terminal projections plus
reservation release against the unchanged 15-second budget.

Commands (2026-09-09):

```sh
python tests/fixtures/qexp/lifecycle_probe.py --source-root /tmp/qqtools-lifecycle-baseline-VrtL4o/src --work-root /tmp/qqtools-lifecycle-baseline-VrtL4o/probe
python tests/fixtures/qexp/lifecycle_probe.py --source-root /mnt/c/Users/Administrator/proj/qqtools/src --work-root /tmp/qqtools-lifecycle-candidate-probe
```

| Source | Four offline completions | Seconds | Local evidence bytes | Launch counts |
| --- | --- | --- | --- | --- |
| Untouched d07461d6249a074ddd099bae35a1d6e8270ca2d0 | converged | 0.769835 | 9786 | 1,1,1,1 |
| Candidate working tree | converged | 1.927747 | 9783 | 1,1,1,1 |

Raw JSON remains under the printed probe work directories. This corrects the earlier unretained
baseline inference: ordinary intact offline completion succeeds on the baseline. These one-run
measurements do not establish a performance improvement, and the candidate was slower in this
sample. Correctness fixes are supported by the separately reproduced interruption failures.

Latest full qexp Integration gate: `tox run -e qexp-integration` passed 510 tests in
311.00 seconds (315.63 seconds including tox). This run includes the strict lifecycle gate,
the authorization partial-write regression fix, and scheduler-test resource isolation fix.
The separate global preflight remains failed for the baseline-reproduced Unit issues below.

Latest installed wheel gate after all production and test edits: `tox run -e artifact-e2e --
tests/e2e/qexp/test_agent_lifecycle_independence.py -xq` passed 1 test in 26.52 seconds
(104.64 seconds including build/install). Static lane and contract checks passed in that run.

Gate logic negative checks: `PYTHONPATH=src pytest tests/unit/test_lifecycle_gate.py -q`
passed five checks covering missing required cases, skip, failed teardown, nonexecution, and
successful complete execution.

### Partial baseline observation

The untouched `d07461d6249a074ddd099bae35a1d6e8270ca2d0` source was isolated under
`/tmp/qqtools-lifecycle-baseline-VrtL4o`. With one registered on-demand binding and no tasks,
the agent remained alive after 3 seconds (`true-idle exit gap reproduced`). This baseline is
kept as a command transcript in the session log; a tracked reproducer and complete baseline
measurements are still required. It is not release evidence.

- `PYTHONPATH=src pytest tests/integration/qexp/test_agent_lifecycle_independence.py -q`:
  18 passed, 124.67 seconds, before subsequent observation-validation and gate changes.
- Representative gate: `PYTHONPATH=src pytest --lifecycle-gate=representative
  tests/integration/qexp/test_agent_lifecycle_independence.py -k 'li01 or li02 or li04' -q`:
  4 passed, 34.55 seconds. Terminal truth and capacity release share a 15-second return budget.
- Installed workflow: `tox run -e artifact-e2e --
  tests/e2e/qexp/test_agent_lifecycle_independence.py -xq`: 1 passed, 29.14 seconds
  (118.88 seconds including wheel build/install). This preceded the final scheduler fixes.
- Installed workflow after the exit-observation/accounting assertions and installed-lane gate:
  `.tox/artifact-e2e/bin/python -E -m pytest --lifecycle-gate=installed -c
  tests/e2e/installed_artifact_pytest.ini tests/e2e/qexp/test_agent_lifecycle_independence.py -xq`:
  1 passed in 10.51 seconds.
- Final candidate lifecycle Integration: `PYTHONPATH=src pytest
  tests/integration/qexp/test_agent_lifecycle_independence.py -q`: 18 passed in 132.75 seconds.
- Runner, lease-hardening and scheduling regression files: 32 passed, 5.63 seconds;
  the subsequently added confirmed-termination regression passed independently.

These are historical candidate observations, not immutable build attestations. Re-run checks
affected by later changes before acceptance.

## Outstanding acceptance evidence

Full qexp Integration run started before the latest cancellation fix: `tox run -e qexp-integration`
passed 520 tests in 367.46 seconds (372.21 seconds including tox). The cancellation regression
was separately verified afterward. This run does not certify all subsequent production edits.

Full source gate attempted with `tox run -e preflight` (then reused its installed editable
environment with `--skip-pkg-install` after formatting fixes). Static/format/lane/matrix checks
passed. Unit stopped the lane: 1882 passed, 5 failed in 239.30 seconds. Failures were
`test_import_mypackage` (`libtmux` already in the shared test process) and four
`test_qlmdbdataset.py` cases reporting an LMDB environment already open in that process.
Their baseline attribution is not yet established; do not classify the entire preflight as
passing or silently exclude these tests. Later Integration commands were not reached.

Baseline attribution: `test_get_raw_blob_uses_global_multishard_indices_and_reports_errors`
also fails with the same LMDB duplicate-environment error when run by the preflight Python
with both `PYTHONPATH` and pytest `pythonpath` set to the untouched baseline source above.
The traceback resolves `qlmdbdataset.py` under that baseline directory. This establishes
that particular failure predates this implementation. The remaining three LMDB failures were
subsequently reproduced together against that source (3 failed in 22.07 seconds), and
`test_import_mypackage` also failed independently against the baseline source (1.55 seconds).
All five preflight failures reproduce without the candidate runtime changes in the same
preflight environment. This attribution does not make the source gate green.

Four-Attempt candidate mixed-work measurement:
`PYTHONPATH=src pytest tests/integration/qexp/test_agent_lifecycle_independence.py -k li07 -xqs`
passed the 2- and 4-project variants in 24.95 seconds. Return to terminal projection and capacity
release for the offline-completed Task measured 1.480 and 1.586 seconds respectively; remaining
live project reservations stayed isolated and all launch counters remained one. Equivalent
untouched-baseline measurement remains pending; the all-completed candidate is recorded below.

The four-offline-completed candidate variant subsequently passed:
`PYTHONPATH=src pytest tests/integration/qexp/test_agent_lifecycle_independence.py -k 'li07 and True' -xqs`:
1 passed in 9.69 seconds; return-to-terminal-and-capacity convergence was 1.578 seconds for all
four Attempts. Every runner observation existed before return, original Attempt IDs and zero
exit codes were checked, and each command's launch counter remained one. Equivalent untouched
baseline measurement remains pending.

| Requirement | Current evidence and remaining work |
| --- | --- |
| LI-01/02/04 | Deterministic progress/finish handshakes now cover live training, offline success/failure, and SIGKILL; representative gate 4 passed in 25.56 seconds. |
| LI-03 | Real peer with natural lease expiry now covers live and completed cases; 2 passed in 20.38 seconds. |
| LI-05 | Authorization and process-registration interruption tests pass; strict identity and no-duplicate assertions retained. |
| LI-06 | Active-claim and orphan publication interruptions at Attempt, Task and reservation boundaries pass. Orphan replay preserves the committed result and rejects a wrong fencing token. |
| LI-07 | Two active projects with mixed live/finished work and isolated reservations pass. |
| LI-08 | Missing/mismatched evidence and explicit supersession have diagnosis/retention coverage; termination boundary is covered by regression tests. |
| LI-09 | Installed CLI verifies runner exit observation before return, original Attempt, exit code, archive and reservation release. Latest installed gate: 1 passed in 10.09 seconds; rebuild after later production edits. |
| Global idle | Multi-binding ordering, consumed-binding empty registry, unresolved demand, and failed-first-binding consumption have real-process coverage. Pending repair operations need final inventory. |
| Gates | Exact required nodes are checked, including parameters. Missing/skip/teardown/nonexecution/over-budget rejection has unit coverage. Budgeted representative gate: 4 passed in 24.97 seconds. Required final source/release runs remain outstanding after production edits. |
| Storage/capacity | Injected terminal-publication, pre-processing, and binding-load outages verify capacity release with evidence retention; broader filesystem corruption remains out of scope. |
| Budgets | Tracked probe measures baseline/candidate convergence and evidence size. Prospective gate budgets are enforced, retaining the 15-second convergence limit. Earlier budget-ordering deviation remains disclosed above. |
| Documentation | Final product/runtime contract, precise mappings, evidence retention size/cleanup and lifecycle decision remain pending final audit. |

No elapsed-downtime cleanup is permitted for unresolved evidence. The current 15-second
convergence limit must not be raised to accept a failing run. Four-Attempt candidate measurements
do not replace the remaining gate-budget and release verification.
