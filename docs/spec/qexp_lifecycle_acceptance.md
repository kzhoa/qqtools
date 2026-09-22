---
doc_type: spec
status: completed
updated_at: 2026-09-12
archived_at:
---

# Lifecycle acceptance evidence

## Prospective acceptance limits

The earlier exploratory candidate runs preceded a complete gate-runtime budget declaration;
they cannot retroactively satisfy the pitch's ordering requirement. For subsequent acceptance
runs, freeze these limits: 15 seconds for return-to-convergence of four healthy-storage Attempts,
90 seconds for the representative lifecycle pytest call, 600 seconds for the complete qexp
Integration aggregate entrypoint (ordinary and lifecycle phases together), and 90 seconds for the
installed lifecycle pytest call. Build and
dependency installation time is reported separately. Exceeding a limit fails that acceptance run;
do not raise it to accept the observation. These prospective limits do not erase the earlier
process deviation.

Measured local launch/registration/process/exit evidence was approximately 9.8 KB total for four
short commands in the probe. This is a workload observation, not a cap: command and path lengths
affect size, and logs are separate. Retain unresolved records without an agent-downtime deadline;
remove them only after durable terminal truth and required claim/accounting convergence. Verified
local absence permits capacity release while keeping those evidence records.

Lifecycle-specific delivery is complete for the supported single-machine boundary. This record
distinguishes those gates from the repository-wide preflight, whose unrelated baseline failures
remain tracked separately. The governing requirement is
[the archived lifecycle pitch](../pitch/arxiv/060-qexp-agent-lifecycle-independence.md).

## Latest verification (2026-09-12)

These runs include the authority-storage and configured-mode fixes described in the latest change:

- `PYTHONPATH=src pytest -q tests/integration/qexp/test_agent_lifecycle_independence.py`:
  36 passed in 314.51 seconds, below the 600-second complete lifecycle budget.
- `tox run -e qexp-integration`: 575 passed in 479.15 seconds (483.51 seconds with tox),
  below the fixed 600-second gate budget.
- `tox run -e artifact-e2e -- tests/e2e/qexp/test_agent_lifecycle_independence.py -xq`:
  1 passed in 34.45 seconds (129.45 seconds including build/install), below the workflow budget.
- `tox run -e preflight`: stopped at the repository-wide Ruff check with 16 import-order findings
  in files outside this lifecycle change, before the Unit and Integration commands ran. The
  lifecycle files changed here pass Ruff and formatting checks.

The repository-wide preflight is a separate source gate. Its current static failure and the
historical Unit failures remain recorded below and are not used as lifecycle-specific acceptance
evidence.

## Historical probe evidence

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

Historical qexp Integration and installed-wheel runs are retained here for audit history. The
current gates are listed in **Latest verification** above.

Historical installed wheel gate before the current verification: `tox run -e artifact-e2e --
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

## Historical repository-wide gate notes

The following entries preserve earlier verification context. They are not outstanding
lifecycle-specific acceptance items; the current lifecycle gates are listed above.

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

Historical four-Attempt candidate mixed-work measurement:
`PYTHONPATH=src pytest tests/integration/qexp/test_agent_lifecycle_independence.py -k li07 -xqs`
passed the 2- and 4-project variants in 24.95 seconds. Return to terminal projection and capacity
release for the offline-completed Task measured 1.480 and 1.586 seconds respectively; remaining
live project reservations stayed isolated and all launch counters remained one. Equivalent
untouched-baseline measurement remains pending; the all-completed candidate is recorded below.

The historical four-offline-completed candidate variant subsequently passed:
`PYTHONPATH=src pytest tests/integration/qexp/test_agent_lifecycle_independence.py -k 'li07 and True' -xqs`:
1 passed in 9.69 seconds; return-to-terminal-and-capacity convergence was 1.578 seconds for all
four Attempts. Every runner observation existed before return, original Attempt IDs and zero
exit codes were checked, and each command's launch counter remained one. Equivalent untouched
baseline measurement remains pending.

| Requirement | Current evidence and remaining work |
| --- | --- |
| LI-01/02/04 | Deterministic progress/finish handshakes cover live training, offline success/failure, and SIGKILL; the current lifecycle file passes all 36 cases. |
| LI-03 | Real peer with natural lease expiry now covers live and completed cases; 2 passed in 20.38 seconds. |
| LI-05 | Authorization and process-registration interruption tests pass; strict identity and no-duplicate assertions retained. |
| LI-06 | Active-claim and orphan publication interruptions at Attempt, Task and reservation boundaries pass. Orphan replay preserves the committed result and rejects a wrong fencing token. |
| LI-07 | Four cold-start bindings preserve mixed live/finished reservation isolation; a separate running-agent scenario proves registry-revision discovery when two more bindings are added dynamically, followed by four-way offline completion recovery. |
| LI-08 | Missing/mismatched evidence and explicit supersession have diagnosis/retention coverage; termination boundary is covered by regression tests. |
| LI-09 | Installed CLI verifies runner exit observation before return, original Attempt, exit code, archive and reservation release. Latest installed gate: 1 passed in 10.09 seconds; rebuild after later production edits. |
| Global idle | Multi-binding ordering, consumed-binding empty registry, unresolved demand, failed-first-binding consumption, and pending repair operations have real-process coverage. |
| Gates | Exact required nodes are checked, including parameters. Ordinary qexp integration and the complete lifecycle matrix each run in controlled four-worker xdist phases; every node retains an isolated temp/runtime/tmux/authority namespace. Missing/skip/teardown/nonexecution/over-budget rejection has unit coverage. |
| Storage/capacity | Injected terminal-publication, pre-processing, and binding-load outages verify capacity release with evidence retention; broader filesystem corruption remains out of scope. |
| Budgets | Tracked probe measures baseline/candidate convergence and evidence size. Prospective gate budgets are enforced, retaining the 15-second convergence limit. Earlier budget-ordering deviation remains disclosed above. |
| Documentation | Product/runtime contract, precise mappings, evidence retention rules, lifecycle decision, and archived-pitch references are synchronized as of 2026-09-12. |

No elapsed-downtime cleanup is permitted for unresolved evidence. The current 15-second
convergence limit must not be raised to accept a failing run. Historical four-Attempt measurements
remain diagnostic; the current gate results above are the acceptance evidence.

## Aggregate gate evidence contract (2026-09-15)

The `qexp-integration` tox environment delegates to `scripts/qexp_integration_gate.py`. The
wrapper preserves the existing two four-worker phases and applies one 600-second soft wall-clock
budget to their combined collection, execution, and teardown path. Both phases normally finish so
their complete reports are retained; a successful over-budget run then fails as `budget_exceeded`.
A separate shared 1200-second hard timeout terminates a stuck process tree and reports
`hard_timeout`. Each phase receives a separate JUnit
XML file, stdout/stderr logs, collection manifest, and raw per-report timing JSON; `summary.json`
records phase exit codes, durations, soft-budget overrun, hard timeout, status, Python version, and
`GITHUB_SHA` when available. The
wrapper removes only its own known report files at startup, so a rerun cannot inherit stale phase
artifacts. A collection failure, worker
failure, timeout, or cleanup failure is non-zero and is not converted to success by a later passing
phase. CI profiling uploads the report directory on both success and failure. The wrapper adds
observability and budget aggregation only; it does not claim current-SHA CI acceptance until a run
on that SHA is archived here.

Two independent local runs of the final aggregate-gate implementation completed on 2026-09-15.
They used Python 3.13.12 and the same checkout, but the checkout still contained the uncommitted
Batch 1 changes, so these are local implementation evidence rather than fixed-SHA or CI
attestations:

- run 1: 95.70 seconds aggregate; ordinary 32.86 seconds (487 passed), lifecycle 62.84
  seconds (34 passed);
- run 2: 96.55 seconds aggregate; ordinary 32.66 seconds (487 passed), lifecycle 63.89
  seconds (34 passed).

Both runs collected 521 required nodes, reported no skips, and completed within the shared
600-second execution budget. The report directory is intentionally ignored because it contains
ephemeral local evidence; fixed-SHA CI artifacts remain the durable acceptance source.

The first profiling run for fixed SHA `687573d` (GitHub Actions run
`35060672167 <https://github.com/kzhoa/qqtools/actions/runs/35060672167>`) completed on
2026-09-16 and failed in the lifecycle phase: ordinary passed, while 32 of 34 lifecycle nodes
passed. The two failures were
`test_li08_mismatched_exit_evidence_is_retained_as_blocker` and
`test_finished_process_releases_capacity_while_publication_is_unavailable[_finalize]`; both
timed out waiting for durable process-observation evidence. The run uploaded the phase JUnit,
manifest, timing, and stdout/stderr artifacts. This is current-SHA failure evidence, not a reason
to relax the 15-second convergence budget or weaken the assertions; the failures remain open for
the next lifecycle-diagnostics batch.
