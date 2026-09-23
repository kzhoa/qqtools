# Training-safe live observation validation

This records the 2026-09-23 feature-candidate evidence for the contracts in
[qexp live progress](../spec/qexp_live_progress.md), the
[product spec](../spec/qexp_product_spec.md), and the
[runtime spec](../spec/qexp_runtime_spec.md). The candidate remains on its
feature branch until every required promotion gate passes.

Focused validation passed for the v1 and v2 producer, protocol, projection,
selection, cleanup, qpipeline adapter, real tmux viewer, and two-rank collective
paths. The tmux fixture detached one read-only client while another remained,
stopped and killed a viewer, killed its isolated tmux server, then recreated the
viewer while the training Task remained running. The two-rank fixture compared
collective calls and evaluation values against its disabled baseline. Repository
governance, Ruff lint and formatting, and compatibility-registry validation
passed. Complete local preflight passed on the feature candidate after the
presentation and CLI characterization repairs. The configured promotion gate
has not run.

One qPipeline end-to-end run timed out after its child spent the existing
180-second limit in filesystem I/O wait. An unchanged rerun passed in 152
seconds. The timeout remains visible as environment instability; neither the
fixture timeout nor its assertions were relaxed.

The released 1.3.21 wheel was built and installed separately from the candidate.
In a 1.3.21-created project, its old Operation writer, recovery, and cleanup
completed while the candidate reader handled the old Task. An old agent
projected a candidate producer's unchanged v1 payload; an old reader displayed
v1 from a candidate agent while the candidate reader displayed the separate v2
metrics. A 1.3.21 machine cannot open a project newly initialized by this
candidate because of the existing `submission-group-publication-v1` root
capability. Rolling validation therefore uses an old-created project; this
root-capability limit is separate from the progress extension.

GPU measurements used reserved GPU 1 (NVIDIA H800, driver 550.144.03) and the
system Python 3.12 Torch 2.8.0+cu128 build for both old and candidate packages.
The project Python 3.13 environment has Torch 2.14.0+cu130, which cannot
initialize CUDA with this driver, so it was not used for GPU evidence. Seeded
one-rank and two-rank runs produced the same 12 steps and checkpoint SHA-256
`90a0836e0b94a67577f93048e2f098cac9a728fdf5a609a7bc6056edee881b9b`
under both packages. The two-rank run made 24 collectives per rank in both
packages; stdout/stderr file descriptors and exit outcomes also matched.

The current acceptance threshold is at most 3% paired median steady-state
throughput regression. Three comparable repetitions remain the default, but
excessive variation or uncertainty at the threshold is inconclusive. One
bounded, stable confirmation series may resolve an inconclusive cell; the
initial result remains part of the evidence. Each
matrix cell is capped below ten minutes; synchronized completed-step latency
is sampled in one representative cell. The nine worker/client comparisons
remain required, and a short run must exercise the actual qexp viewer. Accepted
GPU measurements require no unrelated compute processes. Valid prior results can be reused. The
benchmark should use qexp scheduling where available, with `nvidia-smi` checks
before and during measurements; qexp's own reservations do not establish that
unrelated jobs are absent. An isolated qexp agent attempt here was rejected
because another agent already held the host scheduler authority. This candidate
does not alter that agent's project inventory, so remaining short cells use an
explicitly selected idle GPU directly and verify isolation while running.

The first throughput probe ran 4096×4096 CUDA matmuls with 1, 4, and 16
concurrent producer processes and 0, 1, or 2 passive polling clients. Three
1.5-second repetitions per cell stayed within 1% median regression in both
old-first and candidate-first orders. A 1024×1024 short-step stress run lost
11–16% with one producer and per-step metric offers. Longer five-second,
five-repeat 16-process samples still varied by 3–6% within each version, even
with one Torch CPU thread per worker; the one-client median was 1.13% below
baseline. These clients poll files at the viewer cadence but are not the qexp
viewer. A subsequent GPU 1 run pinned to its NUMA-local CPU cores was stopped
when `nvidia-smi` showed another process holding about 3.4 GiB on GPU 1. Its
one-client baseline repetitions fell from roughly 2,530 to 1,580 steps/s;
those measurements are invalid for the isolated-GPU gate. The short-step cost
is a known limitation; these probes alone do not establish the 3% gate.

A later controlled synthetic run pinned workers to GPU 1's NUMA-local CPU cores
56–79 and used 4096×4096 matmuls, a 12-second window, three paired repeats,
and the fastest supported one-second producer cadence. At 16 workers, candidate
v2's paired median throughput regressions versus released 1.3.21 were 0.015%,
0.043%, and 0.194% for zero, one, and two polling clients. The harness
classified all three within the revised 3% target; every candidate run had at
least five in-window v1/v2 replacements. Separate CUDA-synchronized passes
measured full completed-step p99 latency. At four workers, the completed zero-
and one-client medians were 0.208% regression and 0.057% improvement. The
four-worker two-client cell was stopped when a separate GPU 1 process appeared
using about 8.3 GiB. CPU affinity is not exclusive reservation, and polling
clients are only a proxy for the real qexp viewer.

The remaining four cells ran on idle GPU 3 with 4096×4096 matmuls, an eight-second
window, three paired repeats, and one-second publication cadence. `nvidia-smi`
checks before, during, and after each series showed only the expected one or
four benchmark workers on GPU 3. No additional synchronized latency series was
run; the prior 16-worker passes already include completed-step p99 evidence.
Under the 3% target, candidate v2 versus released 1.3.21 had these paired
median regressions:

| Workers | 0 clients | 1 client | 2 clients |
| --- | ---: | ---: | ---: |
| 1 | -0.020% | -0.114% confirmed | 0.271% |
| 4 | 0.208% | -0.057% | -0.091% |
| 16 | 0.015% | 0.043% | 0.194% |

Negative regression means improvement. All nine listed series classify within
the revised 3% target. The first one-worker/one-client series is separately
**inconclusive**: two repetitions improved by about 0.2%, but one regressed by
8.587%, giving an 8.853-percentage-point range. A single limited confirmation
series had paired regressions of 0.128%, -0.424%, and -0.114%, range 0.552
points, and classified within the target. The initial noisy result is retained;
it is not relabeled as passing. This resolves that cell under the revised
one-confirmation rule, with residual host-noise risk noted. The earlier 1024×1024
per-step-offer stress loss remains a workload-specific limitation and is not
represented by this 4096×4096 qualification.

The real tmux viewer integration case completed in 5.76 seconds with two
read-only clients, pane recreation after viewer termination, tmux server
restart, and a continuing training Task. GPU 3 had no compute process at the
start of that case. This viewer case uses a CPU training fixture; the GPU
throughput clients above remain polling proxies, not real viewer throughput
measurements. The default 30-second producer interval was checked by the focused
runner/progress integration fixture; it was not a 30-second GPU throughput
matrix. The revised local performance and viewer evidence is sufficient
for feature-candidate acceptance, subject to the configured promotion gate.

The reproducible synthetic probe is
[`scripts/benchmarks/qexp_progress_gpu.py`](../../scripts/benchmarks/qexp_progress_gpu.py).
Run it with a CUDA-compatible Python that has Torch installed, an installed
released-package directory for `--baseline-package-path`, this checkout's
`src` directory for `--candidate-source-path`, and the allocated GPU index.
Its three profiles distinguish the released baseline, candidate with v2 off,
and candidate with v2 on. It pairs repetitions, requires in-window periodic
snapshot replacements and client overlap. By default it runs one separate
CUDA-synchronized pass in the four-worker/one-client cell (or the first listed
cell for a subset); `--latency-case WORKERS CLIENTS` selects another
representative cell. Its per-cell clock reserves cleanup time inside a ten-minute
limit. The report labels its polling-client proxy and does not claim to replace
an actual qexp-viewer rollout test.
