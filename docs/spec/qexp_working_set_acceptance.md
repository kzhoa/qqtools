# qexp working-set delivery acceptance

Qualification snapshot: 2026-09-25.

The qualification used Linux, Python 3.13, the source checkout and isolated
runtime roots under `/tmp`. Filesystem caches were warm and uncontrolled. The
profile used real activation consumer records and working-set progress writes;
lease eligibility itself was recorded by an in-process stub so the measurement
isolated roster selection from registration storage latency.

The declared running-agent thresholds were a maximum of four dormant checkpoint
reads per scheduler cycle, 64 dormant registration renewals per heartbeat,
`ceil(N / 4)` checkpoint-poll rounds for a continuously running agent to observe
a wake, and a 100 ms p95 for one unchanged steady cycle. Startup and handoff are
reported separately because they intentionally perform O(N) work.

## Registered Project scale

The maintained command was:

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python \
  -m scripts.qualification.profile_qexp_working_set \
  --output /tmp/qexp-working-set-20260925-reverified \
  --projects 1 100 1000 --cycles 200 --soak-cycles 10000
```

| Registered Projects | Startup (s) | Five-lane retirement (s) | Steady mean / p95 (ms) | Project truth reads | Lease renewals | Wake rounds / bound | FD delta | RSS delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004 | 0.008 | 0.198 / 0.220 | 200 | 200 | 1 / 1 | 0 | 0 B |
| 100 | 0.470 | 1.245 | 0.822 / 1.028 | 800 | 12,800 | 25 / 25 | 0 | 0 B |
| 1,000 | 5.252 | 13.030 | 0.842 / 1.021 | 800 | 12,800 | 50 / 250 | 0 | 8,192 B |

Project truth reads stayed fixed between 100 and 1,000 dormant bindings. The
process retained one compact state record per registration, as allowed by the
protocol, while resident contexts fell to zero before the wake. Traced steady
memory deltas were 7,087, 43,287 and 1,266,903 bytes for 1, 100 and 1,000
registrations. The 1,000-project accelerated soak ran 10,000 cycles in 1.836
seconds. This is churn evidence and is not months of uptime evidence.

The wake-roster cursor had already advanced during the 200 steady cycles, so the
1,000-project wake was observed in 50 rounds; the enforced worst-case bound was
250 rounds. The normal running-agent latency envelope is that round bound times
the configured scheduler interval plus filesystem latency. A stopped on-demand
agent retains evidence but has no wall-clock wake promise.

A separate stress regression uses 1,000 real registration records with a
120-second TTL and a deliberately late 119-second configured renewal interval.
It advances 32 five-second heartbeats (160 simulated seconds) through the
64-binding renewal budget and asserts that every exact registration generation
remains eligible. The renewal pass supplies its bounded roster revisit horizon,
so registrations that would expire before their next turn renew early.

## History and multi-consumer evidence

Retained Task history is qualified independently at 0, 1,000 and 100,000 Tasks
per Project in the [active/history acceptance report](qexp_active_history_acceptance.md).
That matrix reports zero unrelated retained Task opens, so the Project-count
results above and retained-history results cover the pitch's two independent
scale dimensions.

Integration regressions exercise two exact consumers with distinct machine
runtime identities and machine names, with one lagging across
compaction, a new consumer reconstructing below the retention floor, repeated
bounded compaction, lost local progress, handoff races, registration-generation
replacement, and exact retirement recovery. A separate process publishes a
remote activation into the shared Project root and the dormant consumer wakes
from that durable checkpoint. These tests exercise the cross-machine protocol
boundary over the required shared filesystem; physical multi-host storage
durability remains a property of that filesystem rather than qexp's protocol.

The runnable lifecycle suite separately covers running work without recent
submissions, terminal collection, reservation release, dynamic registration,
lease expiry, on-demand exit, and restart recovery. Promotion preflight remains
the integration gate for the exact candidate commit.
