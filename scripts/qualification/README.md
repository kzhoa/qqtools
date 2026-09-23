# qexp qualification tools

These maintained tools run against isolated, newly created roots. Production
never imports them. Use Python 3.13 from the repository root; preserve real fsync.
They qualify same-host filesystem/process behavior, not cross-host durability.

## Active responsibility cost and churn

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python -m scripts.qualification.profile_local_responsibility \
  --output /tmp/responsibility-profile --cycles 10000 --samples 3 --live 4 64 65 256
```

The output directory must not exist. `results.json` retains source hashes,
transition reads/writes/fsyncs, lock time, traversal and retained-file counts.
The production-ledger fault adapter is in `tests/helpers/qexp/local_responsibility.py`;
its regression tests are in `tests/integration/test_local_responsibility_storage.py`.
Initialization and fixture population are outside transition timings. Counters
exclude metadata probes and are not physical device I/O. Warm uncontrolled caches
and small samples do not establish cold-start or tail-latency guarantees.

## Retained history and Task pagination

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python -m pytest \
  -p tests.helpers.qexp.authority_measurement \
  -p tests.helpers.qexp.history_qualification \
  tests/integration/qexp/test_authority_workload.py \
  --settled-history-count=100000 --settled-history-submissions=bulk \
  --authority-workload-output=/tmp/history-100000-bulk.json
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python -m pytest \
  tests/integration/qexp/test_observation_scale.py --run-stress -m stress -q
```

Routine pagination covers 0 and 129 retained Tasks; manual stress qualification
covers 1,000 and 10,000.
Run machine counts 0, 1000, 10000 and 100000, with both bulk and single Submission
shapes at nonzero counts. The fixture prepares quiescent authoritative records and
complete observation indexes before measurement. It is not a benchmark of creating
100000 Tasks through public lifecycle calls. The real measured workload retains
runners, persistence and its original 15-second deadlines. Reports require primary
discovery and zero opens of unrelated retained Task records across agent threads;
runner I/O is excluded. Pagination asserts complete truth traversal and bounded
index work for each page. Run large qualifications sequentially to avoid workload
interference.

## Group service quiescence

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python \
  -m scripts.qualification.profile_group_service_quiescence \
  --output /tmp/qexp-group-service-quiescence
```

The output directory must not exist. The default qualification measures locator
traversal with 0, 1,000, 10,000, and 100,000 retained Groups, performs 100,000
accelerated locator lifecycles, and runs a 24-hour real-process soak. It retains
source hashes, filesystem/cache conditions, descriptor, thread, RSS, traversal,
churn, and soak samples in `results.json`. Run the large profile sequentially and
retain its output as release evidence. Smaller `--histories`, `--cycles`, and
`--soak-seconds` values are development smoke checks rather than Phase C evidence.

## Released writer gate qualification

`probe_qexp_writer_fences.py` characterizes actual source from local release refs,
without installing it or accessing existing project runtimes:

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python -m scripts.qualification.probe_qexp_writer_fences \
  --output /tmp/qexp-writer-fences --ref v1.3.18 --ref v1.3.17
```

The output directory must not exist. Each release is extracted from its resolved
Git commit and every case runs in a fresh subprocess with isolated HOME, XDG,
temporary and machine runtime paths. Module provenance is checked. The dispatch
executor records launch requests without spawning a process; rejected runner
cases forbid process creation. The three continuity cases run a real released
wrapper/guardian with a lightweight, test-owned Python child and bounded waits. Real persistence and locks remain enabled. Results retain
the release SHA, probe hash, imported source path, errors and Task mutation outcome;
each case also retains stdout/stderr and its isolated fixture.

The `live_namespace_upgrade` and `paused_namespace_upgrade` cases run actual
released machine agents and a released runner, then restart with current source.
They require a short output path for tmux sockets, for example:

```bash
PYTHONPATH=src ~/.cache/qqtools/tox/unit/bin/python -m scripts.qualification.probe_qexp_writer_fences \
  --output /tmp/qgn-rollout --ref v1.3.17 --ref v1.3.18 \
  --case live_namespace_upgrade --case paused_namespace_upgrade
```

They assert automatic Group isolation while the original workload is alive,
unchanged process/Attempt identity, one launch, exit evidence while its agent is
offline, and terminal reconciliation after restart. The paused case retains a
second released participant across multiple target enrollment passes before
upgrading it. Each logical participant uses its own temporary namespace for the
host-global scheduler lock. This qualifies same-host multi-process ordering, not
cross-host filesystem semantics or installed-wheel delivery. No production
machine runtime is used. Cleanup releases the test workload, stops owned agents,
and removes only tmux servers under the probe's isolated socket directory.
