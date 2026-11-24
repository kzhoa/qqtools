# Machine Idle Job

Idle Job runs a user command when a machine has no running qexp training and no
training currently eligible for admission. It belongs to the MachineRuntime,
across all registered Projects, and creates no experiment Task or Attempt.
External training processes are outside this feature's admission coordination.

A daemon agent can start without registered Projects, with or without Idle Job.
Projects registered later use the normal scheduling path. Idle Job is opt-in and
requires daemon mode; enabling it in on-demand mode reports a configuration
conflict. Agent readiness does not wait for the idle delay or command startup.

## Configure

Use an initialized MachineRuntime in daemon mode. The following command assumes
that `/opt/jobs/occupy.py`, its interpreter, and its working directory already
exist on the machine; qqtools does not provide the example workload.

```sh
qexp agent config idle-job set --enable \
  --command '/opt/jobs/.venv/bin/python /opt/jobs/occupy.py' \
  --working-directory /opt/jobs \
  --gpu-ids=-1 --idle-delay-seconds 60 --stop-timeout-seconds 10
qexp agent config idle-job show --format=json
qexp agent start
qexp agent status --format=json
```

Use the global `--machine-runtime-root PATH` option when selecting a nondefault
MachineRuntime. Configuration is revisioned under the `idle_job` key in
`<machine-runtime>/idle-job/config.json`.

The command is split into arguments with shell quoting and executed without a
shell. There is no shell expansion, pipeline, redirection, or implicit activation
of a virtual environment. Use an explicit interpreter path. An omitted working
directory defaults to the MachineRuntime root. The command inherits the agent's
environment, with GPU visibility supplied by Idle Job. A direct environment
prefix can use `env NAME=value executable`; shell-prefix assignments and `env -S`
are unsupported. Commands must remain in their inherited process group: do not
daemonize, call `setsid`, or detach children into another session.

Select GPUs only through `--gpu-ids`. The default `-1` selects the final ID in the
sorted set allowed by machine policy and successful hardware discovery. It never
means CUDA's hide-all-devices value. Explicit selections such as `--gpu-ids=1,3`
set `CUDA_VISIBLE_DEVICES=1,3`; the program normally sees logical devices 0 and 1.
The program must itself use those devices. Invalid IDs, empty allowed inventory,
and unavailable discovery block execution visibly.

Detectable `CUDA_VISIBLE_DEVICES` assignments inside the command are rejected.
Move those values to `--gpu-ids`. Static validation cannot detect changes hidden
inside scripts; commands must honor the injected environment. GPU visibility and
process groups are lifecycle mechanisms, not security or hardware isolation.

## Training handoff and shutdown

A continuous idle interval starts only after scheduling and recovery evidence
prove idleness. Locally ineligible queued work does not prevent execution;
unknown scheduling or ownership evidence requires a fresh idle interval when
resolved. Idle Job occupancy does not reduce training's eligibility.

Any eligible qexp training admission requests termination of the entire Idle Job,
including CPU training and training on other GPUs. Admission waits until owned
process cleanup is confirmed. The supervisor requests graceful termination,
then forces termination after the configured timeout. Uncertain cleanup keeps
training blocked; it does not silently release ownership.

```sh
qexp agent config idle-job disable
qexp agent stop
```

Disable requests cleanup through the running agent. Agent stop also reclaims its
Idle Job, including retained ownership after an agent crash. Its bounded wait
includes the configured Idle Job graceful timeout and cleanup allowance.
Changing the command or GPU selection replaces the old instance only
after cleanup. Removing a selected GPU from machine policy stops the current
instance and reevaluates selection. Linux parent-death guards reclaim the command
process group after abrupt agent or supervisor death. Restart reconciles durable
ownership before launching another instance.

## Status and troubleshooting

Idle Job status is separate from experiment progress and completion rates.

| State | Meaning |
| --- | --- |
| `disabled` | Disabled and owned processes cleaned up |
| `waiting` | Waiting for training, proof of idleness, or idle delay |
| `running` | Verified owned process active |
| `stopping` | Termination or cleanup in progress |
| `backoff` | Delayed retry after command exit or launch failure |
| `blocked` | Configuration, discovery, ownership, or cleanup prevents progress |

Status reports configured and resolved GPU IDs, process identity, start time,
last exit reason, retry time, and the independent log path. A ready agent does
not imply a running Idle Job. Stop failures must remain visible as stopping or
blocked, rather than disabled.

When `reason=unknown_demand`, `demand_detail` identifies the first check that
could not prove idleness. Upgrade discovery, upgrade blockers, scheduling
authority, recovery, and the training-demand census have distinct explanations.
With multiple blockers, the earlier check takes precedence; a more specific
supervisor outcome, such as an ownership failure, supersedes demand diagnostics.
`idle_delay` means training is confirmed idle and the configured start delay is
still active.

Ordinary `qexp agent status` shows the idle time observed at publication.
After an established interval restarts, the display keeps the latest reset's
check, elapsed seconds, and UTC time alongside the current reason. For example,
a training reservation acquired and released between ticks can leave the agent
waiting in `idle_delay` with `demand_revision_changed` as the last reset.
Reading status does not advance the displayed interval. A new supervisor starts
with a fresh interval and reset history.

Upgrade blockers name at most three Projects and report the total count. “Needs
repair” requires a matching cached migration observation; when only the blocker
is known, the explanation says the upgrade prevents confirmation of idle demand.
Missing migration fields are null. Inspection commands select the observed
MachineRuntime and locally registered Project. For example:

```sh
qexp --machine-runtime-root /srv/qexp/machine admin upgrade status --project lieformer
```

`qexp status` independently reports recorded repair for the selected Project,
including while its agent is stopped, and explains that migration's effect on new
training. A healthy selected Project can remain quiet while machine status names
a different blocking Project. See the [upgrade guide](spec/qexp_upgrade_guide.md)
for repair procedures; status does not perform repairs.

Demand observations carry the publishing agent's per-start identity and UTC
`observed_at`. They are live only when that identity matches the verified running
agent and their age is between zero and 60 seconds inclusive. Other valid details
are labeled last observed, without an elapsed duration. Heartbeats and status
reads do not refresh demand observations. `since` tracks continuous observations
of the same check and attributed scope within one agent instance; it resets on a
changed blocker, intervening known demand, or restart. If continuity cannot be
established cheaply, it is null. Duration additionally requires
`since <= observed_at`. Old or malformed optional diagnostics remain null without
making the base Idle Job state unreadable.

Each instance writes `command.log` under
`<machine-runtime>/idle-job/instances/<instance-id>/`. Output rotates at 1 MiB,
retaining one rotated file. Completed instance retention is bounded; unresolved
ownership is retained for recovery. Unexpected exits retry with exponential
backoff capped at 60 seconds. Normal exits also wait before restarting. Every
retry rechecks training demand; training-driven stops do not count as failures.

For a blocked instance, inspect its status reason and log first. Correct GPU
policy, command, or working-directory errors through configuration. Do not delete
ownership records or signal a process by an unverified numeric PID to bypass
cleanup. Runtime identity, boot identity, and process start identity protect
against stale records and PID reuse.
