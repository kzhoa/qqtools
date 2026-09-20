# Developer tooling maintenance

The public interface is `./scripts/dev test`, `./scripts/dev preflight`, and
`./scripts/dev env`. Install uv on PATH; no global tox, Python selection, shell
configuration changes, or environment activation are required. Initial setup
requires network access to Python distributions and package indexes. uv settings
that disable Python downloads must be adjusted when no compatible Python exists.
For restricted networks, configure `UV_DEFAULT_INDEX` for uv installations.
The tox-uv backend does not use pip configuration files. Tox forwards the UV
index, cache, link-mode, and certificate settings listed in `tox.ini` to both
test and packaging environments. No mirror is hardcoded in the repository.
CI's initial pip bootstrap still uses pip's own configuration.

## Platforms and prerequisites

The executable entry point uses `env -S` on Linux/macOS. Where shebang execution
is unavailable, use `uv run --script scripts/dev test` (and likewise for other
commands). Full preflight requires Linux and the system `tmux` executable, checked
before tox installs project dependencies. Python `libtmux` is installed through
the project extras; it does not install tmux. Other platforms may use `test` and
`env` subject to the dependencies and selected tests supporting their platform.

### Clock health for qexp lifecycle tests

LI-03's natural lease-expiry cases require a measured clock error bound of at
most 100 ms. qexp supports both chrony and Linux `adjtimex` observations. A large
kernel error bound does not necessarily mean the clock is actually that far off.
If kernel clock information is unavailable, unsynchronized, or exceeds the test
limit, install and configure chrony to obtain an independent measurement. On
Debian/Ubuntu, install the system package with `sudo apt-get install chrony`;
installing qqtools does not install this service. Chrony is unnecessary when the
kernel provider already satisfies the required bound.

In Docker or another container without `CAP_SYS_TIME`, configure the chrony
daemon to run with `-x`. This mode measures against reachable NTP sources without
adjusting the host's system clock. Configure the existing service or container
startup command rather than launching competing daemons. In containers without
an init system, arrange startup explicitly and retain the configuration in the
container image or provisioning setup so recreation does not lose it. See the
[chronyd options](https://chrony-project.org/doc/4.9/chronyd.html).

Check `chronyc tracking -n` and allow the initial measurements to converge before
running clock-sensitive tests. qexp includes system offset, half the root delay,
root dispersion, observation age, drift, and a safety margin in its bound;
`Leap status: Normal` alone does not establish qualification. Keep chrony running
between tests: startup convergence is an environment preparation cost, not a
per-test requirement.

Installation alone cannot guarantee a qualifying measurement. If NTP is
unreachable or the measured bound remains above 100 ms, resolve the time-source
or host-clock issue; do not widen LI-03's threshold or skip its cases. After
preparing the environment, verify with:

```bash
./scripts/dev test tests/integration/qexp/test_agent_lifecycle_independence.py::test_li03_real_peer_observes_natural_lease_expiry -q
```

## Environment ownership

uv reads PEP 723 metadata in `scripts/dev`, obtains or reuses Python 3.13, and
caches an isolated script environment containing pinned tox, tox-uv, and uv. Compatible
system Python installations are allowed. The Python range does not automatically
upgrade an existing interpreter to the newest patch release; maintainers update
uv/Python periodically. No `.python-version` or global default Python is needed.

The script calls tox through `sys.executable` from the repository root. `unit`
and `preflight` inherit that Python version and use separate cached `.tox`
environments. Tox owns their lifecycle and commands; tox-uv delegates creation
and dependency installation to uv. The backend's uv version is pinned, while the
initial uv launcher is provided by the developer. Both environments install the full project extras so optional-feature unit tests
have their dependencies. A cold run can download large PyTorch packages; “quick”
describes the unit-test scope, not first-time dependency installation.

`test` forwards pytest arguments unchanged (an optional leading `--` is removed).
Its pytest default `testpaths` is `tests/unit`, even with options such as `-q` or
`-k retry`. Explicit paths/node IDs override that default using pytest's own
parser. Relative test paths are relative to the repository root. Maintainers may
deliberately use pytest options to override this daily-test configuration.

`env` creates `.venv` with the entry point's interpreter, or reuses an existing
Python 3.13 environment. It installs editable `.[full]` and pytest-xdist explicitly
into that environment, regardless of any activated environment. Rerun after
dependency changes; installation is additive, not an exact dependency sync.
An incompatible or broken `.venv` produces a rebuild instruction instead of
silently deleting it. Select `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`)
in the IDE. IDE dependencies do not control either tox gate.

## Shared validation and versions

Local `preflight` and CI `preflight-ci` run `scripts/ci/run_preflight.py`, the sole
gate manifest. The runner rejects non-3.13 Python and missing Linux/tmux before
running checks. The local entry point also checks prerequisites before tox setup.
No filtering or skip arguments are accepted by `preflight`.

CI continues to obtain Python 3.13 from setup-python, install CPU-only PyTorch
and `.[ci-preflight]`, then invoke `python -m tox run -e preflight-ci`. That tox
environment exposes the prepared interpreter's site-packages. Local full extras
may install GPU-capable PyTorch. A shared gate manifest guarantees the same
commands, not identical dependencies or hardware.

Keep the exact tox, tox-uv, and uv versions synchronized in inline script metadata,
`[tox] requires`, and the `dev`/`ci-preflight` extras. A regression test checks
consistency. Tox can provision these tools when invoked from a bare installation,
so existing artifact CI commands also select the uv backend without workflow edits.
The preflight CI bootstrap installs the pinned tools directly in the setup-python
interpreter; preserve this so its system-site-packages environment sees CPU PyTorch.
Python minor
changes require updating the script range, runner guard, IDE validation, and CI
configuration together; workflow changes require explicit owner approval.
Other version-specific tox lanes retain their own interpreter requirements.
Project dependency versions are not fully locked. Reusing caches does not
promise offline operation or byte-for-byte reproducibility.

Ruff explicitly includes the extensionless `scripts/dev` entry point in the
shared lint and formatting gates.

## Installing environments without running gates

For initial provisioning only, maintainers can run:

```bash
./scripts/dev env
uv run --no-project --python 3.13 --with 'tox==4.61.5' \
  --with 'tox-uv==1.36.0' --with 'uv==0.12.16' \
  python -m tox run -e unit,preflight --notest
```

This does not validate the repository; run the standard test/preflight commands
after installation. Switching from virtualenv/pip may recreate old tox environments
once. Separate environments remain isolated, but reuse uv's download/install cache.
For a cache on a different filesystem from the environments, `UV_LINK_MODE=copy`
explicitly accepts copying. For link-based reuse, place a personal `UV_CACHE_DIR`
on the same filesystem as the environments, provided it supports linking. Neither
choice removes shared-filesystem I/O costs, and neither belongs in committed machine-specific paths.

## Selecting tox gates

Bare `tox` runs only `preflight`, which already includes Unit and the required
Integration selection. Run `unit` or a scoped qexp lane while developing; run
`artifact-e2e` separately when installed delivery behavior or release policy
requires it. The redundant `base` lane and automatic `cleanup` lane are removed;
validation preserves `tmp/` diagnostics instead of deleting them afterward.
The unused pytest-cov dependency and obsolete pip-cache override are removed;
tox-uv uses the uv cache settings described above.
The CI release profile uses the same shared Python runner as ordinary preflight
and replaces representative qexp coverage with the complete qexp integration gate.
`release-e2e` inherits the installed E2E environment and still accepts
`--installpkg` for the exact wheel selected by publishing.

For local Python compatibility checks, run:

```bash
tox run-parallel -e 'py{311,312,313,314}-artifact-smoke'
```

These lanes install wheels, verify that imports come from site-packages, and
exercise `qexp --help`. They do not run four copies of the source Unit suite or
install pytest/coverage tools. Missing interpreters fail instead of silently
skipping a claimed compatibility check; install the requested Python with uv
beforehand. Full source validation stays on Python 3.13. CI keeps Python 3.13
installed E2E plus smoke checks on 3.11, 3.12, and 3.14. This is installation and
CLI-startup compatibility evidence, not full functional coverage on every minor.

The unused `check_qexp_shared_filesystem.py` probe has been removed following
[ADR-QEXP-0003](../adr/qexp/0003-remove-filesystem-qualification-gate.md).
Store regression tests retain exclusive-create conflict and JSON readback
coverage; a local sequential probe does not certify cross-host filesystem semantics.

## Action runtime maintenance

Workflows use GitHub-hosted Ubuntu runners. JavaScript actions use Node 24:
checkout v7, setup-python v7, upload-artifact v7, github-script v9, and
softprops/action-gh-release v3. Node 24 actions require runner 2.327.1 or later;
any future self-hosted runner setup must account for each action's requirements.
The PyPI publisher follows its maintained `release/v1` channel and uses a
composite/container implementation. Action runtime versions are independent of
the Python version selected for package tests.

Review [checkout release notes](https://github.com/actions/checkout/releases),
[setup-python release notes](https://github.com/actions/setup-python/releases),
and the other actions' metadata when upgrading majors,
including checkout credential persistence and github-script API changes.
Preserve the owner credential boundary and the read-only PR inspection path.
