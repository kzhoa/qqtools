"""Qualification child entrypoints retain isolated source provenance."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", ["live_namespace_upgrade", "paused_namespace_upgrade"])
def test_released_probe_child_imports_rollout_from_script_directory(tmp_path, checkout_subprocess_env, case):
    repo = Path(__file__).resolve().parents[3]
    environment = dict(checkout_subprocess_env, PYTHONPATH=str(repo / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path[0] = sys.argv[1]\n"
            "import qexp_namespace_rollout as rollout\n"
            "rollout.probe_live_namespace_upgrade = lambda *a, **k: {'paused': k['should_pause']}\n"
            "import probe_qexp_writer_fences as probe\n"
            "result = probe.probe(sys.argv[2], Path.cwd(), Path(sys.argv[3]))\n"
            "assert result == {'paused': sys.argv[2] == 'paused_namespace_upgrade'}\n",
            str(repo / "scripts/qualification"),
            case,
            str(repo / "src"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
