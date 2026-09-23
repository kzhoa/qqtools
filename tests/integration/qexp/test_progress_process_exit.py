"""Process-exit behavior for managed and direct progress producers."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_managed_process_exit_does_not_wait_for_public_flush(tmp_path):
    script = """
import os
import time
from qqtools.qexp import progress

os.environ['QEXP_PROGRESS_PATH'] = os.environ['TEST_PROGRESS_PATH']
def render(parts):
    return str(parts[0])
assert progress._offer_managed_progress(stage='train', current=1, message_parts=('ready',), render_message=render)
def waiting_close(self, *, timeout=0.1):
    time.sleep(4)
progress._Reporter.close = waiting_close
"""
    env = dict(os.environ, TEST_PROGRESS_PATH=str(tmp_path / "progress.json"))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=2)
    assert result.returncode == 0, result.stderr


def test_direct_progress_caller_retains_bounded_exit_flush(tmp_path):
    script = """
import os
from qqtools.qexp import progress

os.environ['QEXP_PROGRESS_PATH'] = os.environ['TEST_PROGRESS_PATH']
assert progress.update(stage='train', current=1)
def record_close(self, *, timeout=0.1):
    with open(os.environ['TEST_CLOSE_MARKER'], 'w') as marker:
        marker.write(str(timeout))
progress._Reporter.close = record_close
"""
    marker = tmp_path / "direct-close.txt"
    env = dict(os.environ, TEST_PROGRESS_PATH=str(tmp_path / "progress.json"), TEST_CLOSE_MARKER=str(marker))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=2)
    assert result.returncode == 0, result.stderr
    assert 0 <= float(marker.read_text()) <= 0.1
