import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
for module_name in list(sys.modules.keys()):
    if module_name == "qqtools" or module_name.startswith("qqtools."):
        sys.modules.pop(module_name)

from qqtools.plugins.qpipeline.runner.runner_utils.progress import resolve_render_mode


@pytest.mark.parametrize(
    "requested_mode,has_rich,has_tqdm,expected,expected_msg",
    [
        (None, True, True, "rich", "Mode auto -> rich"),
        ("auto", True, True, "rich", "Mode auto -> rich"),
        ("auto", False, True, "tqdm", "Mode auto -> tqdm"),
        ("auto", False, False, "plain", "Mode auto -> plain"),
        ("rich", False, True, "tqdm", "Mode rich -> tqdm"),
        ("tqdm", False, False, "plain", "Mode tqdm -> plain"),
        ("rich", True, True, "rich", None),
    ],
)
def test_resolve_render_mode(requested_mode, has_rich, has_tqdm, expected, expected_msg):
    resolved_mode, message = resolve_render_mode(requested_mode, has_rich, has_tqdm)
    assert resolved_mode == expected
    assert message == expected_msg
