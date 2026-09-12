from __future__ import annotations

import pytest

from qqtools.plugins.qexp.runtime.submission import _resolved_specs


def test_resolved_specs_rejects_string_command() -> None:
    with pytest.raises(ValueError, match="command must be a non-empty list of strings"):
        _resolved_specs([{"command": "echo hello"}], "gpu-1")
