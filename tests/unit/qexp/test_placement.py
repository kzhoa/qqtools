from datetime import datetime
from typing import get_type_hints

from qqtools.plugins.qexp.runtime.placement import is_machine_eligible


def test_machine_eligibility_annotation_resolves():
    assert get_type_hints(is_machine_eligible)["now"] == datetime | None
