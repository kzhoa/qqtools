from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.store import CASConflict, cas_update, create_if_absent, read_json


def test_create_if_absent_and_cas(tmp_path: Path):
    path = tmp_path / "record.json"
    create_if_absent(path, {"meta": {"revision": 1}, "value": 1})
    with pytest.raises(CASConflict):
        create_if_absent(path, {"meta": {"revision": 1}, "value": 2})
    value = read_json(path)
    value["value"] = 2
    cas_update(path, 1, value)
    assert read_json(path)["meta"]["revision"] == 2
