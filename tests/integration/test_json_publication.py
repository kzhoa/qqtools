import json

import pytest

from qqtools import save_json


@pytest.mark.parametrize("has_existing", [False, True])
def test_failed_serialization_preserves_destination(tmp_path, has_existing):
    target = tmp_path / "state.json"
    if has_existing:
        target.write_bytes(b'{"version": 1}')
    assert not save_json({"items": [1, object()]}, target, verbose=False)
    if has_existing:
        assert target.read_bytes() == b'{"version": 1}'
    assert set(tmp_path.iterdir()) == ({target} if has_existing else set())


def test_save_preserves_link_permissions_and_no_overwrite(tmp_path):
    target = tmp_path / "state.json"
    link = tmp_path / "current"
    link.symlink_to(target)
    assert save_json({"v": 1}, link, overwrite=False)
    target.chmod(0o640)
    assert not save_json({"v": 2}, link, overwrite=False)
    assert json.loads(target.read_bytes()) == {"v": 1}
    assert save_json({"v": 3}, link)
    assert link.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o640
    assert json.loads(target.read_bytes()) == {"v": 3}


def test_no_overwrite_preserves_file_created_during_serialization(tmp_path):
    target = tmp_path / "state.json"

    def publish_other_writer(_):
        target.write_bytes(b'{"writer": 2}')
        return "first"

    assert not save_json(object(), target, overwrite=False, default_serializer=publish_other_writer)
    assert target.read_bytes() == b'{"writer": 2}'
    assert list(tmp_path.iterdir()) == [target]
