import hashlib
from unittest.mock import MagicMock

import pytest

from qqtools.config.fetch import gdown

URL = "https://drive.google.com/file/d/example/view"
MD5 = hashlib.md5(b"new contents").hexdigest()


def response(chunks=(), *, token=None):
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.cookies = {"download_warning_fixture": token} if token else {}
    reply.iter_content.return_value = iter(chunks)
    return reply


@pytest.fixture
def session(monkeypatch):
    requests = MagicMock()
    monkeypatch.setattr(gdown, "requests", requests)
    return requests.Session.return_value.__enter__.return_value


@pytest.mark.parametrize("confirm", [False, True])
def test_verified_download_and_cache_hit(tmp_path, session, confirm):
    target = tmp_path / "download"
    target.write_bytes(b"stale cache")
    final = response([b"new ", b"contents"])
    replies = [response(token="ticket"), final] if confirm else [final]
    session.get.side_effect = replies
    gdown.download_from_gdrive_sharelink(URL, target, MD5)
    assert target.read_bytes() == b"new contents"
    if confirm:
        assert session.get.call_args.kwargs["params"]["confirm"] == "ticket"
    gdown.download_from_gdrive_sharelink(URL, target, MD5.upper())
    assert session.get.call_count == len(replies)
    for reply in replies:
        reply.__exit__.assert_called_once()


@pytest.mark.parametrize("fault", ["http", "confirmed_http", "checksum", "interrupted"])
@pytest.mark.parametrize("has_existing", [True, False])
def test_failed_download_preserves_destination(tmp_path, session, fault, has_existing):
    target = tmp_path / "download"
    if has_existing:
        target.write_bytes(b"previous")
    final = response([b"wrong contents"])
    if fault in ("http", "confirmed_http"):
        final.raise_for_status.side_effect = OSError("HTTP request failed")
    if fault == "interrupted":

        def interrupted(_):
            yield b"partial"
            raise OSError("connection closed")

        final.iter_content.side_effect = interrupted
    session.get.side_effect = [response(token="ticket"), final] if fault == "confirmed_http" else [final]
    with pytest.raises(ValueError if fault == "checksum" else OSError):
        gdown.download_from_gdrive_sharelink(URL, target, MD5 if fault == "checksum" else None)
    assert set(tmp_path.iterdir()) == ({target} if has_existing else set())
    if has_existing:
        assert target.read_bytes() == b"previous"
