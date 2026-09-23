"""Private webhook storage never publishes secret bytes with broad permissions."""

import json
import os
import stat

import pytest

from qqtools.plugins.qexp.notification_credentials import credential_path, read_webhook, stage_webhook

WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/example-token"


def test_staged_credentials_are_immutable_and_owner_private(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()

    first = stage_webhook(runtime_root, WEBHOOK)
    second = stage_webhook(runtime_root, WEBHOOK)

    assert first != second
    assert len(first) == len(second) == 32
    assert read_webhook(runtime_root, first) == WEBHOOK
    assert read_webhook(runtime_root, second) == WEBHOOK
    assert stat.S_IMODE(credential_path(runtime_root, first).parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(credential_path(runtime_root, first).stat().st_mode) == 0o600
    assert {path.name for path in credential_path(runtime_root, first).parent.iterdir()} == {
        f"{first}.json",
        f"{second}.json",
    }


@pytest.mark.parametrize(
    "webhook",
    [
        "",
        "https://example.invalid/open-apis/bot/v2/hook/token",
        "http://open.feishu.cn/open-apis/bot/v2/hook/token",
        "https://open.feishu.cn/open-apis/bot/v2/hook/",
        "https://open.feishu.cn/open-apis/bot/v2/hook/token?secret=value",
        "https://user:pass@open.feishu.cn/open-apis/bot/v2/hook/token",
    ],
)
def test_invalid_webhooks_do_not_stage_files(tmp_path, webhook):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()

    with pytest.raises(ValueError, match="webhook"):
        stage_webhook(runtime_root, webhook)

    assert not (runtime_root / "notifications").exists()


def test_reader_rejects_bad_identifier_and_corrupt_record_without_secret_in_error(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()
    credential_id = stage_webhook(runtime_root, WEBHOOK)

    with pytest.raises(ValueError):
        read_webhook(runtime_root, "../" + credential_id)
    credential_path(runtime_root, credential_id).write_text(json.dumps({"webhook": "secret-only-invalid"}))
    with pytest.raises((ValueError, OSError)) as error:
        read_webhook(runtime_root, credential_id)
    assert "secret-only-invalid" not in str(error.value)


def test_reader_refuses_exposed_or_replaced_credential(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()
    credential_id = stage_webhook(runtime_root, WEBHOOK)
    path = credential_path(runtime_root, credential_id)

    path.chmod(0o644)
    with pytest.raises((ValueError, OSError)):
        read_webhook(runtime_root, credential_id)

    path.unlink()
    other = tmp_path / "outside.json"
    other.write_text(json.dumps({"webhook": WEBHOOK}))
    path.symlink_to(other)
    with pytest.raises((ValueError, OSError)):
        read_webhook(runtime_root, credential_id)


def test_existing_unsafe_directory_is_not_used(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    directory = runtime_root / "notifications" / "credentials"
    directory.mkdir(parents=True)
    os.chmod(directory, 0o755)

    with pytest.raises((ValueError, OSError)):
        stage_webhook(runtime_root, WEBHOOK)
    assert list(directory.iterdir()) == []
