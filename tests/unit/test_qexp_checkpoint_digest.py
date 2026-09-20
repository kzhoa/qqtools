"""Pure unit coverage for the checkpointable qexp projection fingerprint."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tracemalloc

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.fingerprint import BLOCK, ChainedDigest

_INIT = hashlib.sha256(b"qexp-projection-digest-v1:init").digest()
_BLOCK_PREFIX = b"qexp-projection-digest-v1:block\0"
_FINAL_PREFIX = b"qexp-projection-digest-v1:final\0"


def _golden(data: bytes) -> str:
    chain = _INIT
    offset = 0
    while len(data) - offset >= BLOCK:
        chain = hashlib.sha256(_BLOCK_PREFIX + chain + data[offset : offset + BLOCK]).digest()
        offset += BLOCK
    remainder = data[offset:]
    return hashlib.sha256(_FINAL_PREFIX + chain + str(len(data)).encode("ascii") + b"\0" + remainder).hexdigest()


def _payload(length: int) -> bytes:
    pattern = bytes(range(251))
    return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]


def _feed(data: bytes, chunk_size: int) -> ChainedDigest:
    digest = ChainedDigest()
    for offset in range(0, len(data), chunk_size):
        digest.update(data[offset : offset + chunk_size])
    return digest


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", "cff633746cec53bc43d81cc63da876aff0ea1c232392a9681159152b87aee859"),
        (b"abc", "27e8b2b3c0aac0a78a663f001a5292a71a6e27f622924615b28c1f65340c5664"),
        (b"a" * (BLOCK - 1), "e09b9306cb1a525d6b426e51e4db4707a2c3a0225063167aca0bf77ce87b1812"),
        (b"a" * BLOCK, "bb681dd2aa5c6a94f14256571750c1927af9509c3c13e87180a879f592957586"),
        (b"a" * (BLOCK + 1), "b021a7b6f25a6667303ae3a806f45d216c385e449e2eb1ef345c0b98f218b021"),
        (b"a" * (2 * BLOCK + 7), "d11988e024440f518727ce9f6712ff9714c41126b731967d87653b88f5acf8fa"),
    ],
)
def test_fixed_golden_vectors(data: bytes, expected: str):
    digest = ChainedDigest()
    digest.update(data)

    assert digest.size == len(data)
    assert digest.hexdigest() == expected
    assert digest.digest() == bytes.fromhex(expected)


@pytest.mark.parametrize("length", [0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 7])
@pytest.mark.parametrize("chunk_size", [1, 7, BLOCK - 1, BLOCK, BLOCK + 1])
def test_feed_segmentation_is_canonical(length: int, chunk_size: int):
    data = _payload(length)
    digest = _feed(data, chunk_size)

    assert digest.size == length
    assert digest.hexdigest() == _golden(data)
    assert len(digest._pending) < BLOCK


@pytest.mark.parametrize("offset", [0, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 7])
def test_json_roundtrip_snapshot_continues_to_same_digest(offset: int):
    data = _payload(2 * BLOCK + 7)
    uninterrupted = ChainedDigest()
    uninterrupted.update(data)

    partial = ChainedDigest()
    partial.update(data[:offset])
    snapshot = json.loads(json.dumps(partial.snapshot()))
    assert set(snapshot) == {"version", "total_bytes_hex", "chain_hex", "pending_b64"}
    restored = ChainedDigest.from_snapshot(snapshot)
    partial.update(data[offset:])
    restored.update(data[offset:])

    assert restored.size == len(data)
    assert restored.digest() == uninterrupted.digest() == bytes.fromhex(_golden(data))
    assert partial.snapshot() == uninterrupted.snapshot()


def test_finalization_is_nondestructive_and_updates_remain_allowed():
    digest = ChainedDigest()
    digest.update(b"abc")
    snapshot = digest.snapshot()
    first = digest.digest()

    assert digest.digest() == first
    assert digest.hexdigest() == first.hex()
    assert digest.snapshot() == snapshot

    digest.update(b"def")
    assert digest.size == 6
    assert digest.digest() == bytes.fromhex(_golden(b"abcdef"))


def test_snapshot_is_exact_and_defensive():
    digest = ChainedDigest()
    digest.update(b"checkpoint")
    snapshot = digest.snapshot()

    assert snapshot.keys() == {"version", "total_bytes_hex", "chain_hex", "pending_b64"}
    restored = ChainedDigest.from_snapshot(snapshot)
    snapshot["pending_b64"] = ""
    snapshot["total_bytes_hex"] = "0"
    assert restored.size == len(b"checkpoint")
    assert restored.digest() == digest.digest()


@pytest.mark.parametrize(
    "bad_update",
    [None, "text", bytearray(b"bytes"), memoryview(b"bytes"), 1],
)
def test_update_rejects_non_bytes_and_accepts_empty_bytes(bad_update):
    digest = ChainedDigest()
    with pytest.raises(TypeError, match="data must be bytes"):
        digest.update(bad_update)
    digest.update(b"")
    assert digest.size == 0


def test_size_is_read_only():
    digest = ChainedDigest()
    with pytest.raises(AttributeError):
        digest.size = 1


@pytest.mark.parametrize("missing", ["version", "total_bytes_hex", "chain_hex", "pending_b64"])
def test_snapshot_rejects_missing_keys(missing: str):
    snapshot = ChainedDigest().snapshot()
    del snapshot[missing]
    with pytest.raises(ValueError, match="keys are invalid"):
        ChainedDigest.from_snapshot(snapshot)


def test_snapshot_rejects_unknown_keys_and_non_dicts():
    snapshot = ChainedDigest().snapshot()
    snapshot["extra"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        ChainedDigest.from_snapshot(snapshot)
    with pytest.raises(TypeError, match="snapshot must be a dict"):
        ChainedDigest.from_snapshot([])


@pytest.mark.parametrize("version", [True, False, "1", 2])
def test_snapshot_rejects_bad_versions(version):
    snapshot = ChainedDigest().snapshot()
    snapshot["version"] = version
    expected = TypeError if isinstance(version, (bool, str)) else ValueError
    with pytest.raises(expected):
        ChainedDigest.from_snapshot(snapshot)


@pytest.mark.parametrize("total_bytes_hex", [True, False, 0, -1, 1.5, "", "00", "0x1", "-1", "F", "g"])
def test_snapshot_rejects_bad_total_bytes(total_bytes_hex):
    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = total_bytes_hex
    expected = ValueError if isinstance(total_bytes_hex, str) else TypeError
    with pytest.raises(expected):
        ChainedDigest.from_snapshot(snapshot)


@pytest.mark.parametrize("chain_hex", ["0" * 63, "0" * 65, "A" * 64, "g" * 64, b"0" * 64])
def test_snapshot_rejects_noncanonical_chain_hex(chain_hex):
    snapshot = ChainedDigest().snapshot()
    snapshot["chain_hex"] = chain_hex
    expected = TypeError if isinstance(chain_hex, bytes) else ValueError
    with pytest.raises(expected):
        ChainedDigest.from_snapshot(snapshot)


def test_snapshot_rejects_bad_and_noncanonical_base64():
    snapshot = ChainedDigest().snapshot()
    snapshot["pending_b64"] = "!"
    with pytest.raises(ValueError, match="base64"):
        ChainedDigest.from_snapshot(snapshot)

    snapshot["total_bytes_hex"] = "1"
    snapshot["pending_b64"] = "YR=="  # Decodes to b"a", but has nonzero discarded pad bits.
    with pytest.raises(ValueError, match="canonical base64"):
        ChainedDigest.from_snapshot(snapshot)


def test_snapshot_rejects_full_or_mismatched_pending_remainders():
    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = format(BLOCK, "x")
    snapshot["pending_b64"] = base64.b64encode(b"x" * BLOCK).decode("ascii")
    with pytest.raises(ValueError, match="remainder"):
        ChainedDigest.from_snapshot(snapshot)

    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = "1"
    snapshot["pending_b64"] = ""
    with pytest.raises(ValueError, match="length"):
        ChainedDigest.from_snapshot(snapshot)


def test_snapshot_rejects_a_non_initial_chain_before_first_block():
    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = "1"
    snapshot["pending_b64"] = base64.b64encode(b"x").decode("ascii")
    snapshot["chain_hex"] = "00" * 32
    with pytest.raises(ValueError, match="initial chain"):
        ChainedDigest.from_snapshot(snapshot)


def test_initial_chain_is_representable_after_a_full_block():
    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = format(BLOCK, "x")

    restored = ChainedDigest.from_snapshot(snapshot)

    assert restored.size == BLOCK
    assert restored.snapshot()["chain_hex"] == _INIT.hex()


def test_large_counter_roundtrips_and_finalizes_without_changing_digit_guard():
    original_limit = sys.get_int_max_str_digits()
    total = BLOCK * (10**5000)
    snapshot = ChainedDigest().snapshot()
    snapshot["total_bytes_hex"] = format(total, "x")
    restored = ChainedDigest.from_snapshot(json.loads(json.dumps(snapshot)))

    expected_count = b"65536" + b"0" * 5000
    expected = hashlib.sha256(_FINAL_PREFIX + _INIT + expected_count + b"\0").digest()
    assert restored.size == total
    assert restored.digest() == expected
    assert json.loads(json.dumps(restored.snapshot())) == snapshot
    assert sys.get_int_max_str_digits() == original_limit


def test_large_update_has_bounded_working_memory_and_restores():
    data = (bytes(range(251)) * ((8 * 1024 * 1024 + 250) // 251))[: 8 * 1024 * 1024]
    digest = ChainedDigest()
    tracemalloc.start()
    try:
        digest.update(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1 * 1024 * 1024
    assert digest.size == len(data)
    assert len(digest._pending) < BLOCK
    restored = ChainedDigest.from_snapshot(json.loads(json.dumps(digest.snapshot())))
    digest.update(b"tail")
    restored.update(b"tail")
    assert restored.digest() == digest.digest() == bytes.fromhex(_golden(data + b"tail"))
