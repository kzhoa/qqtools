"""Internal runtime fingerprints for normalized qexp projection inputs.

This extraction is a deterministic comparison aid, not an authenticating hash
or a claim that a parser or a whole projection can resume from this state.
Equal fingerprints remain ambiguous between a duplicate and a hash collision;
an unequal pair proves that the normalized inputs differ under this scheme.
It does not certify Group membership or publish runtime authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Final

BLOCK: Final = 65_536

_INIT_DOMAIN = b"qexp-projection-digest-v1:init"
_BLOCK_DOMAIN = b"qexp-projection-digest-v1:block\0"
_FINAL_DOMAIN = b"qexp-projection-digest-v1:final\0"
_INIT_CHAIN = hashlib.sha256(_INIT_DOMAIN).digest()
_SNAPSHOT_KEYS = frozenset({"version", "total_bytes_hex", "chain_hex", "pending_b64"})
_MAX_PENDING_B64 = ((BLOCK - 1 + 2) // 3) * 4


class ChainedDigest:
    """Build a checkpointable, non-authenticating scalar fingerprint.

    The input is treated as already normalized by the caller.  Full 65536-byte
    blocks are chained, while the final incomplete block is retained until
    the next update or finalization.  Snapshot state is intentionally limited
    to the chain digest, byte count, and pending remainder; it is not a parser
    or whole-projection resume protocol.
    """

    def __init__(self) -> None:
        self._total_bytes = 0
        self._chain = _INIT_CHAIN
        self._pending = bytearray()

    @property
    def size(self) -> int:
        """Return the total number of bytes accepted by :meth:`update`."""

        return self._total_bytes

    def update(self, data: bytes) -> None:
        """Consume bytes without retaining more than one incomplete block."""

        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        data_length = len(data)
        self._total_bytes += data_length
        offset = 0

        if self._pending:
            needed = BLOCK - len(self._pending)
            if data_length < needed:
                self._pending.extend(data)
                return
            self._pending.extend(data[:needed])
            self._consume_block(self._pending)
            self._pending.clear()
            offset = needed

        while data_length - offset >= BLOCK:
            self._consume_block(data[offset : offset + BLOCK])
            offset += BLOCK

        if offset < data_length:
            self._pending.extend(data[offset:])

    def digest(self) -> bytes:
        """Return the final fingerprint without changing the chaining state."""

        digest = hashlib.sha256(_FINAL_DOMAIN)
        digest.update(self._chain)
        digest.update(_decimal_bytes(self._total_bytes))
        digest.update(b"\0")
        digest.update(self._pending)
        return digest.digest()

    def hexdigest(self) -> str:
        """Return :meth:`digest` as lowercase hexadecimal text."""

        return self.digest().hex()

    def snapshot(self) -> dict[str, object]:
        """Return JSON-compatible state without decimal integer size limits."""

        return {
            "version": 1,
            "total_bytes_hex": format(self._total_bytes, "x"),
            "chain_hex": self._chain.hex(),
            "pending_b64": base64.b64encode(self._pending).decode("ascii"),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> ChainedDigest:
        """Restore a validated checkpoint without retaining caller data."""

        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")
        keys = frozenset(snapshot)
        if keys != _SNAPSHOT_KEYS:
            missing = sorted(_SNAPSHOT_KEYS - keys)
            unknown = sorted(keys - _SNAPSHOT_KEYS, key=repr)
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unknown:
                details.append(f"unknown keys: {unknown}")
            raise ValueError("snapshot keys are invalid (" + "; ".join(details) + ")")

        version = snapshot["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("snapshot version must be a non-boolean integer")
        if version != 1:
            raise ValueError("snapshot version must be 1")

        total_hex = snapshot["total_bytes_hex"]
        if not isinstance(total_hex, str):
            raise TypeError("snapshot total_bytes_hex must be a string")
        if (
            not total_hex
            or any(character not in "0123456789abcdef" for character in total_hex)
            or (len(total_hex) > 1 and total_hex[0] == "0")
        ):
            raise ValueError("snapshot total_bytes_hex must be canonical nonnegative hexadecimal")
        total_bytes = int(total_hex, 16)

        chain_hex = snapshot["chain_hex"]
        if not isinstance(chain_hex, str):
            raise TypeError("snapshot chain_hex must be a string")
        if len(chain_hex) != 64 or any(character not in "0123456789abcdef" for character in chain_hex):
            raise ValueError("snapshot chain_hex must be exactly 64 lowercase hexadecimal characters")
        chain = bytes.fromhex(chain_hex)

        pending_b64 = snapshot["pending_b64"]
        if not isinstance(pending_b64, str):
            raise TypeError("snapshot pending_b64 must be a string")
        if len(pending_b64) > _MAX_PENDING_B64:
            raise ValueError("snapshot pending_b64 exceeds the maximum remainder size")
        try:
            encoded = pending_b64.encode("ascii")
            pending = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise ValueError("snapshot pending_b64 must be strict canonical base64") from exc
        if base64.b64encode(pending).decode("ascii") != pending_b64:
            raise ValueError("snapshot pending_b64 must use canonical base64 encoding")
        if len(pending) >= BLOCK:
            raise ValueError("snapshot pending remainder must be shorter than one block")
        if len(pending) != total_bytes % BLOCK:
            raise ValueError("snapshot pending remainder length does not match total_bytes")
        if total_bytes < BLOCK and chain != _INIT_CHAIN:
            raise ValueError("snapshot chain_hex must be the initial chain before the first full block")

        restored = cls()
        restored._total_bytes = total_bytes
        restored._chain = bytes(chain)
        restored._pending = bytearray(pending)
        return restored

    def _consume_block(self, block: bytes | bytearray) -> None:
        digest = hashlib.sha256(_BLOCK_DOMAIN)
        digest.update(self._chain)
        digest.update(block)
        self._chain = digest.digest()


def _decimal_bytes(value: int) -> bytes:
    """Encode the final-domain count without changing Python's digit guard."""

    if value == 0:
        return b"0"
    chunks: list[int] = []
    while value:
        value, remainder = divmod(value, 1_000_000_000)
        chunks.append(remainder)
    return str(chunks[-1]).encode("ascii") + b"".join(f"{chunk:09d}".encode("ascii") for chunk in reversed(chunks[:-1]))
