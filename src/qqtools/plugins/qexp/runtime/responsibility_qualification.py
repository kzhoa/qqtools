"""Fixed-metadata qualification for machine-owned primary responsibility discovery."""

from pathlib import Path

from ..layout import LOCAL_RECOVERY_CAPABILITY
from .paths import shared_paths
from .responsibility_capture import capture_admission
from .responsibility_completion import read_capture_completion
from .responsibility_store import DurableIO, Unavailable
from .store import read_json_limited


def qualify_discovery(runtime_root: Path, owner: dict, *, previous_generation: str | None = None) -> str | None:
    """Return the exact capture generation, or None while legacy capture is building.

    The caller still checks current registration authority before every mutation.
    This proof concerns discovery coverage and never authorizes a launch itself.
    """
    proof = read_discovery_completion(runtime_root, owner)
    if proof is None:
        if previous_generation is not None:
            raise Unavailable("activated discovery completion disappeared")
        return None
    if proof["capture_digest"] != previous_generation:
        DurableIO().sync_directory(runtime_root, "capture_completion")
    return proof["capture_digest"]


def read_discovery_completion(runtime_root: Path, owner: dict) -> dict | None:
    """Read qualified coverage without synchronizing or granting authority."""
    proof = read_capture_completion(runtime_root, should_sync=False)
    if proof is None:
        return None
    expected = capture_admission(owner)
    if any(proof.get(key) != value for key, value in expected.items() if key != "format"):
        raise Unavailable("discovery completion belongs to another binding")
    schema = read_json_limited(
        shared_paths(Path(owner["shared_root"]))["schema"] / "version.json", max_bytes=16384
    ).get("schema")
    capabilities = schema.get("required_capabilities") if isinstance(schema, dict) else None
    if not isinstance(capabilities, list) or LOCAL_RECOVERY_CAPABILITY not in capabilities:
        raise Unavailable("primary discovery admission capability is unavailable")
    return proof
