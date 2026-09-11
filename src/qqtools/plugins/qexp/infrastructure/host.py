"""Host identity accessors."""

from pathlib import Path


def host_instance_id() -> str:
    """Return a host-local token that is not copied with the machine runtime."""
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    raise RuntimeError("qexp cannot verify host identity; machine-id is unavailable.")
