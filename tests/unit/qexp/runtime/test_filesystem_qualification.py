import pytest

from qqtools.plugins.qexp.runtime.filesystem_qualification import (
    FilesystemProbeEvidence,
    evaluate_filesystem_probe,
)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"peer_host": "host-a"}, "probe hosts must be distinct"),
        ({"initiator_host": "   "}, "both host identities are required"),
        ({"exclusive_lock": False}, "cross-host exclusive lock failed"),
        ({"exclusive_lock": 1}, "cross-host exclusive lock failed"),
        ({"atomic_replace": False}, "atomic replace visibility failed"),
        ({"fsync_visibility": False}, "fsync durability visibility failed"),
        ({"failure_cleanup": False}, "failure cleanup behavior failed"),
    ],
)
def test_filesystem_probe_fails_closed(changes: dict[str, object], reason: str) -> None:
    values = {
        "initiator_host": "host-a",
        "peer_host": "host-b",
        "exclusive_lock": True,
        "atomic_replace": True,
        "fsync_visibility": True,
        "failure_cleanup": True,
    }
    values.update(changes)

    result = evaluate_filesystem_probe(FilesystemProbeEvidence(**values))

    assert result.is_qualified is False
    assert reason in result.reasons


def test_filesystem_probe_accepts_complete_two_host_evidence() -> None:
    evidence = FilesystemProbeEvidence("host-a", "host-b", True, True, True, True)

    result = evaluate_filesystem_probe(evidence)

    assert result.is_qualified is True
    assert result.reasons == ()
