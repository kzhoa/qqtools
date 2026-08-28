from qqtools.plugins.qexp.lease import LeasePolicy, chrony_health


def test_chrony_health_rejects_large_offset() -> None:
    class Completed:
        returncode = 0
        stdout = (
            "System time     : 2.0 seconds slow of NTP time\n"
            "Root delay      : 0.1 seconds\n"
            "Root dispersion : 0.1 seconds\n"
            "Skew            : 0.01 ppm\n"
            "Leap status     : Normal\n"
        )

    result = chrony_health(LeasePolicy(), run=lambda *_args, **_kwargs: Completed())

    assert result == (False, "chrony_error_bound_exceeds_policy")
