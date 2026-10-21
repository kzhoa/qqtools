"""Frequency rules must not consume subsequent thermochemical properties."""

import pytest

from qqtools.plugins.qchem.gaus_reader.gaus_reader import (
    extract_frequencies,
    frequency_rule,
    g16_opt_rules,
    g16_singlepoint_rules,
)
from qqtools.qlogreader import GeneralLogReader

BLOCK = """Harmonic frequencies (cm**-1), IR intensities (KM/Mole)
 Low frequencies --- -8.0 0.0 1.0
 1 2 3
 A A A
 Frequencies -- -12.5 1.23D+02 2.34e+02
 Red. masses -- 1 1 1
 4
 A
 Frequencies -- 345.6
 - Thermochemistry -
 Zero-point correction= 0.123 (Hartree/Particle)
 Thermal correction to Energy= 0.234
 Thermal correction to Enthalpy= 0.345
 Thermal correction to Gibbs Free Energy= 0.456
 filler
 filler
 Forces (Hartrees/Bohr)
 header
 -----
 1 6 -0.100 0.200 -0.300
 -----
"""


@pytest.mark.parametrize("rules", [g16_singlepoint_rules, g16_opt_rules])
def test_optional_frequency_rule_preserves_following_properties(rules):
    start = next(i for i, rule in enumerate(rules) if rule["name"] == "freq")
    subset = rules[start:]
    old = GeneralLogReader([rule for rule in subset if rule["name"] != "freq"]).read_lines(BLOCK.splitlines())
    new = GeneralLogReader(subset).read_lines(BLOCK.splitlines())
    assert new.pop("freq") == [-12.5, 123.0, 234.0, 345.6]
    assert new == old
    assert new["zpe_correction"] == 0.123
    assert new["thermal_corrections"] == {"energy": 0.234, "enthalpy": 0.345, "gibbs": 0.456}
    assert new["forces"] == [[-0.1, 0.2, -0.3]]


@pytest.mark.parametrize("rules", [g16_singlepoint_rules, g16_opt_rules])
def test_missing_frequency_block_does_not_block_existing_rules(rules):
    start = next(i for i, rule in enumerate(rules) if rule["name"] == "freq")
    text = "Zero-point correction=" + BLOCK.split("Zero-point correction=", 1)[1]
    result = GeneralLogReader(rules[start:]).read_lines(iter(text.splitlines()))
    assert "freq" not in result
    assert result["zpe_correction"] == 0.123
    assert result["forces"] == [[-0.1, 0.2, -0.3]]


def test_truncated_block_does_not_return_partial_frequencies():
    rules = [frequency_rule(), {"name": "end", "pattern": "END", "nlines": 1}]
    result = GeneralLogReader(rules).read_lines(BLOCK.split(" - Thermochemistry -")[0].splitlines())
    assert "freq" not in result


@pytest.mark.parametrize(
    "replacement",
    [
        "Frequencies -- ***** 123 234",
        "Frequencies -- nan 123 234",
        "Frequencies -- inf 123 234",
        "Frequencies -- 1 2",
        "Frequencies -- 1 2 3 4",
    ],
)
def test_malformed_frequency_values_are_not_silently_accepted(replacement):
    text = BLOCK.replace("Frequencies -- -12.5 1.23D+02 2.34e+02", replacement)
    with pytest.raises(ValueError):
        extract_frequencies(text.splitlines())


def test_repeated_modes_are_rejected():
    with pytest.raises(ValueError, match="mode numbers"):
        extract_frequencies(BLOCK.replace(" 4\n", " 1\n").splitlines())


def test_empty_frequency_block_is_rejected():
    with pytest.raises(ValueError, match="no Frequencies"):
        extract_frequencies(["Harmonic frequencies", "Thermochemistry"])


HIGH = """ 1 2 3 4 5
 A A A A A
 Frequencies --- -12.123456 1.23123456D+02 234.123456 345.123456 456.123456
 6
 A
 Frequencies --- 567.123456
"""
ORDINARY = """ 1 2 3
 A A A
 Frequencies -- -12.1235 123.1235 234.1235
 4 5 6
 A A A
 Frequencies -- 345.1235 456.1235 567.1235
"""
EXPECTED_HIGH = [-12.123456, 123.123456, 234.123456, 345.123456, 456.123456, 567.123456]


@pytest.mark.parametrize("text", [HIGH, HIGH + ORDINARY, ORDINARY + HIGH])
def test_hpmodes_is_preferred_without_duplicate_modes(text):
    assert extract_frequencies(text.splitlines()) == EXPECTED_HIGH


def test_hpmodes_without_symmetry_labels():
    assert extract_frequencies(HIGH.replace(" A A A A A\n", "").replace(" A\n", "").splitlines()) == EXPECTED_HIGH


@pytest.mark.parametrize("rules", [g16_singlepoint_rules, g16_opt_rules])
def test_hpmodes_rule_preserves_following_properties(rules):
    text = (
        "Harmonic frequencies (cm**-1)\n"
        + HIGH
        + ORDINARY
        + " - Thermochemistry -"
        + BLOCK.split(" - Thermochemistry -", 1)[1]
    )
    start = next(i for i, rule in enumerate(rules) if rule["name"] == "freq")
    new = GeneralLogReader(rules[start:]).read_lines(text.splitlines())
    old = GeneralLogReader([rule for rule in rules[start:] if rule["name"] != "freq"]).read_lines(text.splitlines())
    assert new.pop("freq") == EXPECTED_HIGH
    assert new == old
    assert new["zpe_correction"] == 0.123
    assert new["forces"] == [[-0.1, 0.2, -0.3]]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (HIGH + ORDINARY.replace("567.1235", "568.1235"), "values disagree"),
        (HIGH + ORDINARY.replace("4 5 6", "4 5").replace(" 567.1235", ""), "counts disagree"),
        (HIGH + HIGH, "mode numbers"),
        (HIGH.replace(" 6\n", " 7\n"), "mode numbers"),
        (HIGH.replace("567.123456", "nan"), "not finite"),
        (HIGH.replace("567.123456", "*****"), "Malformed"),
    ],
)
def test_hpmodes_inconsistency_is_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        extract_frequencies(text.splitlines())
