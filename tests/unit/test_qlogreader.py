from qqtools.plugins.qchem.xtb_reader import xtbLogReader
from qqtools.qlogreader import GeneralLogReader, extract_float


def test_scientific_notation_in_xtb_output():
    result = xtbLogReader.read_lines([":: total energy -1.25E-03 Eh", ":: -> dispersion -2.5e-04 Eh"])
    assert result == {"totalEnergy": -0.00125, "dispersion": -0.00025}
    assert extract_float("1D+02, -.5d-1  +3. -4") == [100.0, -0.05, 3.0, -4.0]


def test_trailing_optional_rules_can_match_or_be_absent():
    reader = GeneralLogReader(
        [
            {"name": "a", "pattern": "A", "nlines": 1, "required": False, "callback": len},
            {"name": "b", "pattern": "B", "nlines": 1, "required": False, "callback": len},
        ]
    )
    assert reader.read_lines(["irrelevant"]) == {}
    assert reader.read_lines(["B"]) == {"b": 1}
    assert reader.read_lines(["A", "B"]) == {"a": 1, "b": 1}


def test_raw_rules_preserve_prior_results_and_feed_later_callbacks():
    reader = GeneralLogReader(
        [
            {"name": "first", "pattern": "A", "nlines": 1, "callback": len},
            {"name": "block", "pattern": "BEGIN", "end_pattern": "END"},
            {"name": "line", "pattern": "DATA", "nlines": 1},
            {"name": "last", "pattern": "Z", "nlines": 1, "callback": len},
        ]
    )
    assert reader.read_lines(["A", " BEGIN", " example ", "END", " DATA ", "Z"]) == {
        "first": 1,
        "block": ["BEGIN", "example", "END"],
        "line": ["DATA"],
        "last": 1,
    }
