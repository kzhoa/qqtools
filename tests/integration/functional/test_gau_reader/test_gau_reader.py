from pathlib import Path

this_dir = Path(__file__).resolve().parent


def test_parse():
    fp = this_dir / "examples/ex_homo_lumo.log"

    from qqtools.plugins.qchem.gaus_reader.gaus_reader import create_g16_reader

    reader = create_g16_reader(opt=False)
    results = reader.read_file(fp)
    assert results["HOMO"] == -0.38431
    assert results["LUMO"] == 0.04327
