from rdkit import Chem


def creatMol(z, pos, charges=None):
    """creatMol

    Args:
        z (np.array or torch.Tensor): Long-type atomic numbers
        pos (np.array or torch.Tensor): coordinates
        charges (np.array or torch.Tensor, optional): formal charges. Defaults to None.

    Returns:
        Rdkit.Chem.Mol:
    """
    elements = z.int().tolist()
    coordinates = pos.tolist()
    if charges is not None:
        charges = charges.tolist()

    mol = Chem.RWMol()
    for i, atomic_num in enumerate(elements):
        atom = Chem.Atom(atomic_num)
        if charges is not None:
            atom.SetFormalCharge(int(charges[i]))
        mol.AddAtom(atom)

    conf = Chem.Conformer(len(elements))
    for i, pos in enumerate(coordinates):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)
    return mol


def writeSDF(mol, file_name):
    """
    Write an RDKit molecule to an SDF file.

    :param mol: RDKit Mol object
    :param file_name: Path to the output SDF file
    """

    with open(file_name, "w") as f:
        w = Chem.SDWriter(f)
        w.write(mol)
        w.close()
