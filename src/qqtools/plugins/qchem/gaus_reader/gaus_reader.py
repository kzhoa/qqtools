"""
Usage example:
from qqtools.qm.g16 import create_g16_reader

g16Reader = create_g16_reader(opt=True)  # for optimization log files
g16Reader = create_g16_reader(opt=False)  # for single-point log files

results = g16Reader.read_file('output.log')
"""

import re
from typing import Dict, List

import numpy as np

from qqtools.qlogreader import GeneralLogReader, extract_float

__all__ = ["create_g16_reader"]


def extract_molecule_charge_multiplicity(lines: List[str]) -> tuple:
    for line in lines:
        if "Charge =" in line and "Multiplicity =" in line:
            parts = line.split()
            charge = int(parts[2])
            multiplicity = int(parts[5])
            return charge, multiplicity
    return None, None


def handle_g16_coord_lines(lines):
    elements = []
    coords = []
    for line in lines:
        eles = line.split()
        assert len(eles) == 6
        idx, atom_number, a_type, x, y, z = eles
        elements.append(int(atom_number))
        coords.append((float(x), float(y), float(z)))

    coords = np.array(coords)
    elements = np.array(elements)
    return coords, elements


def handle_g16_input_coords(lines):
    lines = lines[:-1]  # remove end line
    coords, elements = handle_g16_coord_lines(lines)
    return coords


def handle_g16_std_coords(lines, results):
    lines = lines[:-1]  # remove end line
    coords, elements = handle_g16_coord_lines(lines)
    results["coords_standard"] = coords
    results["elements"] = elements
    # results["nAtoms"] = len(elements)
    return results


def extract_number_of_atoms(lines: List[str]) -> int:
    for line in lines:
        if "NAtoms=" in line:
            match = re.search(r"NAtoms=\s*(\d+)", line)
            if match:
                return int(match.group(1))
    return 0


def extract_scf_energy(lines):
    """
    Extract energy value from SCF Done line
    Typical format: ' SCF Done:  E(RB3LYP) =  -114.567890123     A.U. after   10 cycles'
    """
    if not lines:
        return None

    # Iterate through all read lines to find lines containing SCF Done
    for line in lines:
        if "SCF Done:" in line:
            # Use regular expression to match energy value (scientific notation or decimal format)
            # Match pattern: -114.567890123 or -1.14567890123E+02 etc.
            energy_match = re.search(r"E\([A-Za-z0-9]+\)\s*=\s*([-+]?\d*\.?\d+(?:[EDed][-+]?\d+)?)", line)
            if energy_match:
                energy_str = energy_match.group(1)
                try:
                    # Handle possible scientific notation representation (e.g., D in Fortran output)
                    if "D" in energy_str.upper():
                        energy_value = float(energy_str.replace("D", "E"))
                    else:
                        energy_value = float(energy_str)
                    return energy_value
                except ValueError:
                    continue
    return None


def extract_homo_lumo(lines: List[str], results: dict):
    HOMO = extract_float(lines[-2])[-1]
    LUMO = extract_float(lines[-1])[0]
    results["HOMO"] = HOMO
    results["LUMO"] = LUMO
    return results


def extract_zpe_correction(lines: List[str]) -> float:
    for line in lines:
        if "Zero-point correction=" in line:
            match = re.search(r"Zero-point correction=\s*([-\d.]+)", line)
            if match:
                return float(match.group(1))
    return None


def extract_thermal_correction(lines: List[str]) -> Dict[str, float]:
    """Extract Thermalization Correction"""
    corrections = {}
    for line in lines:
        if "Thermal correction to Energy=" in line:
            match = re.search(r"Thermal correction to Energy=\s*([-\d.]+)", line)
            if match:
                corrections["energy"] = float(match.group(1))
        elif "Thermal correction to Enthalpy=" in line:
            match = re.search(r"Thermal correction to Enthalpy=\s*([-\d.]+)", line)
            if match:
                corrections["enthalpy"] = float(match.group(1))
        elif "Thermal correction to Gibbs Free Energy=" in line:
            match = re.search(r"Thermal correction to Gibbs Free Energy=\s*([-\d.]+)", line)
            if match:
                corrections["gibbs"] = float(match.group(1))
    return corrections


def extract_forces(lines: List[str]) -> dict:
    """
    Extract forces from given lines.
    Each line format example:
         "1        8          -0.000451775    0.000736685   -0.000457916"
    return: {'forces': [[fx, fy, fz], ...]}
    """

    lines = lines[:-1]  # remove end line
    forces = []

    for line in lines:
        # fmt: idx + atomic_number + fx + fy + fz
        force_match = re.search(
            r"^\s*\d+\s+\d+\s+([-+]?\d+\.\d+(?:E[+-]?\d+)?)\s+([-+]?\d+\.\d+(?:E[+-]?\d+)?)\s+([-+]?\d+\.\d+(?:E[+-]?\d+)?)",
            line,
        )
        if force_match:
            fx = float(force_match.group(1))
            fy = float(force_match.group(2))
            fz = float(force_match.group(3))
            forces.append([fx, fy, fz])

    return forces


def extract_dipole_moment_final(lines):
    """
    Parse dipole moment data.

    Args:
        lines: List of four lines containing Tot, x, y, z dipole moment data
               Format example:
               Tot        0.153584D+01      0.390371D+01      0.130214D+02
               x          0.151568D+01      0.385248D+01      0.128505D+02
               y          0.442014D-01      0.112349D+00      0.374755D+00
               z          0.244027D+00      0.620254D+00      0.206895D+01

    Returns:
        Dictionary containing x, y, z components and total value in Debye units
        Example: {'total': 3.90371, 'x': 3.85248, 'y': 0.112349, 'z': 0.620254}
    """
    dipole_data = {}

    for line in lines:
        if not line.strip():
            continue

        # Split the line into columns
        parts = line.split()

        # parts[0] contains label (Tot, x, y, z)
        # parts[1] contains value in atomic units (au)
        # parts[2] contains value in Debye units
        # parts[3] contains value in SI units (10^-30 C·m)

        label = parts[0].lower()  # Convert label to lowercase for consistency

        # Convert Fortran scientific notation (D) to Python (E) and parse to float
        value_debye = float(parts[2].replace("D", "E"))

        # Store in dictionary
        dipole_data[label] = value_debye

    return (dipole_data["x"], dipole_data["y"], dipole_data["z"])


# sp without optimization
g16_singlepoint_rules = [
    # --- basic information ---
    # need complex information, maybe later
    # {
    #     "name": "method_basis",
    #     "pattern": "#",
    #     "nlines": 1,
    #     "callback": extract_method_basis,
    # },
    {
        "name": "charge_multiplicity",
        "pattern": "Charge =",
        "nlines": 1,
        "callback": extract_molecule_charge_multiplicity,
    },
    # --- enter l202
    {
        "name": "coords_input",
        "pattern": "Input orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_input_coords,
        "required": False,  # input orientation does not always exist
    },
    {
        "name": "coords_standard",
        "pattern": "Standard orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_std_coords,
    },
    {
        "name": "nAtoms",
        "pattern": "NAtoms=",
        "nlines": 1,
        "callback": extract_number_of_atoms,
    },
    # --- leave l202 ---
    # --- enter l508
    {
        "name": "scf_energy",
        "pattern": "SCF Done:",  # Match lines containing SCF Done
        "nlines": 1,  # Read only this matched line
        "skip_when_meet": 0,  # Don't skip any lines
        "callback": extract_scf_energy,  # Use custom callback function to extract energy
    },
    # --- leave l508 ---
    # --- enetr l1002
    {
        "name": "isotropic_polarizability",
        "pattern": "Isotropic polarizability for W=",
        "nlines": 1,
        "skip_when_meet": 0,
        "callback": lambda lines: extract_float(lines[0])[1],
        "required": True,
    },
    # --- leave l1002
    # --- enter l601 ---
    {
        "name": "population_analysis_sign",
        "pattern": "Population analysis using the SCF Density",
        "nlines": 0,
        "skip_when_meet": 0,
        "callback": lambda lines, results: results,
    },
    {
        "name": "HOMO-LUMO",
        "pattern": "Alpha  occ. eigenvalues",
        "nlines": -1,
        "end_pattern": "Alpha virt. eigenvalues",
        "skip_when_meet": 0,
        "callback": extract_homo_lumo,
    },
    # Electronic spatial extent R^2
    {
        "name": "elec_spatial_extent",
        "pattern": "Electronic spatial extent (au):",
        "nlines": 1,
        "callback": lambda lines: extract_float(lines[0])[-1],
    },
    #  Dipole Moment `μ`
    {
        "name": "dipole_moment",
        "pattern": "Dipole moment (field-independent basis, Debye):",
        "nlines": 2,
        "callback": lambda lines: extract_float(lines[1])[:3],
    },
    # --- leave l601 ---
    # frequencies, maybe later
    {
        "name": "zpe_correction",
        "pattern": "Zero-point correction=",
        "nlines": 1,
        "callback": extract_zpe_correction,
    },
    {
        "name": "thermal_corrections",
        "pattern": "Thermal correction to Energy=",
        "nlines": 5,
        "callback": extract_thermal_correction,
    },
    # force
    {
        "name": "forces",
        "pattern": "Forces (Hartrees/Bohr)",
        "end_pattern": "-----",
        "skip_when_meet": 3,
        "callback": extract_forces,
    },
    # --- enter l19999
    {
        "name": "dipole_moment_inpt_orient",
        "pattern": "Electric dipole moment (input orientation)",
        "skip_when_meet": 3,
        "nlines": 4,
        "callback": extract_dipole_moment_final,
    },
    # --- leave l19999
]

# sp with optimization
g16_opt_rules = [
    # --- basic information ---
    # need complex information, maybe later
    # {
    #     "name": "method_basis",
    #     "pattern": "#",
    #     "nlines": 1,
    #     "callback": extract_method_basis,
    # },
    {
        "name": "charge_multiplicity",
        "pattern": "Charge =",
        "nlines": 1,
        "callback": extract_molecule_charge_multiplicity,
    },
    # only take effect when `opt` in the routine
    {
        "name": "opt_complete_sign",
        "pattern": "Optimization completed",
        "nlines": 0,
        "callback": lambda lines, results: results,
    },
    # --- enter l202
    {
        "name": "coords_input",
        "pattern": "Input orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_input_coords,
        "required": False,  # input orientation does not always exist
    },
    {
        "name": "coords_standard",
        "pattern": "Standard orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_std_coords,
    },
    # --- leave l202 ---
    # --- enter l508
    {
        "name": "scf_energy",
        "pattern": "SCF Done:",  # Match lines containing SCF Done
        "nlines": 1,  # Read only this matched line
        "skip_when_meet": 0,  # Don't skip any lines
        "callback": extract_scf_energy,  # Use custom callback function to extract energy
    },
    # --- leave l508 ---
    # --- enetr l1002
    # Isotropic Polarizability
    {
        "name": "isotropic_polarizability",
        "pattern": "Isotropic polarizability for W=",
        "nlines": 1,
        "skip_when_meet": 0,
        "callback": lambda lines: extract_float(lines[0])[1],
        "required": True,
    },
    # --- leave l1002
    # --- enter l601 ---
    {
        "name": "population_analysis_sign",
        "pattern": "Population analysis using the SCF Density",
        "nlines": 0,
        "skip_when_meet": 0,
        "callback": lambda lines, results: results,
    },
    {
        "name": "HOMO-LUMO",
        "pattern": "Alpha  occ. eigenvalues",
        "nlines": -1,
        "end_pattern": "Alpha virt. eigenvalues",
        "skip_when_meet": 0,
        "callback": extract_homo_lumo,
    },
    # Electronic spatial extent R^2
    {
        "name": "elec_spatial_extent",
        "pattern": "Electronic spatial extent (au):",
        "nlines": 1,
        "callback": lambda lines: extract_float(lines[0])[-1],
    },
    #  Dipole Moment `μ`
    {
        "name": "dipole_moment_std_orient",
        "pattern": "Dipole moment (field-independent basis, Debye):",
        "nlines": 2,
        "callback": lambda lines: extract_float(lines[1])[:3],
    },
    # --- leave l601 ---
    # --- enter l716
    # frequencies, maybe later
    {
        "name": "zpe_correction",
        "pattern": "Zero-point correction=",
        "nlines": 1,
        "callback": extract_zpe_correction,
    },
    {
        "name": "thermal_corrections",
        "pattern": "Thermal correction to Energy=",
        "nlines": 5,
        "callback": extract_thermal_correction,
    },
    # force
    {
        "name": "forces",
        "pattern": "Forces (Hartrees/Bohr)",
        "skip_when_meet": 3,
        "end_pattern": "-----",
        "callback": extract_forces,
    },
    # --- leave l716 ---
    # --- enter l19999
    {
        "name": "dipole_moment_inpt_orient",
        "pattern": "Electric dipole moment (input orientation)",
        "skip_when_meet": 3,
        "nlines": 4,
        "callback": extract_dipole_moment_final,
    },
    # --- leave l19999
]


def create_g16_reader(opt: bool):
    if opt:
        return GeneralLogReader(g16_opt_rules)
    else:
        return GeneralLogReader(g16_singlepoint_rules)
