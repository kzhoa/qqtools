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

from qqtools.qlogreader import GeneralLogReader

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


def handle_g16_coordinates(lines):
    coords, elements = handle_g16_coord_lines(lines)
    return coords


def handle_g16_std_output(lines, results):
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
    {
        "name": "coords_input",
        "pattern": "Input orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_coordinates,
    },
    {
        "name": "coords_standard",
        "pattern": "Standard orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_std_output,
    },
    {
        "name": "nAtoms",
        "pattern": "NAtoms=",
        "nlines": 1,
        "callback": extract_number_of_atoms,
    },
    {
        "name": "scf_energy",
        "pattern": "SCF Done:",  # Match lines containing SCF Done
        "nlines": 1,  # Read only this matched line
        "skip_when_meet": 0,  # Don't skip any lines
        "callback": extract_scf_energy,  # Use custom callback function to extract energy
    },
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
    # {"name": "homo_lumo", "pattern": "Alpha  occ. eigenvalues", "nlines": 1, "callback": extract_homo_lumo_energies},
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
        "callback": lambda lines: True,
    },
    {
        "name": "coords_input",
        "pattern": "Input orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_coordinates,
    },
    {
        "name": "coords_standard",
        "pattern": "Standard orientation:",
        "end_pattern": "-----",
        "skip_when_meet": 5,
        "callback": handle_g16_std_output,
    },
    {
        "name": "scf_energy",
        "pattern": "SCF Done:",  # Match lines containing SCF Done
        "nlines": 1,  # Read only this matched line
        "skip_when_meet": 0,  # Don't skip any lines
        "callback": extract_scf_energy,  # Use custom callback function to extract energy
    },
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
    # {"name": "homo_lumo", "pattern": "Alpha  occ. eigenvalues", "nlines": 1, "callback": extract_homo_lumo_energies},
]


def create_g16_reader(opt: bool):
    if opt:
        return GeneralLogReader(g16_opt_rules)
    else:
        return GeneralLogReader(g16_singlepoint_rules)
