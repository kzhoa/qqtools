"""
qq: Speed Test
always use counter implementation.
its the most efficient for large lists and large dicts, which is the most common scenario in our use case.
"""

import random
from collections import Counter
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np

# --- Function Definitions (Unchanged) ---


def calculate_refe_original(elements: List[int], atom_ref: Dict[int, float]) -> float:
    if not elements:
        return 0.0
    return sum(atom_ref.get(el) for el in elements)


def calculate_refe_counter(elements: List[int], atom_ref: Dict[int, float]) -> float:
    if not elements:
        return 0.0
    element_counts = Counter(elements)
    return sum(count * atom_ref.get(el) for el, count in element_counts.items())


def calculate_refe_numpy_vectorized(
    elements: List[int],
    atom_ref_values: np.ndarray,
    max_atomic_number: int,  # This max_atomic_number is the highest index supported in atom_ref_values
) -> float:
    if not elements:
        return 0.0
    elements_np = np.array(elements, dtype=int)

    # Ensure atomic numbers are within the valid bounds for indexing the numpy array.
    # Indices are 0-based. If max_atomic_number is 100, valid indices are 0..100.
    # min_atomic_number is fixed at 1, so we check >= 0 (which covers 1 and higher).
    valid_indices_mask = (elements_np >= 0) & (elements_np <= max_atomic_number)
    valid_elements_np = elements_np[valid_indices_mask]

    if valid_elements_np.size == 0:
        return 0.0
    return np.sum(atom_ref_values[valid_elements_np])


# --- Revised Random Data Generation Function ---


def generate_controlled_mock_data_revised(
    num_distinct_atoms_in_ref: int = 20,
    max_atomic_number_overall: int = 100,  # The highest possible atomic number and max index for numpy
    elements_list_length: int = 50,
) -> Tuple[Dict[int, float], List[int], np.ndarray, int]:
    """
    Generates mock data with:
    - Fixed min_atomic_number = 1
    - Fixed ref_energy_range = (-10000.0, 10000.0)
    - Elements list always picks from reference dictionary keys.

    Args:
        num_distinct_atoms_in_ref: How many unique atomic numbers will have assigned energies in the ref dict.
        max_atomic_number_overall: The highest atomic number to consider for the ref dict and numpy array.
        elements_list_length: The number of elements to generate in the list.

    Returns:
        A tuple containing:
        - mock_atom_ref: The generated dictionary of atomic number to reference energy.
        - elements_list: The generated list of atomic numbers (all keys from mock_atom_ref).
        - atom_ref_values_np: The pre-built NumPy array for vectorized lookup.
        - max_atomic_num_for_np: The highest atomic number used for the NumPy array (max_atomic_number_overall).
    """
    # --- Hardcoded constants ---
    min_atomic_number = 1
    ref_energy_range = (-10000.0, 10000.0)

    # --- Parameter validation and adjustment ---
    if min_atomic_number > max_atomic_number_overall:
        raise ValueError(
            f"min_atomic_number ({min_atomic_number}) cannot be greater than max_atomic_number_overall ({max_atomic_number_overall})"
        )

    # Calculate the total number of unique atomic numbers possible in the defined range
    possible_range_size = max_atomic_number_overall - min_atomic_number + 1

    # If requested distinct atoms exceed the possible range, clamp it.
    if num_distinct_atoms_in_ref > possible_range_size:
        print(
            f"Warning: num_distinct_atoms_in_ref ({num_distinct_atoms_in_ref}) exceeds possible range ({possible_range_size} from {min_atomic_number} to {max_atomic_number_overall}). Clamping to {possible_range_size}."
        )
        num_distinct_atoms_in_ref = possible_range_size

    # --- Data Generation ---
    mock_atom_ref: Dict[int, float] = {}

    # Determine which atomic numbers will have assigned energies in the reference dictionary
    # All possible atomic numbers in the system are from min_atomic_number to max_atomic_number_overall
    all_possible_atomic_nums = list(range(min_atomic_number, max_atomic_number_overall + 1))

    reference_keys = []
    if num_distinct_atoms_in_ref > 0:
        # Randomly select 'num_distinct_atoms_in_ref' atomic numbers to be keys in the reference dict
        selected_atomic_nums = random.sample(all_possible_atomic_nums, num_distinct_atoms_in_ref)

        # Assign random energies within the specified range to these selected atomic numbers
        for atomic_num in selected_atomic_nums:
            mock_atom_ref[atomic_num] = random.uniform(ref_energy_range[0], ref_energy_range[1])

        reference_keys = selected_atomic_nums  # Store these keys for elements list generation

    # Create the full NumPy array for vectorized lookups.
    # The size needs to be max_atomic_number_overall + 1 to accommodate index max_atomic_number_overall.
    # All values are initialized to 0.0, representing atomic numbers not explicitly in mock_atom_ref.
    atom_ref_values_np = np.zeros(max_atomic_number_overall + 1, dtype=float)

    # Populate the numpy array with the assigned energies from mock_atom_ref
    for atomic_num, energy in mock_atom_ref.items():
        # Indexing is safe as atomic_num is guaranteed to be within [min_atomic_number, max_atomic_number_overall]
        atom_ref_values_np[atomic_num] = energy

    # Generate the elements list: STRICTLY pick from reference_keys.
    # This ensures every element in elements_list is a key present in mock_atom_ref.
    elements_list: List[int] = []

    if not reference_keys:  # Handle edge case where num_distinct_atoms_in_ref is 0
        print(
            "Warning: Reference dictionary is empty (num_distinct_atoms_in_ref was 0). Cannot generate elements list from reference keys. Filling with random numbers in overall range."
        )
        # Ensure randint range is valid (max >= min) before calling.
        safe_max_num = max(min_atomic_number, max_atomic_number_overall)
        for _ in range(elements_list_length):
            elements_list.append(random.randint(min_atomic_number, safe_max_num))
    else:
        # Pick elements_list_length times from the reference_keys
        for _ in range(elements_list_length):
            elements_list.append(random.choice(reference_keys))

    random.shuffle(elements_list)  # Shuffle the list for randomness in processing order

    # Return:
    # - mock_atom_ref: The dictionary itself.
    # - elements_list: The list of elements to process.
    # - atom_ref_values_np: The NumPy array for vectorized lookups.
    # - max_atomic_number_overall: The maximum index used in the NumPy array.
    return mock_atom_ref, elements_list, atom_ref_values_np, max_atomic_number_overall


# --- Configuration Definitions ---


def run_test_scenario(scenario_name: str, config: Dict[str, Any], num_runs: int = 100):
    """
    Runs the performance test for a given scenario configuration.
    Returns a dictionary with the configuration parameters and performance results.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Generate data using the revised generator function
    (mock_atom_ref, mock_elements, atom_ref_values_np, max_atomic_num_for_np) = generate_controlled_mock_data_revised(
        **config
    )

    print(
        f"  Data Info: Dict Size={len(mock_atom_ref)}, List Length={len(mock_elements)}, Max Atomic Num={max_atomic_num_for_np}"
    )

    results = {}
    # --- Performance Calculations ---
    try:
        start_time = perf_counter()
        for _ in range(num_runs):
            refe_orig_perf = calculate_refe_original(mock_elements, mock_atom_ref)
        end_time = perf_counter()
        results["Original"] = (end_time - start_time) / num_runs
    except Exception as e:
        print(f"Original: Error - {e}")
        results["Original"] = float("inf")

    try:
        start_time = perf_counter()
        for _ in range(num_runs):
            refe_count_perf = calculate_refe_counter(mock_elements, mock_atom_ref)
        end_time = perf_counter()
        results["Counter"] = (end_time - start_time) / num_runs
    except Exception as e:
        print(f"Counter: Error - {e}")
        results["Counter"] = float("inf")

    try:
        start_time = perf_counter()
        for _ in range(num_runs):
            # Pass the max_atomic_num_for_np correctly to the vectorized function
            refe_np_perf = calculate_refe_numpy_vectorized(mock_elements, atom_ref_values_np, max_atomic_num_for_np)
        end_time = perf_counter()
        results["NumPy"] = (end_time - start_time) / num_runs
    except Exception as e:
        print(f"NumPy: Error - {e}")
        results["NumPy"] = float("inf")

    # Return config details along with performance results
    return {
        "Scenario": scenario_name,
        "Dict Size": len(mock_atom_ref),
        "List Length": len(mock_elements),
        "Original": results["Original"],
        "Counter": results["Counter"],
        "NumPy": results["NumPy"],
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    # Define different configurations for testing
    # Focus on varying num_distinct_atoms_in_ref (Dict Size) and elements_list_length
    test_configs = {
        # Small Dictionary Tier (10 distinct atoms in ref)
        "Small Dict (10), List (100)": {
            "num_distinct_atoms_in_ref": 10,
            "max_atomic_number_overall": 50,  # Max possible atomic number in the system
            "elements_list_length": 100,
        },
        "Small Dict (10), List (1k)": {
            "num_distinct_atoms_in_ref": 10,
            "max_atomic_number_overall": 50,
            "elements_list_length": 1000,
        },
        "Small Dict (10), List (10k)": {
            "num_distinct_atoms_in_ref": 10,
            "max_atomic_number_overall": 50,
            "elements_list_length": 10000,
        },
        "Small Dict (10), List (100k)": {
            "num_distinct_atoms_in_ref": 10,
            "max_atomic_number_overall": 50,
            "elements_list_length": 100000,
        },
        # Medium Dictionary Tier (50 distinct atoms in ref)
        "Medium Dict (50), List (100)": {
            "num_distinct_atoms_in_ref": 50,
            "max_atomic_number_overall": 100,
            "elements_list_length": 100,
        },
        "Medium Dict (50), List (1k)": {
            "num_distinct_atoms_in_ref": 50,
            "max_atomic_number_overall": 100,
            "elements_list_length": 1000,
        },
        "Medium Dict (50), List (10k)": {
            "num_distinct_atoms_in_ref": 50,
            "max_atomic_number_overall": 100,
            "elements_list_length": 10000,
        },
        "Medium Dict (50), List (100k)": {
            "num_distinct_atoms_in_ref": 50,
            "max_atomic_number_overall": 100,
            "elements_list_length": 100000,
        },
        # *** Renamed: Large Dictionary Tier (100 distinct atoms in ref) ***
        "Large Dict (100), List (100)": {
            "num_distinct_atoms_in_ref": 100,
            "max_atomic_number_overall": 150,  # Max possible atomic number in the system
            "elements_list_length": 100,
        },
        "Large Dict (100), List (1k)": {
            "num_distinct_atoms_in_ref": 100,
            "max_atomic_number_overall": 150,
            "elements_list_length": 1000,
        },
        "Large Dict (100), List (10k)": {
            "num_distinct_atoms_in_ref": 100,
            "max_atomic_number_overall": 150,
            "elements_list_length": 10000,
        },
        "Large Dict (100), List (100k)": {
            "num_distinct_atoms_in_ref": 100,
            "max_atomic_number_overall": 150,
            "elements_list_length": 100000,
        },
        # *** End Renamed ***
        # *** Renamed: Very Large Dictionary Tier (200 distinct atoms in ref) ***
        "Very Large Dict (200), List (100)": {
            "num_distinct_atoms_in_ref": 200,
            "max_atomic_number_overall": 300,  # Needs to be large enough for 200 distinct atoms
            "elements_list_length": 100,
        },
        "Very Large Dict (200), List (1k)": {
            "num_distinct_atoms_in_ref": 200,
            "max_atomic_number_overall": 300,
            "elements_list_length": 1000,
        },
        "Very Large Dict (200), List (10k)": {
            "num_distinct_atoms_in_ref": 200,
            "max_atomic_number_overall": 300,
            "elements_list_length": 10000,
        },
        "Very Large Dict (200), List (100k)": {
            "num_distinct_atoms_in_ref": 200,
            "max_atomic_number_overall": 300,
            "elements_list_length": 100000,
        },
        # *** End Renamed ***
    }

    all_scenario_results = []
    for name, config in test_configs.items():
        # Adjust num_runs based on expected computation time, prioritizing longer lists
        current_num_runs = 100  # Default for smaller lists
        if config["elements_list_length"] >= 100000:
            current_num_runs = 20
        elif config["elements_list_length"] >= 10000:
            current_num_runs = 50
        elif config["elements_list_length"] >= 1000:
            current_num_runs = 75
        # Lists of length 100 will use the default 100 runs.

        scenario_data = run_test_scenario(name, config, num_runs=current_num_runs)
        scenario_data["Scenario"] = name  # Store scenario name for summary
        all_scenario_results.append(scenario_data)

    print("\n" + "=" * 115)
    print("--- Performance Summary: Impact of Dictionary Size and List Length ---")
    print("=" * 115)
    print(
        f"{'Scenario':<35} | {'Dict Size':>12} | {'List Length':>15} | {'Original (s)':>15} | {'Counter (s)':>15} | {'NumPy (s)':>15}"
    )
    print("-" * 115)

    for result in all_scenario_results:
        dict_size_str = f"{result['Dict Size']:,}"
        list_len_str = f"{result['List Length']:,}"

        orig_time = f"{result['Original']:.6f}" if result.get("Original") != float("inf") else "Error"
        count_time = f"{result['Counter']:.6f}" if result.get("Counter") != float("inf") else "Error"
        numpy_time = f"{result['NumPy']:.6f}" if result.get("NumPy") != float("inf") else "Error"

        scenario_name_display = result.get("Scenario", "N/A")
        if len(scenario_name_display) > 33:
            scenario_name_display = scenario_name_display[:30] + "..."

        print(
            f"{scenario_name_display:<35} | {dict_size_str:>12} | {list_len_str:>15} | {orig_time:>15} | {count_time:>15} | {numpy_time:>15}"
        )
    print("-" * 115)
