#!/usr/bin/env python3
"""
Gaussian Log File Processor
Extract key information from Gaussian log files and save as JSON format
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from .gaus_reader import create_g16_reader

__all__ = ["process_gaussian_log"]


def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to native Python types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return complex(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.bytes_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def process_gaussian_log(input_file, output_file=None):
    """
    Process Gaussian log file and extract information

    Args:
        input_file (str): Path to input log file
        output_file (str, optional): Path to output JSON file

    Returns:
        dict: Extracted result data
    """

    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if not input_path.suffix.lower() in [".log", ".out"]:
        print(f"Warning: Input file extension is not .log or .out, but will attempt to process anyway: {input_file}")

    print(f"Processing file: {input_file}")

    # Create reader and process file
    try:
        reader = create_g16_reader(False)
        results = reader.read_file(str(input_path))

        results = convert_numpy_types(results)
        print(f"Successfully extracted {len(results)} data items")

        # for k, v in results.items():
        #     print(k, " ", v)

    except Exception as e:
        print(f"Error processing file: {e}")
        traceback.print_exc()

    # Determine output file path
    if output_file is None:
        # Auto-generate output filename (same as input file with .json extension)
        output_path = input_path.parent / (input_path.stem + ".json")
        print(f"No output file specified, will save to: {output_path}")
    else:
        output_path = Path(output_file)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"Error saving results")
        raise e

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process Gaussian log files and extract key information as JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s calculation.log
  %(prog)s calculation.log results.json
  %(prog)s /path/to/calculation.log /output/path/results.json
        """,
    )

    parser.add_argument("input_file", help="Path to input Gaussian log file")
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Path to output JSON file (optional, defaults to same name with .json extension)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output mode")

    args = parser.parse_args()

    try:
        # Process file
        results = process_gaussian_log(args.input_file, args.output_file)
        print("\nProcessing complete!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
