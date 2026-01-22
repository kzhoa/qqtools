#!/usr/bin/env python3
"""
Gaussian Log File Processor
Extracts key information from Gaussian log files and saves as JSON format
Supports processing single files or entire directories
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from .gaus_reader import create_g16_reader

READER_INSTANCE = create_g16_reader(opt=False)

__all__ = ["process_gaussian_log"]


def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to Python native types for JSON serialization
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


def process_single_log_file(input_file, output_file=None, verbose=False):
    """
    Process a single Gaussian log file and extract information

    Args:
        input_file (Path): Input log file path (Path object)
        output_file (Path, optional): Output JSON file path (Path object)
        verbose (bool): Whether to display detailed information

    Returns:
        dict: Extracted result data
    """
    if verbose:
        print(f"Processing file: {input_file}")

    # Process file
    try:
        results = READER_INSTANCE.read_file(str(input_file))

        results = convert_numpy_types(results)
        if verbose:
            print(f"Successfully extracted {len(results)} data items")

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        traceback.print_exc()
        return None

    # Determine output file path
    if output_file is None:
        # Auto-generate output filename (same as input file with .json extension)
        output_path = input_file.parent / (input_file.stem + ".json")
        if verbose:
            print(f"No output file specified, saving to: {output_path}")
    else:
        output_path = output_file
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"Results saved to: {output_path}")
            print(f"File size: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        raise e

    return results


def process_directory(input_dir, output_base_dir=None, verbose=False):
    """
    Process all log files in a directory

    Args:
        input_dir (Path): Input directory path
        output_base_dir (Path, optional): Base output directory path
        verbose (bool): Whether to display detailed information

    Returns:
        dict: Processing result statistics
    """
    if verbose:
        print(f"Scanning directory: {input_dir} for log files")

    # Recursively find all .log files
    log_files = list(input_dir.rglob("*.log"))

    if not log_files:
        print(f"No .log or .out files found in directory {input_dir}")
        return {"total_files": 0, "processed_files": 0, "failed_files": 0}

    if verbose:
        print(f"Found {len(log_files)} log/out files")

    stats = {
        "total_files": len(log_files),
        "processed_files": 0,
        "failed_files": 0,
        "failed_file_list": [],  # Record list of failed files
    }

    for log_file in log_files:
        if verbose:
            print(f"\nProcessing file {stats['processed_files'] + 1}/{stats['total_files']}: {log_file.name}")

        # Determine output file path
        if output_base_dir is None:
            # Output to same directory as input file
            output_file = log_file.parent / (log_file.stem + ".json")
        else:
            # Output to specified base directory, keeping same filename
            output_file = output_base_dir / log_file.with_suffix(".json")
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process single file
        try:
            result = process_single_log_file(log_file, output_file, verbose)

            stats["processed_files"] += 1
        except Exception as e:
            stats["failed_files"] += 1
            # Record failed file and error information
            stats["failed_file_list"].append(
                {
                    "file_path": str(log_file),
                    "error_message": str(e),
                }
            )

    return stats


def process_gaussian_log(input_path, output_path=None, verbose=False):
    """
    Process Gaussian log files or directories and extract information

    Args:
        input_path (str): Input log file path or directory path
        output_path (str, optional): Output JSON file path or directory path
        verbose (bool): Whether to display detailed information

    Returns:
        dict or None: Returns result data for single file, statistics for directory
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        # Process single file
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Output to same directory
            output_path = input_path.parent / (input_path.stem + ".json")

        return process_single_log_file(input_path, output_path, verbose)

    elif input_path.is_dir():
        # Process directory
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.parent  # Output to same directory

        return process_directory(input_path, output_path, verbose)

    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process Gaussian log files and extract key information as JSON format\nSupports processing single files or entire directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  Single file:
    %(prog)s calculation.log
    %(prog)s calculation.log results.json
    %(prog)s /path/to/calculation.log /output/path/results.json

  Directory processing:
    %(prog)s /path/to/log_files/
    %(prog)s /path/to/log_files/ /output/path/
    %(prog)s /path/to/log_files/ /output/path/ --verbose

  Note: When processing directories, if no output path is specified,
        JSON files will be saved in the same directory as the log files
        """,
    )

    parser.add_argument("input_path", help="Path to input Gaussian log file or directory containing log files")
    parser.add_argument(
        "output_path", nargs="?", default=None, help="Output JSON file path or directory path (optional)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output mode")

    args = parser.parse_args()

    try:
        # Process file or directory
        result = process_gaussian_log(args.input_path, args.output_path, args.verbose)

        if Path(args.input_path).is_file():  # Single file processing result
            print("\nProcessing complete!")
            for k, v in result.items():
                print(k, v)
        else:  # Directory processing result
            print(
                f"\nProcessing complete! Processed {result['total_files']} files, "
                f"successful: {result['processed_files']}, failed: {result['failed_files']}"
            )
            # Show detailed information for failed files
            if result["failed_files"] > 0:
                print(f"\nFailed file details:")
                for i, failed_file in enumerate(result["failed_file_list"], 1):
                    print(f"  {i}. {failed_file['file_path']}")
                    print(f"     Error: {failed_file['error_message']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
