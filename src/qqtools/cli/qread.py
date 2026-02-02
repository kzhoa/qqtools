import argparse
import json

import numpy as np

from qqtools.plugins.qchem.gaus_reader import create_g16_reader
from qqtools.utils.qtypecheck import numpy_to_native


def write_json(data, file_path):
    data = numpy_to_native(data)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="qRead Tool")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--g16", action="store_true", help="use Gaussian 16 log reader")
    group.add_argument("--xtb", action="store_true", help="use xTB log reader")

    parser.add_argument("input_file", type=str, help="input file path")
    parser.add_argument("-o", "--output", required=True, help="output file path")

    # reliance args
    parser.add_argument("--opt", help="G16 only", default=None, action="store_true")
    args = parser.parse_args()

    # warning check
    if args.g16 and args.opt is None:
        print("Warining: --g16 mode requires --opt argument. Set to False by default")
        args.opt = False

    if args.xtb and args.opt is not None:
        print("Warining:  --opt argument will be ignored under --xtb mode")

    return args


def main():

    args = parse_args()

    # run
    if args.g16:
        print(f"Mode: G16 \nInput: {args.input_file} \nOutput: {args.output}")
        g16Reader = create_g16_reader(opt=args.opt)
        result = g16Reader.read_file(args.input_file)
        write_json(result, args.output)
    elif args.xtb:
        print(f"Mode: xTB \nInput: {args.input_file} \nOutput: {args.output}")
        raise NotImplementedError("xtb mode is not implemented yet")
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
