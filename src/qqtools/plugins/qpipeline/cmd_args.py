import argparse
import warnings
from pathlib import Path

import qqtools as qt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class BoolOrFlagAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, str2bool(values))


def basic_argparser():
    parser = argparse.ArgumentParser("QQ BASIC ARGS")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file (default: None)")
    parser.add_argument(
        "--ckp",
        "--ckp-file",
        dest="ckp_file",
        type=str,
        default=None,
        help="Path to checkpoint file (support both --ckp and --ckp_file)",
    )
    parser.add_argument("--test", action="store_true", help="whether use infer mode")
    parser.add_argument(
        "--ddp-detect",
        dest="ddp_detect",
        action=BoolOrFlagAction,
        nargs="?",
        const=True,
        default=False,
        help="auto detect ddp env",
    )
    parser.add_argument(
        "--ddp",
        dest="ddp_detect",
        action=BoolOrFlagAction,
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="auto detect ddp env, same as --ddp-detect",
    )

    parser.add_argument("--local-rank", type=int, default=None, help="for ddp compatibility, not used")
    return parser


def merge_basic_args(cmd_args):
    BASIC_KEYS = ["config", "ckp_file", "test", "local_rank"]

    # Prioritize using configfile
    base_args = qt.qDict()
    if cmd_args.config is not None and Path(cmd_args.config).exists():
        file_args = qt.load_yaml(cmd_args.config)
        base_args.recursive_update(file_args)
    else:
        warnings.warn(f"{cmd_args.config} not found, config file will be ignored.", UserWarning)

    # provide
    if cmd_args.ckp_file is not None:
        base_args.ckp_file = cmd_args.ckp_file
    base_args.test = cmd_args.test

    # merge extra keys
    for k, v in cmd_args.items():
        if k not in BASIC_KEYS:
            base_args[k] = v
    return base_args


def prepare_cmd_args(patch=None):
    """cmd config"""
    parser = basic_argparser()
    if patch is not None:
        parser = patch(parser)
    cmd_args = parser.parse_args()
    cmd_args = qt.qDict.from_namespace(cmd_args)

    args = merge_basic_args(cmd_args)
    return args
