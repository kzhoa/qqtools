import argparse
import re
import sys
import warnings
from pathlib import Path

import qqtools as qt

MAX_DOTTED_OVERRIDE_DEPTH = 8
_VALID_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GENERIC_BOOL_LITERALS = {
    "yes": True,
    "true": True,
    "t": True,
    "y": True,
    "no": False,
    "false": False,
    "f": False,
    "n": False,
}


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


def _collect_parser_owned_dests(parser: argparse.ArgumentParser) -> set[str]:
    owned_dests = set()
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        owned_dests.add(action.dest)
    return owned_dests


def _collect_reserved_top_level_keys(parser_owned_dests: set[str]) -> set[str]:
    return {dest for dest in parser_owned_dests if "." not in dest}


def _split_dotted_override_tokens(parser: argparse.ArgumentParser, tokens: list[str]) -> list[str]:
    override_tokens = []
    invalid_tokens = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if not token.startswith("--"):
            invalid_tokens.append(token)
            index += 1
            continue

        key = token[2:].split("=", 1)[0]
        if "." not in key:
            invalid_tokens.append(token)
            index += 1
            continue

        override_tokens.append(token)
        if "=" not in token:
            next_index = index + 1
            if next_index < len(tokens) and not tokens[next_index].startswith("-"):
                override_tokens.append(tokens[next_index])
                index += 1
        index += 1

    if invalid_tokens:
        parser.error(f"unrecognized arguments: {' '.join(invalid_tokens)}")
    return override_tokens


def _parse_dotted_override_tokens(tokens: list[str]) -> list[tuple[str, str | None, bool]]:
    overrides = []
    seen_keys = set()
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Invalid override token '{token}'.")

        body = token[2:]
        is_flag_only = False
        if not body:
            raise ValueError(f"Invalid override token '{token}'.")

        if "=" in body:
            key, raw_value = body.split("=", 1)
        else:
            key = body
            next_index = index + 1
            if next_index < len(tokens) and not tokens[next_index].startswith("-"):
                raw_value = tokens[next_index]
                index += 1
            else:
                raw_value = None
                is_flag_only = True

        if key in seen_keys:
            raise ValueError(f"Duplicate dotted override key '{key}'.")
        seen_keys.add(key)
        overrides.append((key, raw_value, is_flag_only))
        index += 1

    return overrides


def _validate_override_key(
    key: str,
    parser_owned_dests: set[str],
    reserved_top_level_keys: set[str],
) -> list[str]:
    path_segments = key.split(".")
    if any(segment == "" for segment in path_segments):
        raise ValueError(f"Invalid override key '{key}'.")
    if len(path_segments) > MAX_DOTTED_OVERRIDE_DEPTH:
        raise ValueError(f"Override key '{key}' exceeds max depth {MAX_DOTTED_OVERRIDE_DEPTH}.")
    if key in parser_owned_dests:
        raise ValueError(f"Override key '{key}' is reserved by the explicit parser.")

    top_level_key = path_segments[0]
    if top_level_key in reserved_top_level_keys:
        raise ValueError(f"Override key '{key}' targets reserved top-level key '{top_level_key}'.")

    for segment in path_segments:
        if "\x00" in segment or "/" in segment or "\\" in segment:
            raise ValueError(f"Invalid override key '{key}'.")
        if not _VALID_SEGMENT_RE.fullmatch(segment):
            raise ValueError(f"Invalid override key '{key}'.")

    return path_segments


def _try_generic_autocast(raw_value: str):
    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        normalized = raw_value.lower()
        if normalized in _GENERIC_BOOL_LITERALS:
            return _GENERIC_BOOL_LITERALS[normalized]
        return raw_value


def _coerce_override_value(key: str, raw_value: str, existing_value):
    if existing_value is None:
        return _try_generic_autocast(raw_value)
    if isinstance(existing_value, bool):
        try:
            return str2bool(raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid boolean override for '{key}' with value '{raw_value}'.") from exc
    if isinstance(existing_value, int) and not isinstance(existing_value, bool):
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid int override for '{key}' with value '{raw_value}'.") from exc
    if isinstance(existing_value, float):
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid float override for '{key}' with value '{raw_value}'.") from exc
    if isinstance(existing_value, str):
        return raw_value
    raise ValueError(f"Unsupported override target '{key}' for value '{raw_value}'.")


def _set_dotted_path(args: qt.qDict, path_segments: list[str], raw_value: str | None, is_flag_only: bool):
    current = args
    for segment in path_segments[:-1]:
        next_value = current.get(segment)
        if next_value is None:
            next_value = qt.qDict()
            current[segment] = next_value
        elif not isinstance(next_value, dict):
            dotted_prefix = ".".join(path_segments)
            raise ValueError(f"Invalid override path '{dotted_prefix}'.")

        if not isinstance(next_value, qt.qDict):
            next_value = qt.qDict(next_value)
            current[segment] = next_value
        current = next_value

    leaf_key = path_segments[-1]
    key = ".".join(path_segments)
    leaf_exists = leaf_key in current
    existing_value = current.get(leaf_key) if leaf_exists else None

    if is_flag_only:
        if leaf_exists and not isinstance(existing_value, bool):
            raise ValueError(f"Flag-only override requires boolean target for '{key}'.")
        current[leaf_key] = True
        return

    current[leaf_key] = _coerce_override_value(key, raw_value, existing_value)


def apply_dotted_overrides(
    args: qt.qDict,
    tokens: list[str],
    parser_owned_dests: set[str],
    reserved_top_level_keys: set[str],
):
    overrides = _parse_dotted_override_tokens(tokens)
    for key, raw_value, is_flag_only in overrides:
        path_segments = _validate_override_key(key, parser_owned_dests, reserved_top_level_keys)
        _set_dotted_path(args, path_segments, raw_value, is_flag_only)
    return args


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
    parser_owned_dests = _collect_parser_owned_dests(parser)
    reserved_top_level_keys = _collect_reserved_top_level_keys(parser_owned_dests)
    cmd_args, unknown_tokens = parser.parse_known_args(sys.argv[1:])
    override_tokens = _split_dotted_override_tokens(parser, unknown_tokens)
    cmd_args = qt.qDict.from_namespace(cmd_args)

    args = merge_basic_args(cmd_args)
    apply_dotted_overrides(args, override_tokens, parser_owned_dests, reserved_top_level_keys)
    return args
