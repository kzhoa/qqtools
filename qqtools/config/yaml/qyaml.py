import argparse
import os
import warnings
from pathlib import Path

import yaml

from ...qdict import qDict
from ...torch.qgpu import parse_device

from .qInheritLoader import InheritLoader


def parse_none(cfg):
    """add support for `none` parse to yaml"""
    for k, v in cfg.items():
        if v in ["none", "None"]:
            cfg[k] = None
        if isinstance(v, dict):
            # recursive
            v = parse_none(v)
            cfg[k] = v
    return cfg


def _str2science_number(v):
    if isinstance(v, str) and "e" in v:
        try:
            num = float(v)
            return num
        except Exception:
            pass
    return v


def parse_science(cfg):
    """add science indicator support to yaml
    By default, yaml treat '5e-4' as string, but not float.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            cfg[k] = _str2science_number(v)
        elif isinstance(v, dict):
            # recursive
            v = parse_science(v)
            cfg[k] = v
        elif isinstance(v, list):
            v = [_str2science_number(u) for u in v]
            cfg[k] = v
    return cfg


def _str2userhome(v):
    if isinstance(v, str) and "$USER_HOME" in v:
        try:
            USER_HOME = os.environ["HOME"]
            v_ = v.replace("$USER_HOME", USER_HOME)
            return v_
        except Exception:
            pass
    return v


def parse_userhome(cfg):
    """parse_userhome"""
    for k, v in cfg.items():
        if isinstance(v, str) and "$USER_HOME" in v:
            cfg[k] = _str2userhome(v)
        elif isinstance(v, dict):
            # recursive
            v = parse_userhome(v)
            cfg[k] = v
        elif isinstance(v, list):
            v = [_str2userhome(u) for u in v]
            cfg[k] = v
    return cfg


def save_yaml(cfg, path):
    """alias"""
    dump_yaml(cfg, path)


def dump_yaml(cfg, path, sort_keys=False, verbose=False):
    if isinstance(cfg, qDict):
        cfg = cfg.to_dict()  # or will cause no constructor ERROR
    elif isinstance(cfg, dict):
        cfg = dict(cfg)
    elif isinstance(cfg, argparse.Namespace):
        cfg = dict(cfg.__dict__)
    yaml.dump(cfg, open(path, "w"), sort_keys=sort_keys)
    if verbose:
        print(f"yaml dump to : {path} .")


def load_yaml(path, inherit=True, ignore_keys=[]) -> qDict:
    """load_yaml with qinheritance support"""
    if (path is None) or (not Path(path).exists()):
        warnings.warn(f"file:{path} not exists")
        return qDict()

    loader = InheritLoader if inherit else yaml.UnsafeLoader
    cfg = yaml.load(open(path, "r"), Loader=loader)
    cfg = qDict(cfg)
    for k in ignore_keys:
        if k in cfg:
            del cfg[k]

    cfg = parse_science(cfg)

    cfg = parse_none(cfg)

    cfg = parse_userhome(cfg)

    if "device" in cfg:
        cfg.device = parse_device(cfg.device)
    return cfg
