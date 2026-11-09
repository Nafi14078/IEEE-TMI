import json
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path, overrides=()):
    """Load a YAML config and apply `a.b.c=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or ():
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def run_name(cfg) -> str:
    name = "{m}_{a}_lab{f}_s{s}".format(
        m=cfg["method"], a=cfg["student"]["arch"],
        f=cfg["data"]["label_fraction"], s=cfg["train"]["seed"])
    if cfg.get("tag"):
        name = f"{cfg['tag']}_{name}"
    return name


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
