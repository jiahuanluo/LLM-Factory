"""Shared argument parsing logic for YAML, JSON, and CLI arguments."""

import os
import re
import sys

import yaml

_NUM_RE = re.compile(r'^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')


def _coerce_numeric(config):
    """Convert CLI string values that look like numbers to int/float."""
    for key, value in config.items():
        if isinstance(value, str) and _NUM_RE.match(value):
            try:
                float_val = float(value)
                if '.' not in value and 'e' not in value.lower():
                    config[key] = int(float_val)
                else:
                    config[key] = float_val
            except ValueError:
                pass
    return config


def _expand_paths(config):
    """Expand ~ and environment variables in string values that look like paths."""
    for key, value in config.items():
        if isinstance(value, str) and ("~" in value or "$" in value or value.startswith("./") or value.startswith("../")):
            config[key] = os.path.expanduser(os.path.expandvars(value))
    return config


def _merge_cli_overrides(config, skip_indices):
    """Scan argv for --key / --key=value and merge into config (later wins).

    Supports CLI flags on either side of the YAML file so that launcher-injected
    args like `--local_rank=0` (from `deepspeed` launcher) still reach the
    TrainingArguments dataclass instead of being silently dropped.
    """
    argv = sys.argv
    i = 1
    while i < len(argv):
        if i in skip_indices:
            i += 1
            continue
        arg = argv[i]
        if arg.startswith("--"):
            body = arg[2:]
            if "=" in body:
                key, value = body.split("=", 1)
                config[key] = value
                i += 1
            else:
                key = body
                next_idx = i + 1
                if (next_idx < len(argv)
                        and next_idx not in skip_indices
                        and not argv[next_idx].startswith("--")):
                    config[key] = argv[next_idx]
                    i += 2
                else:
                    config[key] = True
                    i += 1
        else:
            i += 1
    return config


def read_args(parser):
    """Parse arguments from YAML, JSON, or CLI.

    YAML file may appear anywhere in argv (not just argv[1]) so launcher-
    injected flags don't break detection. Example: `deepspeed script.py
    --local_rank=0 configs/x.yaml --epochs 5` parses correctly.
    """
    yaml_idx = next(
        (i for i, a in enumerate(sys.argv[1:], 1) if a.endswith((".yaml", ".yml"))),
        None,
    )
    if yaml_idx is not None:
        with open(sys.argv[yaml_idx]) as f:
            config = yaml.safe_load(f) or {}
        config = _merge_cli_overrides(config, skip_indices={0, yaml_idx})
        config = _expand_paths(config)
        config = _coerce_numeric(config)
        return parser.parse_dict(config)
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()
