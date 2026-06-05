"""Shared argument parsing logic for YAML, JSON, and CLI arguments."""

import os
import re
import sys
from datetime import datetime

import yaml

_NUM_RE = re.compile(r'^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
_VAR_RE = re.compile(r'\$\{([^}]+)\}')


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


def _expand_vars(config):
    """Expand ${var} references in string values, supporting cross-parameter refs and ${timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    max_iterations = 5  # prevent infinite loops from circular refs
    for _ in range(max_iterations):
        changed = False
        for key, value in config.items():
            if isinstance(value, str) and _VAR_RE.search(value):
                def replacer(m):
                    nonlocal changed
                    ref = m.group(1)
                    if ref == "timestamp":
                        return timestamp
                    if ref in config and config[ref] is not None:
                        changed = True
                        return str(config[ref])
                    return m.group(0)  # leave unresolved refs as-is
                config[key] = _VAR_RE.sub(replacer, value)
        if not changed:
            break
    return config


def _expand_paths(config):
    """Expand ~ and environment variables in string values that look like paths."""
    for key, value in config.items():
        if isinstance(value, str) and ("~" in value or "$" in value or value.startswith("./") or value.startswith("../")):
            config[key] = os.path.expanduser(os.path.expandvars(value))
    return config


def read_args(parser):
    """Parse arguments from YAML, JSON, or CLI."""
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f) or {}
        # CLI overrides: --key value pairs after the YAML path
        if len(sys.argv) > 2:
            cli_args = sys.argv[2:]
            i = 0
            while i < len(cli_args):
                if cli_args[i].startswith("--"):
                    key = cli_args[i][2:]
                    if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                        config[key] = cli_args[i + 1]
                        i += 2
                    else:
                        config[key] = True
                        i += 1
                else:
                    i += 1
        config = _expand_vars(config)
        config = _expand_paths(config)
        config = _coerce_numeric(config)
        return parser.parse_dict(config)
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()
