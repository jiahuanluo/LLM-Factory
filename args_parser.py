"""Shared argument parsing logic for YAML, JSON, and CLI arguments."""

import os
import sys

import yaml


def _expand_paths(config):
    """Expand ~ and environment variables in string values that look like paths."""
    for key, value in config.items():
        if isinstance(value, str) and ("~" in value or value.startswith("./") or value.startswith("../")):
            config[key] = os.path.expanduser(os.path.expandvars(value))
    return config


def _parse_numeric(config):
    """Convert YAML strings that look like scientific notation (e.g., 2e-5) to float."""
    for key, value in config.items():
        if isinstance(value, str):
            try:
                config[key] = float(value)
            except ValueError:
                pass
    return config


def read_args(parser):
    """Parse arguments from YAML, JSON, or CLI."""
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f) or {}
        config = _expand_paths(config)
        config = _parse_numeric(config)
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
        return parser.parse_dict(config)
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()
