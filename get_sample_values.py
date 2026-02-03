#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone script to generate sample values for PyTorch operators using FACTO SpecDB.

Usage:
    python get_sample_values.py add.Tensor sub.Tensor mul.Tensor
    python get_sample_values.py --list  # List all available operators
"""

import argparse
import json
from typing import Any, Dict, List

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB


def tensor_to_dict(t: torch.Tensor) -> Dict[str, Any]:
    """Convert a tensor to a JSON-serializable dict."""
    return {
        "type": "Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "values": t.tolist()
    }


def value_to_json(val: Any) -> Any:
    """Convert a value to JSON-serializable format."""
    if isinstance(val, torch.Tensor):
        return tensor_to_dict(val)
    elif isinstance(val, list):
        return [value_to_json(item) for item in val]
    elif isinstance(val, (int, float, bool, str, type(None))):
        return val
    elif isinstance(val, torch.dtype):
        return str(val)
    else:
        return str(val)


def get_sample_values(op_name: str, max_samples: int = 1) -> Dict[str, Any]:
    """Generate and return sample values for an operator as a dict."""
    if op_name not in SpecDictDB:
        return {"error": f"Operator '{op_name}' not found in SpecDB"}

    spec = SpecDictDB[op_name]
    generator = ArgumentTupleGenerator(spec)

    samples = []
    for i, (posargs, inkwargs, outargs) in enumerate(generator.gen()):
        if i >= max_samples:
            break

        sample = {
            "posargs": [value_to_json(arg) for arg in posargs],
            "kwargs": {k: value_to_json(v) for k, v in inkwargs.items()},
        }
        if outargs:
            sample["outargs"] = {k: value_to_json(v) for k, v in outargs.items()}

        samples.append(sample)

    return {"operator": op_name, "samples": samples}


def list_operators() -> List[str]:
    """List all available operators in SpecDB."""
    return sorted(SpecDictDB.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample values for PyTorch operators using FACTO SpecDB"
    )
    parser.add_argument(
        "operators",
        nargs="*",
        help="List of operator names (e.g., add.Tensor sub.Tensor)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available operators"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=1,
        help="Maximum number of samples per operator (default: 1)"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2, use 0 for compact)"
    )

    args = parser.parse_args()

    if args.list:
        result = {"operators": list_operators()}
        print(json.dumps(result, indent=args.indent if args.indent > 0 else None))
        return

    if not args.operators:
        parser.print_help()
        return

    results = []
    for op_name in args.operators:
        results.append(get_sample_values(op_name, max_samples=args.max_samples))

    print(json.dumps(results, indent=args.indent if args.indent > 0 else None))


if __name__ == "__main__":
    main()
