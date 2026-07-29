"""Batch runner for smooth relax-and-round alpha/mesh sweeps.

This script prepares alpha and mesh sweeps, then invokes fenics_model.py.
It supports both explicit lists and generated ranges for alpha values.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np


def _parse_float_list(raw: str) -> List[float]:
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    if not tokens:
        raise ValueError("Empty alpha list. Pass values like '0.01,0.05,0.1'.")
    return [float(t) for t in tokens]


def _parse_int_list(raw: str) -> List[int]:
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    if not tokens:
        raise ValueError("Empty mesh list. Pass values like '32,64,128'.")
    return [int(t) for t in tokens]


def _format_float_list(values: List[float]) -> str:
    # Keep command-line values compact while preserving precision.
    return ",".join(f"{v:.12g}" for v in values)


def _build_alpha_values(args: argparse.Namespace) -> List[float]:
    if args.alpha_list is not None:
        values = _parse_float_list(args.alpha_list)
    else:
        if args.alpha_range is None:
            raise ValueError("Provide either --alpha-list or --alpha-range START STOP COUNT.")

        start, stop, count = args.alpha_range
        if count <= 0:
            raise ValueError("alpha COUNT must be a positive integer.")

        if args.alpha_scale == "linear":
            values = np.linspace(start, stop, int(count)).tolist()
        else:
            if start <= 0 or stop <= 0:
                raise ValueError("Log alpha range requires START and STOP > 0.")
            values = np.logspace(np.log10(start), np.log10(stop), int(count)).tolist()

    if any(v <= 0 for v in values):
        raise ValueError("All alpha values must be > 0.")

    return values


def _build_mesh_values(args: argparse.Namespace) -> List[int]:
    if args.mesh_list is not None:
        meshes = _parse_int_list(args.mesh_list)
    elif args.mesh_size is not None:
        meshes = [int(args.mesh_size)]
    else:
        raise ValueError("Provide either --mesh-size or --mesh-list.")

    if any(m <= 0 for m in meshes):
        raise ValueError("All mesh sizes must be positive integers.")

    return meshes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run alpha sweep for relax_and_round_smooth/fenics_model.py"
    )

    alpha_group = parser.add_mutually_exclusive_group(required=True)
    alpha_group.add_argument(
        "--alpha-list",
        type=str,
        help="Comma/space-separated alphas. Example: '0.01,0.05,0.1'",
    )
    alpha_group.add_argument(
        "--alpha-range",
        nargs=3,
        metavar=("START", "STOP", "COUNT"),
        type=float,
        help="Generate alpha values between START and STOP (inclusive) with COUNT points.",
    )

    parser.add_argument(
        "--alpha-scale",
        choices=["linear", "log"],
        default="linear",
        help="Spacing for --alpha-range. Default: linear.",
    )

    mesh_group = parser.add_mutually_exclusive_group(required=True)
    mesh_group.add_argument("--mesh-size", type=int, help="Single mesh size. Example: 64")
    mesh_group.add_argument(
        "--mesh-list",
        type=str,
        help="Comma/space-separated mesh sizes. Example: '32,64,128'",
    )

    parser.add_argument("--source_strength", type=float, default=1.0)
    parser.add_argument("--vol_frac", type=float, default=0.4)

    args = parser.parse_args()

    alpha_values = _build_alpha_values(args)
    mesh_values = _build_mesh_values(args)

    model_script = Path(__file__).resolve().parent / "fenics_model.py"
    command = [
        sys.executable,
        str(model_script),
        "--alpha",
        _format_float_list(alpha_values),
        "--mesh-list",
        ",".join(str(m) for m in mesh_values),
        "--source_strength",
        str(args.source_strength),
        "--vol_frac",
        str(args.vol_frac),
    ]

    print("=== Running smooth relax-and-round sweep ===", flush=True)
    print(f"alpha values: {alpha_values}", flush=True)
    print(f"mesh sizes: {mesh_values}", flush=True)
    print(f"command: {' '.join(command)}", flush=True)

    subprocess.run(command, check=True)

    print("=== Sweep complete ===", flush=True)


if __name__ == "__main__":
    main()
