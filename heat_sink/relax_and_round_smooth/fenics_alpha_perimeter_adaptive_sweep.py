"""Adaptive alpha sweep for the smooth relax-and-round Fenics model.

This script mirrors the adaptive perimeter-guided sampling strategy used for
ADMM and OC sweeps, but targets the Fenics relax-and-round solver.

At each step it evaluates one alpha, reads the resulting discrete TV
(perimeter-like metric) and compliance from the generated HDF5 output, and
uses that perimeter to decide how to split the remaining sample budget between
left and right alpha subintervals.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class SampleResult:
    alpha: float
    perimeter: float
    compliance: float
    file_path: Path


class AdaptiveAlphaSampler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_script = Path(__file__).resolve().with_name("fenics_model.py")
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = self.model_script.parent / output_root
        self.output_root = output_root.resolve()
        self.cache: Dict[float, SampleResult] = {}
        self.samples: List[SampleResult] = []
        self.run_idx = 0

    def _alpha_folder_name(self, alpha: float) -> str:
        return str(float(np.round(alpha, self.args.alpha_round_digits)))

    def _run_file_path(self, alpha: float) -> Path:
        alpha_dir = self._alpha_folder_name(alpha)
        return self.output_root / alpha_dir / f"{self.args.mesh_size}.h5"

    def _read_metrics(self, h5_path: Path) -> Tuple[float, float]:
        with h5py.File(h5_path, "r") as h5f:
            if "summary" not in h5f:
                raise KeyError(f"Missing summary group in {h5_path}")
            summary = h5f["summary"]
            if "disc_TV" not in summary or "disc_compliance" not in summary:
                raise KeyError(
                    f"Expected datasets disc_TV/disc_compliance in {h5_path}"
                )
            perimeter = float(np.asarray(summary["disc_TV"][()], dtype=float).reshape(-1)[-1])
            compliance = float(np.asarray(summary["disc_compliance"][()], dtype=float).reshape(-1)[-1])
            return perimeter, compliance

    def _evaluate_alpha(self, alpha_raw: float) -> SampleResult:
        alpha = float(np.round(alpha_raw, self.args.alpha_round_digits))
        key = float(np.round(alpha, 12))
        if key in self.cache:
            return self.cache[key]

        h5_path = self._run_file_path(alpha)

        if h5_path.exists() and self.args.reuse_existing:
            perimeter, compliance = self._read_metrics(h5_path)
            print(
                f"Reused alpha={alpha:.10g} | perimeter={perimeter:.6f} | compliance={compliance:.6f}",
                flush=True,
            )
        else:
            if h5_path.exists() and not self.args.reuse_existing:
                h5_path.unlink()

            print(f"\n=== Running alpha={alpha:.10g} (mesh={self.args.mesh_size}) ===", flush=True)
            command = [
                sys.executable,
                str(self.model_script),
                "--alpha",
                str(alpha),
                "--mesh-list",
                str(self.args.mesh_size),
                "--source_strength",
                str(self.args.source_strength),
                "--vol_frac",
                str(self.args.vol_frac),
            ]
            subprocess.run(command, check=True, cwd=str(self.model_script.parent))
            self.run_idx += 1

            if not h5_path.exists():
                raise FileNotFoundError(f"Expected output not found after run: {h5_path}")

            perimeter, compliance = self._read_metrics(h5_path)
            print(
                f"Saved alpha={alpha:.10g} | perimeter={perimeter:.6f} | compliance={compliance:.6f}",
                flush=True,
            )

        res = SampleResult(alpha=alpha, perimeter=perimeter, compliance=compliance, file_path=h5_path)
        self.cache[key] = res
        self.samples.append(res)
        return res

    def _split_counts_inverse(self, p1: float, n_remaining: int) -> Tuple[int, int]:
        ps = float(self.args.perimeter_start)
        pe = float(self.args.perimeter_end)
        span = pe - ps
        if span <= 0:
            raise ValueError("Require perimeter_end > perimeter_start")

        p_clamped = min(max(p1, ps), pe)

        left_frac = (pe - p_clamped) / span
        right_frac = (p_clamped - ps) / span

        left_n = int(round(left_frac * n_remaining))
        left_n = max(0, min(left_n, n_remaining))
        right_n = n_remaining - left_n

        _ = right_frac
        return left_n, right_n

    def _recurse(self, a_lo: float, a_hi: float, n: int, a_seed: float) -> None:
        if n <= 0:
            return

        a_min = min(a_lo, a_hi)
        a_max = max(a_lo, a_hi)

        if n == 1:
            a_eval = min(max(a_seed, a_min), a_max)
            self._evaluate_alpha(a_eval)
            return

        if abs(a_hi - a_lo) < self.args.min_alpha_gap:
            a_eval = 0.5 * (a_lo + a_hi)
            self._evaluate_alpha(a_eval)
            return

        a_eval = min(max(a_seed, a_min), a_max)
        res = self._evaluate_alpha(a_eval)

        n_remaining = n - 1
        left_n, right_n = self._split_counts_inverse(res.perimeter, n_remaining)

        print(
            f"Split @ alpha={a_eval:.10g}, p={res.perimeter:.6f}: left={left_n}, right={right_n}",
            flush=True,
        )

        if left_n > 0:
            self._recurse(a_lo, a_eval, left_n, 0.5 * (a_lo + a_eval))
        if right_n > 0:
            self._recurse(a_eval, a_hi, right_n, 0.5 * (a_eval + a_hi))

    def run(self) -> None:
        a_lo = float(self.args.alpha_start)
        a_hi = float(self.args.alpha_end)

        if a_lo == a_hi:
            raise ValueError("alpha_start and alpha_end must differ")
        if self.args.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if self.args.alpha_init is None:
            a0 = 0.5 * (a_lo + a_hi)
        else:
            a0 = float(self.args.alpha_init)

        print("=== Adaptive perimeter-guided Fenics alpha sampling ===", flush=True)
        print(
            f"alpha range: [{min(a_lo, a_hi):.6g}, {max(a_lo, a_hi):.6g}], "
            f"num_samples={self.args.num_samples}, alpha_init={a0:.6g}",
            flush=True,
        )
        print(
            f"target perimeter range: [{self.args.perimeter_start:.6g}, {self.args.perimeter_end:.6g}]",
            flush=True,
        )
        print(f"output root: {self.output_root}", flush=True)

        self._recurse(a_lo, a_hi, int(self.args.num_samples), a0)

        unique_n = len({float(np.round(s.alpha, 12)) for s in self.samples})
        print(
            f"=== Done. Evaluations attempted: {len(self.samples)}, unique alpha: {unique_n} ===",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive smooth relax-and-round alpha sweep using perimeter-guided allocation."
    )

    parser.add_argument("--alpha-start", type=float, required=True)
    parser.add_argument("--alpha-end", type=float, required=True)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--num-samples", type=int, required=True)

    parser.add_argument("--perimeter-start", type=float, required=True)
    parser.add_argument("--perimeter-end", type=float, required=True)

    parser.add_argument("--mesh-size", type=int, default=64)
    parser.add_argument("--source_strength", type=float, default=1.0)
    parser.add_argument("--vol_frac", type=float, default=0.4)
    parser.add_argument("--min-alpha-gap", type=float, default=1e-8)
    parser.add_argument("--alpha-round-digits", type=int, default=12)
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing HDF5 result for an alpha if present (default: true).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="fenics_model_tri_1e-06",
        help="Root directory for the generated HDF5 files.",
    )

    args = parser.parse_args()

    if args.perimeter_end <= args.perimeter_start:
        raise ValueError("Require --perimeter-end > --perimeter-start")
    if args.alpha_start <= 0 or args.alpha_end <= 0:
        raise ValueError("alpha_start and alpha_end must be > 0")

    sampler = AdaptiveAlphaSampler(args)
    sampler.run()


if __name__ == "__main__":
    main()
