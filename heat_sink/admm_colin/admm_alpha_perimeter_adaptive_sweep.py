"""Adaptive ADMM alpha sweep using perimeter-guided sample allocation.

This script mirrors the adaptive OC sampling idea, but for ADMM over alpha.
Because perimeter is inversely related to alpha here, the split uses:
  left_count  ~ (p_end - p1)
  right_count ~ (p1 - p_start)
for a sample with observed perimeter p1.

The script reuses admm_run.py entry points and reads perimeter/compliance
from generated HDF5 run files.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from admm_run import CONFIG_FILE, load_config, run_trial


def _safe_last(ds: h5py.Dataset) -> float:
    arr = np.asarray(ds[()], dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Dataset {ds.name} is empty")
    return float(arr[-1])


def _seed_groups(h5f: h5py.File) -> List[str]:
    names = [k for k in h5f.keys() if k.startswith("seed_")]
    if not names:
        raise ValueError("No seed_* groups found in run file")
    return names


def _median_seed_name(h5f: h5py.File) -> str:
    names = _seed_groups(h5f)
    objs = np.asarray([_safe_last(h5f[n]["objective_list"]) for n in names], dtype=float)
    order = np.argsort(objs)
    return names[int(order[len(order) // 2])]


def _infer_base_dir(params: argparse.Namespace) -> str:
    penalty_update_method = str(params.PENALTY_UPDATE_METHOD).strip().lower()
    backend = str(params.BACKEND).strip().lower()

    if penalty_update_method == "none":
        base_dir = f"run_data_admm_{backend}"
    else:
        base_dir = f"run_data_admm_{penalty_update_method}_{backend}"

    if bool(params.USE_MIP):
        base_dir += "_mip"
    return base_dir


def _alpha_folder_name(alpha: float, round_digits: int) -> str:
    # Match admm_run behavior (str(float)) while controlling excessive float noise.
    return str(float(np.round(alpha, round_digits)))


@dataclass
class SampleResult:
    alpha: float
    perimeter: float
    compliance: float
    file_path: Path


class AdaptiveAlphaSampler:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        cfg = load_config(args.config)
        self.base_params = argparse.Namespace(**cfg)
        self.mesh_size = int(args.mesh_size if args.mesh_size is not None else self.base_params.MESH_SIZE)

        self.base_dir = Path(_infer_base_dir(self.base_params))
        self.cache: Dict[float, SampleResult] = {}
        self.samples: List[SampleResult] = []
        self.run_idx = 0

    def _run_file_path(self, alpha: float) -> Path:
        alpha_dir = _alpha_folder_name(alpha, self.args.alpha_round_digits)
        return self.base_dir / alpha_dir / f"{self.mesh_size}.h5"

    def _read_metrics(self, h5_path: Path) -> Tuple[float, float]:
        with h5py.File(h5_path, "r") as h5f:
            med = _median_seed_name(h5f)
            grp = h5f[med]

            if "tv_disc_list" not in grp or "compliance_disc_list" not in grp:
                raise KeyError(
                    f"Expected datasets tv_disc_list/compliance_disc_list in {med} of {h5_path}"
                )

            perimeter = _safe_last(grp["tv_disc_list"])
            compliance = _safe_last(grp["compliance_disc_list"])
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

            run_args = copy.deepcopy(self.base_params)
            run_args.ALPHA = alpha
            run_args.MESH_SIZE = self.mesh_size

            print(f"\n=== Running alpha={alpha:.10g} (mesh={self.mesh_size}) ===", flush=True)
            run_trial(dim=self.mesh_size, idx=self.run_idx, params=run_args)
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

        # Inverse relation alpha -> perimeter:
        # left(alpha_lo..alpha_mid) gets weight (pe - p1)
        # right(alpha_mid..alpha_hi) gets weight (p1 - ps)
        left_frac = (pe - p_clamped) / span
        right_frac = (p_clamped - ps) / span

        left_n = int(round(left_frac * n_remaining))
        left_n = max(0, min(left_n, n_remaining))
        right_n = n_remaining - left_n

        # Keep fractions explicit for potential debugging consistency checks.
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

        print("=== Adaptive perimeter-guided ADMM alpha sampling ===", flush=True)
        print(
            f"alpha range: [{min(a_lo, a_hi):.6g}, {max(a_lo, a_hi):.6g}], "
            f"num_samples={self.args.num_samples}, alpha_init={a0:.6g}",
            flush=True,
        )
        print(
            f"target perimeter range: [{self.args.perimeter_start:.6g}, {self.args.perimeter_end:.6g}]",
            flush=True,
        )
        print(f"output root: {self.base_dir}", flush=True)

        self._recurse(a_lo, a_hi, int(self.args.num_samples), a0)

        unique_n = len({float(np.round(s.alpha, 12)) for s in self.samples})
        print(
            f"=== Done. Evaluations attempted: {len(self.samples)}, unique alpha: {unique_n} ===",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive ADMM alpha sweep using inverse perimeter-guided sample allocation."
        )
    )

    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path to admm_config.cfg")
    parser.add_argument("--mesh-size", type=int, default=None, help="Override MESH_SIZE from config")

    parser.add_argument("--alpha-start", type=float, required=True)
    parser.add_argument("--alpha-end", type=float, required=True)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--num-samples", type=int, required=True)

    parser.add_argument("--perimeter-start", type=float, required=True)
    parser.add_argument("--perimeter-end", type=float, required=True)

    parser.add_argument("--min-alpha-gap", type=float, default=1e-8)
    parser.add_argument("--alpha-round-digits", type=int, default=12)
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing run file for an alpha if present (default: true).",
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
