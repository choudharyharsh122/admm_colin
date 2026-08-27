"""Adaptive OC sweep in r using perimeter-targeted sample allocation.

Strategy:
1) Evaluate one r sample.
2) Use its observed perimeter p to split the remaining sample budget between
   left/right r-subintervals with the rule:
       n_right = (p_end - p) / (p_end - p_start) * n_remaining
       n_left  = (p - p_start) / (p_end - p_start) * n_remaining
3) Recurse until branch budget is 0 (or 1 sample left, which is evaluated once).

This script reuses the OC solver utilities from oc_r_sweep.py.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from oc_r_sweep import (
    build_exact_filter_template,
    build_graph,
    build_scale,
    build_unitsquaremesh_right_tri,
    compute_tv,
    discretize_control,
    run_topology_optimization,
    save_result,
    solve_state_and_compliance,
)


@dataclass
class SampleResult:
    r: float
    perimeter: float
    compliance_disc: float
    runtime_filter: float
    runtime_algorithm: float
    runtime_total: float


class AdaptivePerimeterSampler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._cache: Dict[float, SampleResult] = {}
        self.samples: List[SampleResult] = []

        # Template/filter precomputation for all future r values.
        t_template_start = time.perf_counter()
        base_coords, base_tris, _, _, _, _ = build_unitsquaremesh_right_tri(args.mesh_dim)
        base_centroids = base_coords[base_tris].mean(axis=1)
        rmin_max = max(args.r_start, args.r_end) / args.mesh_dim
        self.neighbor_template = build_exact_filter_template(base_centroids, rmax=float(rmin_max))
        self.runtime_template_prep = time.perf_counter() - t_template_start

        self.graph = build_graph(args.mesh_dim, 2 * args.mesh_dim)
        self.scale = build_scale(self.graph)

    def _eval_r(self, r: float) -> SampleResult:
        # Avoid duplicate expensive OC runs due to recursion collisions.
        key = float(np.round(r, 12))
        if key in self._cache:
            return self._cache[key]

        t_run_start = time.perf_counter()
        rmin = float(r / self.args.mesh_dim)

        print(f"\n=== Running r={r:.10g} (rmin={rmin:.10g}) ===", flush=True)
        control_cont, coords, tris, tri_type, fixeddofs, iK, jK, F, runtime_filter, runtime_algorithm = run_topology_optimization(
            mesh_dim=self.args.mesh_dim,
            volfrac=self.args.volfrac,
            penal=self.args.penal,
            rmin=rmin,
            eps=self.args.eps,
            f0=self.args.f0,
            tol=self.args.tol,
            maxiter=self.args.maxiter,
            neighbor_template=self.neighbor_template,
        )

        control_disc = discretize_control(control_cont, self.args.volfrac)
        _, compliance_disc = solve_state_and_compliance(
            control_disc,
            coords,
            tris,
            tri_type,
            fixeddofs,
            iK,
            jK,
            F,
            self.args.penal,
            self.args.eps,
        )
        perimeter = compute_tv(control_disc, self.graph, self.scale)

        runtime_total = time.perf_counter() - t_run_start
        runtime_template_filter_algo = (
            self.runtime_template_prep + runtime_filter + runtime_algorithm
        )

        r_folder = f"{r:.4f}"
        output_path = Path(self.args.output_root) / r_folder / f"{self.args.mesh_dim}.h5"
        save_result(
            output_path=output_path,
            control_cont=control_cont,
            control_disc=control_disc,
            compliance_disc=compliance_disc,
            tv_disc=perimeter,
            mesh_dim=self.args.mesh_dim,
            r=r,
            rmin=rmin,
            runtime_filter=runtime_filter,
            runtime_algorithm=runtime_algorithm,
            runtime_total=runtime_total,
            runtime_template_prep=self.runtime_template_prep,
            runtime_template_filter_algo=runtime_template_filter_algo,
        )

        result = SampleResult(
            r=r,
            perimeter=perimeter,
            compliance_disc=float(compliance_disc),
            runtime_filter=float(runtime_filter),
            runtime_algorithm=float(runtime_algorithm),
            runtime_total=float(runtime_total),
        )
        self._cache[key] = result
        self.samples.append(result)

        print(
            f"Saved {output_path} | perimeter={perimeter:.6f} | compliance={compliance_disc:.6f}",
            flush=True,
        )
        return result

    def _split_counts(self, p: float, n_remaining: int) -> Tuple[int, int]:
        ps = self.args.perimeter_start
        pe = self.args.perimeter_end

        p_clamped = min(max(p, ps), pe)
        span = pe - ps
        if span <= 0:
            raise ValueError("Require perimeter_end > perimeter_start")

        left_frac = (pe - p_clamped) / span
        right_frac = (p_clamped - ps) / span

        left = int(round(left_frac * n_remaining))
        left = max(0, min(left, n_remaining))
        right = n_remaining - left
        return left, right

    def _recurse(self, r_lo: float, r_hi: float, n: int, r_seed: float) -> None:
        if n <= 0:
            return

        # If only one sample is requested on this branch, run once and return.
        if n == 1:
            r_eval = min(max(r_seed, min(r_lo, r_hi)), max(r_lo, r_hi))
            self._eval_r(r_eval)
            return

        if abs(r_hi - r_lo) < self.args.min_r_gap:
            # Interval collapsed numerically: do one eval only.
            r_eval = 0.5 * (r_lo + r_hi)
            self._eval_r(r_eval)
            return

        r_eval = min(max(r_seed, min(r_lo, r_hi)), max(r_lo, r_hi))
        res = self._eval_r(r_eval)

        n_remaining = n - 1
        left_n, right_n = self._split_counts(res.perimeter, n_remaining)

        print(
            f"Split @ r={r_eval:.10g}, p={res.perimeter:.6f}: left={left_n}, right={right_n}",
            flush=True,
        )

        if left_n > 0:
            self._recurse(r_lo, r_eval, left_n, 0.5 * (r_lo + r_eval))
        if right_n > 0:
            self._recurse(r_eval, r_hi, right_n, 0.5 * (r_eval + r_hi))

    def run(self) -> None:
        r_lo = float(self.args.r_start)
        r_hi = float(self.args.r_end)
        if r_lo == r_hi:
            raise ValueError("r_start and r_end must differ")

        if self.args.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if self.args.r_init is None:
            r0 = 0.5 * (r_lo + r_hi)
        else:
            r0 = float(self.args.r_init)

        print("=== Adaptive perimeter-guided OC sampling ===", flush=True)
        print(
            f"r range: [{min(r_lo, r_hi):.6g}, {max(r_lo, r_hi):.6g}], "
            f"num_samples={self.args.num_samples}, r_init={r0:.6g}",
            flush=True,
        )
        print(
            f"target perimeter range: [{self.args.perimeter_start:.6g}, {self.args.perimeter_end:.6g}]",
            flush=True,
        )

        self._recurse(r_lo, r_hi, int(self.args.num_samples), r0)

        unique_n = len({float(np.round(s.r, 12)) for s in self.samples})
        print(
            f"=== Done. Evaluations attempted: {len(self.samples)}, unique r: {unique_n} ===",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive OC sweep over r using perimeter-based recursive sample allocation."
        )
    )

    parser.add_argument("--mesh-dim", type=int, default=64)
    parser.add_argument("--r-start", type=float, required=True)
    parser.add_argument("--r-end", type=float, required=True)
    parser.add_argument("--r-init", type=float, default=None)
    parser.add_argument("--num-samples", type=int, required=True)

    parser.add_argument("--perimeter-start", type=float, required=True)
    parser.add_argument("--perimeter-end", type=float, required=True)

    parser.add_argument("--volfrac", type=float, default=0.4)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--f0", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--min-r-gap", type=float, default=1e-8)
    parser.add_argument("--output-root", type=str, default="OC_results_adaptive")

    args = parser.parse_args()

    if args.perimeter_end <= args.perimeter_start:
        raise ValueError("Require --perimeter-end > --perimeter-start")

    sampler = AdaptivePerimeterSampler(args)
    sampler.run()


if __name__ == "__main__":
    main()
