import argparse
import configparser
import math
import pickle
import time
from typing import Dict, Tuple
import os
import h5py
import random


import numpy as np
import networkx as nx  
from cyipopt import Problem  

# Local imports
from design_variables import DesignVariables

from admm_methods.admm_colin.subproblem1_solver import (
    Subproblem1Solver,
    MaterialInterpolation,
    generate_unit_square_mesh,
    build_dirichlet_bc_from_config,
)
from subproblem2_solver import Subproblem2Solver


"""
When using the mergesplit backend, we run multiple random seeds and save all results in an HDF5 file.
We then select the best and median seeds based on final objective values.
"""
def find_best_and_median_seeds(h5f):
    """
    Scan all seed_* groups in h5 file and find:
      - best seed (min final objective)
      - median seed (median final objective)
    Uses all seeds; no infeasibility-threshold filtering.
    """
    seed_names = [k for k in h5f.keys() if k.startswith("seed_")]

    objs = []
    names = []

    for name in seed_names:
        grp = h5f[name]
        objs.append(grp["objective_list"][-1])
        names.append(name)

    if len(names) == 0:
        raise ValueError("No seed_* groups found in HDF5 file.")

    objs = np.asarray(objs, float)

    # best among all seeds
    best_pos = np.argmin(objs)
    best_seed = names[best_pos]

    # median among all seeds
    order = np.argsort(objs)
    median_pos = order[len(order) // 2]
    median_seed = names[median_pos]

    return best_seed, median_seed


def save_data(
    h5_path,
    seed_idx,
    num_seeds,
    dim,
    idx,
    backend,
    alpha,
    rho_init,
    rho_final,
    final_iter,
    obj_list,
    tv_list,
    compliance_list,
    obj_disc_list,
    tv_disc_list,
    compliance_disc_list,
    runtime1_list,
    runtime2_list,
    infeas_list,
    rho_track_list,
    a_list,
    b_list,
    a_disc_list,
    u_list,
    lam_list,
    track_oc_convergence,
    oc_track_per_admm_iter,
):
    with h5py.File(h5_path, "a") as h5f:  # 'a' = append mode
        seed_group = h5f.create_group(f"seed_{seed_idx}")

        # --- Metadata ---
        seed_group.attrs["dim"] = dim
        seed_group.attrs["idx"] = idx
        seed_group.attrs["backend"] = backend
        seed_group.attrs["final_iter"] = final_iter
        seed_group.attrs["alpha"] = alpha
        seed_group.attrs["rho_init"] = rho_init
        seed_group.attrs["rho_final"] = rho_final

        # --- Scalar & simple lists ---
        seed_group.create_dataset("objective_list", data=np.array(obj_list, dtype=np.float64))
        seed_group.create_dataset("tv_list", data=np.array(tv_list, dtype=np.float64))
        seed_group.create_dataset("compliance_list", data=np.array(compliance_list, dtype=np.float64))
        seed_group.create_dataset("objective_disc_list", data=np.array(obj_disc_list, dtype=np.float64))
        seed_group.create_dataset("tv_disc_list", data=np.array(tv_disc_list, dtype=np.float64))
        seed_group.create_dataset("compliance_disc_list", data=np.array(compliance_disc_list, dtype=np.float64))
        seed_group.create_dataset("runtime_sub1_list", data=np.array(runtime1_list, dtype=np.float64))
        seed_group.create_dataset("runtime_sub2_list", data=np.array(runtime2_list, dtype=np.float64))
        seed_group.create_dataset("infeas_list", data=np.array(infeas_list, dtype=np.float64))
        seed_group.create_dataset("rho_list", data=np.array(rho_track_list, dtype=np.float64))

        # --- Iterates ---
        grp_iters = seed_group.create_group("iters")
        grp_iters.create_dataset("a_list", data=np.vstack(a_list))
        grp_iters.create_dataset("b_list", data=np.vstack(b_list))
        grp_iters.create_dataset("a_disc_list", data=np.vstack(a_disc_list))
        grp_iters.create_dataset("u_list", data=np.vstack(u_list))
        grp_iters.create_dataset("lambda_list", data=np.vstack(lam_list))

        # OC convergence data (saved only when flag is enabled)
        if track_oc_convergence and len(oc_track_per_admm_iter) > 0:
            grp_oc = seed_group.create_group("oc_subproblem1_tracking")
            for admm_iter_idx, track_data in enumerate(oc_track_per_admm_iter):
                grp_iter = grp_oc.create_group(f"admm_iter_{admm_iter_idx}")
                grp_iter.create_dataset("F_list", data=np.asarray(track_data["F_list"], dtype=np.float64))
                grp_iter.create_dataset("grad_F_list", data=np.asarray(track_data["grad_F_list"], dtype=np.float64))
                grp_iter.create_dataset("grad_F_norm_list", data=np.asarray(track_data["grad_F_norm_list"], dtype=np.float64))
        summary = h5f.require_group("summary")

        def save(name, data):
            if name in summary:
                del summary[name]
            summary.create_dataset(name, data=data)

        # For gurobi (single seed), use that seed for both best and median
        if num_seeds == 1:
            seed_name = "seed_0"
            grp = h5f[seed_name]

            save("best_seed_name", np.bytes_(seed_name))
            save("best_objective", grp["objective_list"][-1])
            save("best_tv", grp["tv_list"][-1])
            save("best_compliance", grp["compliance_list"][-1])

            save("median_seed_name", np.bytes_(seed_name))
            save("median_objective", grp["objective_list"][-1])
            save("median_tv", grp["tv_list"][-1])
            save("median_compliance", grp["compliance_list"][-1])

            best_iters = summary.require_group("best_iters")
            for k in list(best_iters.keys()):
                del best_iters[k]
            for name in grp["iters"]:
                best_iters.create_dataset(name, data=grp["iters"][name][()])

            med_iters = summary.require_group("median_iters")
            for k in list(med_iters.keys()):
                del med_iters[k]
            for name in grp["iters"]:
                med_iters.create_dataset(name, data=grp["iters"][name][()])

            print(f"\n=== Summary saved (single seed: {seed_name}) ===")
        else:
            best_seed, median_seed = find_best_and_median_seeds(h5f)

            grp_best = h5f[best_seed]
            save("best_seed_name", np.bytes_(best_seed))
            save("best_objective", grp_best["objective_list"][-1])
            save("best_tv", grp_best["tv_list"][-1])
            save("best_compliance", grp_best["compliance_list"][-1])

            best_iters = summary.require_group("best_iters")
            for k in list(best_iters.keys()):
                del best_iters[k]
            for name in grp_best["iters"]:
                best_iters.create_dataset(name, data=grp_best["iters"][name][()])

            grp_med = h5f[median_seed]
            save("median_seed_name", np.bytes_(median_seed))
            save("median_objective", grp_med["objective_list"][-1])
            save("median_tv", grp_med["tv_list"][-1])
            save("median_compliance", grp_med["compliance_list"][-1])

            med_iters = summary.require_group("median_iters")
            for k in list(med_iters.keys()):
                del med_iters[k]
            for name in grp_med["iters"]:
                med_iters.create_dataset(name, data=grp_med["iters"][name][()])

            print(f"\n=== Summary saved (best: {best_seed}, median: {median_seed}) ===")


def run_trial(dim: int, idx: int, params) -> None:
    """
    Runs one mesh-size trial and writes results into an HDF5 file (key-value structure).
    """
    print(f"=== TRIAL {idx} | dim={dim} ===", flush=True)

    
    V_max = params.VOL_FRAC
    f = params.SOURCE_STRENGTH
    penalty_update_method = str(params.PENALTY_UPDATE_METHOD).strip().lower()
    
    # Determine number of seeds based on backend
    backend = (params.BACKEND or "").lower()
    if backend == "mergesplit":
        num_seeds = 10
        print(f"Using mergesplit backend: running {num_seeds} random seeds")
    else:
        num_seeds = 1
        print(f"Using gurobi backend: running {num_seeds} seed (deterministic solver)")
    
    seeds = [random.randint(0, 10000) for _ in range(num_seeds)]

    alpha = float(params.ALPHA)
    rho = float(params.RHO)
    print(f"\n--- Running with alpha={alpha} ---")

    for (i, seed_i) in enumerate(seeds):

        # --- ADMM variables
        dv = DesignVariables(seed=42, size=dim, Vmax=V_max)
        a_k, b_k, lam_k = dv.a, dv.b, dv.lam
        rho_k = rho

        print(f"======= Running for seed number {i}, value {seed_i}")

        # --- Create mesh and BC ---
        mesh = generate_unit_square_mesh(dim)
        
        # Read dirichlet boundaries and values from config
        dirichlet_boundaries = params.DIRICHLET_BOUNDARIES
        bc_values = params.BC_VALUES
        bc = build_dirichlet_bc_from_config(mesh, dirichlet_boundaries, bc_values)
        
        # --- Create load vector (uniform) ---
        f_vector = np.full(mesh.coords.shape[0], float(params.SOURCE_STRENGTH), dtype=float)
        
        # --- Solver setup ---
        sub2 = Subproblem2Solver(
            n_x=dim,
            n_y=2 * dim,
            alpha=alpha,
            seed=seed_i,
            use_mip=bool(params.USE_MIP),
            cutoff_time=float(params.CUTOFF_TIME),
        )
        sub1 = Subproblem1Solver(
            mesh=mesh,
            bc=bc,
            f=f_vector,
            volfrac=params.VOL_FRAC,
            material=MaterialInterpolation(penal=3.0, eps=1e-3),
        )

        # --- Data storage lists, regular list storing objectives at iter k
        a_list, b_list, a_disc_list, u_list, lam_list, grad_list = [], [], [], [], [], []
        obj_list, tv_list, compliance_list, infeas_list = [], [], [], []
        obj_disc_list, tv_disc_list, compliance_disc_list = [], [], []
        rho_track_list = []

        runtime1_list, runtime2_list = [], []
        rel_decrease_list = []

        # OC convergence tracking for each subproblem-1 solve call
        # (each item corresponds to one accepted ADMM iterate).
        oc_track_per_admm_iter = []


        # ---- Initial values of iterates ---
        a_list.append(a_k)
        b_list.append(b_k)
        a_disc_list.append(a_k.copy())
        lam_list.append(lam_k)
        infeas_list.append(np.linalg.norm(a_k - b_k)**2)
        #grad_list.append(0.4*np.ones(len(b_k)))
        u_list.append(np.zeros((dim+1)*(dim+1)))
        sub1_obj_k, compliance_k, pen_k, _ = sub1.compute_objective(a_k, a_k, lam_k, rho_k)
        compliance_list.append(compliance_k)
        tv_list.append(sub2.compute_TV(a_k, b_k, lam_k, rho_k))
        obj_list.append(compliance_list[-1] + tv_list[-1])

        k = 0

        while k < params.ITER_MAX and infeas_list[-1] > params.INFEAS_TOL:
                # --- Subproblem 1 ---
                t0 = time.perf_counter()
                b_k1, u_k1, oc_track_sub1 = sub1.solve(a_k, b_k, lam_k, rho_k, k, track_oc_convergence=params.TRACK_OC_CONVERGENCE)
                runtime1_list.append(time.perf_counter() - t0)

                # --- Subproblem 2 ---
                t0 = time.perf_counter()
                sol, status = sub2.run(a_k, b_k1, lam_k, rho_k, V_max, seed_i, k, params.BACKEND)
                a_k1 = sol.copy()

                # Rounding the solution when solving continuous relaxation
                if params.USE_MIP == False:
                    sorted_idx = np.argsort(sol.copy())[::-1]
                    a_disc_k1 = np.zeros_like(a_k1)
                    a_disc_k1[sorted_idx[:int(V_max*len(a_k1))]] = 1
                else:
                    a_disc_k1 = a_k1.copy()

                runtime2_list.append(time.perf_counter() - t0)
                
                # --- dual update step (regular ADMM) ---
                lam_k1 = lam_k + rho_k * (b_k1 - a_k1)
        
                '''----------Compute everything----------
                This includes computing Compliance, TV and Objective values
                '''
                sub1_obj_k1, compliance_k1, pen_k1, _ = sub1.compute_objective(a_k1, a_k1, lam_k1, rho_k)
                _, compliance_disc_k1, _, _ = sub1.compute_objective(a_k1, a_disc_k1, lam_k1, rho_k)
                u_disc_k1 = sub1.solve_state(a_disc_k1)
                tv_k1 = sub2.compute_TV(a_k1, b_k1, lam_k1, rho_k)
                tv_disc_k1 = sub2.compute_TV(a_disc_k1, b_k1, lam_k1, rho_k)


                # store iterates
                a_list.append(a_k1.copy())
                b_list.append(b_k1.copy())
                a_disc_list.append(a_disc_k1.copy())
                lam_list.append(lam_k1.copy())
                # gradient is the value \nabla_b L_{\rho} (with respect to continuous variable)
                
                u_list.append(u_disc_k1.copy())
                infeas_list.append(np.linalg.norm(b_k1 - a_k1)**2)
                rho_track_list.append(rho_k)
                    
                
                # --- Store regular list at iter k
                # --- Thee are our objectives that we actually care about
                compliance_list.append(compliance_k1)
                tv_list.append(tv_k1)
                obj_list.append(compliance_k1 + tv_k1)
                compliance_disc_list.append(compliance_disc_k1)
                tv_disc_list.append((np.sqrt(len(a_k1)/2))*tv_disc_k1/alpha)
                obj_disc_list.append(compliance_disc_k1 + tv_disc_k1)
                
                
                #### Penalty parameter update logic (if enabled) ####
                ### running average based which keeps track of infeasibility
                ### right from the first iteration and updates based on 
                ### the running average of relative infeasibility decrease.
                if penalty_update_method == "running_avg":
                    # Adaptive rho update based on running mean of relative infeasibility decrease.
                    if len(infeas_list) >= 2:
                        e_prev = infeas_list[-2]
                        e_curr = infeas_list[-1]
                        d_k = (e_prev - e_curr) / (e_prev + params.DECREASE_EPS)
                        rel_decrease_list.append(d_k)

                        if len(rel_decrease_list) >= params.STAG_WINDOW:
                            d_bar_k = np.mean(rel_decrease_list[-params.STAG_WINDOW:])
                            if d_bar_k > params.SLOW_THRESH:
                                trend = "good"
                            elif d_bar_k > params.STAG_THRESH:
                                trend = "slow"
                            else:
                                trend = "stagnation/increase"

                            print(
                                f"> Running mean relative decrease over last {params.STAG_WINDOW} steps: "
                                f"{d_bar_k:.6e} ({trend})",
                                flush=True,
                            )

                            if d_bar_k <= params.STAG_THRESH:
                                rho_k *= params.RHO_INCREASE_FACTOR
                                # Scaled-dual form: keep unscaled multiplier consistent when rho changes.
                                #lam_k1 /= params.RHO_INCREASE_FACTOR
                                print(
                                    f"> Stagnation detected. Increasing rho to {rho_k:.6e} "
                                    f"(factor={params.RHO_INCREASE_FACTOR:.3f})",
                                    flush=True,
                                )
                ### This is a bit weird but seems to work too, here I keep the
                ### penalty parameter fixed for the first 10 iterations and then
                ###  update it based on a schedule but here I just update every iteration.
                elif penalty_update_method == "periodic":
                    # Start periodic updates after iter 10, then update every 2 iterations.
                    if (k + 1) > 10 and ((k + 1 - 10) % 1 == 0):
                        rho_k *= params.RHO_INCREASE_FACTOR
                        # Scaled-dual form: keep unscaled multiplier consistent when rho changes.
                        #lam_k1 /= params.RHO_INCREASE_FACTOR
                        print(
                            f"> Periodic rho update at iter {k + 1}: rho={rho_k:.6e} "
                            f"(factor={params.RHO_INCREASE_FACTOR:.3f})",
                            flush=True,
                        )

                if params.TRACK_OC_CONVERGENCE and oc_track_sub1 is not None:
                    oc_track_per_admm_iter.append(oc_track_sub1)

                ''' Update iterates'''
                a_k, b_k, lam_k = a_k1, b_k1, lam_k1

                
                k += 1
                infeas_k = (np.linalg.norm(a_k - b_k) ** 2)
                print(f"> Infeasibility: {infeas_k:.6e}", flush=True)
                
        # --- Save
        if penalty_update_method == "none":
            base_dir = f"run_data_admm_{backend}"
        else:
            base_dir = f"run_data_admm_{penalty_update_method}_{backend}"
        if params.USE_MIP:
            base_dir += "_mip"
        new_dir = os.path.join(base_dir, str(alpha))
        os.makedirs(new_dir, exist_ok=True)
        h5_path = os.path.join(new_dir, f"{dim}.h5")

        print(f"Saving results to {h5_path}")
        save_data(
            h5_path=h5_path,
            seed_idx=i,
            num_seeds=num_seeds,
            dim=dim,
            idx=idx,
            backend=backend,
            alpha=alpha,
            rho_init=params.RHO,
            rho_final=rho_k,
            final_iter=k,
            obj_list=obj_list,
            tv_list=tv_list,
            compliance_list=compliance_list,
            obj_disc_list=obj_disc_list,
            tv_disc_list=tv_disc_list,
            compliance_disc_list=compliance_disc_list,
            runtime1_list=runtime1_list,
            runtime2_list=runtime2_list,
            infeas_list=infeas_list,
            rho_track_list=rho_track_list,
            a_list=a_list,
            b_list=b_list,
            a_disc_list=a_disc_list,
            u_list=u_list,
            lam_list=lam_list,
            track_oc_convergence=params.TRACK_OC_CONVERGENCE,
            oc_track_per_admm_iter=oc_track_per_admm_iter,
        )
        print(f">> Added results for seed={seed_i} (dim={dim}, {k} iterations)")


REQUIRED_CONFIG_TYPES = {
    "ITER_MAX": int,
    "RHO": float,
    "SEED_INIT": int,
    "USE_PREV": bool,
    "USE_MIP": bool,
    "BACKEND": str,
    "CUTOFF_TIME": float,
    "SOURCE_STRENGTH": float,
    "VOL_FRAC": float,
    "MESH_SIZE": int,
    "ALPHA": float,
    "PENALTY_UPDATE_METHOD": str,
    "RHO_INCREASE_FACTOR": float,
    "STAG_WINDOW": int,
    "DECREASE_EPS": float,
    "SLOW_THRESH": float,
    "STAG_THRESH": float,
    "INFEAS_TOL": float,
    "TRACK_OC_CONVERGENCE": bool,
}


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "admm_config.cfg")


def parse_cfg_value(key: str, raw_value: str):
    value_type = REQUIRED_CONFIG_TYPES[key]

    if value_type is bool:
        return raw_value.strip().lower() in ("1", "true", "yes", "on")
    if value_type is int:
        return int(raw_value)
    if value_type is float:
        return float(raw_value)
    return raw_value


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = {}

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    parser.read(config_path)

    # Load standard config values
    for section_name in parser.sections():
        for key, raw_value in parser.items(section_name):
            key_upper = key.upper()
            if key_upper in REQUIRED_CONFIG_TYPES:
                config[key_upper] = parse_cfg_value(key_upper, raw_value)

    # Load boundary condition values dynamically
    if "BOUNDARY_CONDITIONS" in parser.sections():
        bc_section = parser["BOUNDARY_CONDITIONS"]
        
        # Parse dirichlet_boundaries
        dirichlet_boundaries_str = bc_section.get("dirichlet_boundaries", "").strip()
        dirichlet_boundaries = [b.strip() for b in dirichlet_boundaries_str.split() if b.strip()]
        config["DIRICHLET_BOUNDARIES"] = dirichlet_boundaries
        
        # Extract dirichlet values for each boundary
        config["BC_VALUES"] = {}
        for boundary in dirichlet_boundaries:
            value_key = f"dirichlet_value_{boundary}"
            if value_key in bc_section:
                config["BC_VALUES"][boundary] = float(bc_section[value_key])
            else:
                config["BC_VALUES"][boundary] = 0.0
        
        # Neumann value
        config["NEUMANN_VALUE"] = float(bc_section.get("neumann_value", "0.0"))

    # Check for missing required keys (exclude BC keys since they're optional/dynamic)
    bc_optional_keys = {"DIRICHLET_BOUNDARIES", "NEUMANN_VALUE"}
    required_keys = set(REQUIRED_CONFIG_TYPES.keys()) - bc_optional_keys
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            "Missing required config keys in admm_config.cfg: " + ", ".join(missing)
        )

    return config


def main():
    args = argparse.Namespace()
    config = load_config(CONFIG_FILE)
    for key, value in config.items():
        setattr(args, key, value)

    # Reproducibility
    np.random.seed(args.SEED_INIT)

    # Single config-driven run
    run_trial(dim=int(args.MESH_SIZE), idx=0, params=args)
    
    print("=== Job finished and data saved ===", flush=True)


if __name__ == "__main__":
    main()
