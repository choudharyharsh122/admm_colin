# fenics_api.py
# API-style loader + plotting for FEniCS-backed fields (DG0 control, CG1 state).
# Usage:
#   from fenics_api import fenicsModel
#   fm = fenicsModel(alpha=1e-6, n=16, seed=0)
#   fm.obj_cont, fm.control_cont, fm.solution_disc
#   fm.plot_state(cont=True)         # continuous
#   fm.plot_control(cont=True)       # continuous
#   fm.plot_state()                  # discrete default
#   fm.plot_control()                # discrete default
#   fm.describe()                    # list attributes + methods

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dolfin import UnitSquareMesh, FunctionSpace, Function, plot as fenics_plot


# ----------------------------
# Internal helpers
# ----------------------------

def _build_spaces(n: int):
    """
    Triangular mesh with 2*n*n cells.
    DG0 for control (cellwise constants), CG1 for state.
    """
    mesh = UnitSquareMesh(n, n)            # triangles by default
    Vc = FunctionSpace(mesh, "DG", 0)      # control
    Vu = FunctionSpace(mesh, "CG", 1)      # state
    return mesh, Vc, Vu


def _as_control_function(c: np.ndarray, Vc: FunctionSpace) -> Function:
    f = Function(Vc)
    f.vector().set_local(np.asarray(c, dtype=float))
    f.vector().apply("insert")
    return f


def _as_state_function(u: np.ndarray, Vu: FunctionSpace) -> Function:
    f = Function(Vu)
    f.vector().set_local(np.asarray(u, dtype=float))
    f.vector().apply("insert")
    return f


def _plot_fenics_function_on_ax(f: Function, ax, title: str):
    """
    Reliable matplotlib + FEniCS plot with per-axis colorbar.
    """
    plt.sca(ax)
    ax.clear()
    m = fenics_plot(f)  # attaches to current axis
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)              # remove internal padding
    ax.autoscale_view()        # rescale to data

    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="4%", pad=0.1)

    #cbar = plt.colorbar(m, cax=cax, shrink=0.85)
    #cbar = plt.colorbar(m, ax=ax)
    #cbar.ax.tick_params(labelsize=14)
    #ax.set_title(title)
    ax.set_axis_off()
    #ax.set_title(title)


# ----------------------------
# Data container + API
# ----------------------------

@dataclass
class FenicsModel:
    # identifiers
    alpha: float
    n: int
    seed: int

    # raw blocks
    metadata: Dict[str, Any]
    objectives: Dict[str, float]
    arrays: Dict[str, np.ndarray]

    # FEniCS objects
    mesh: Any
    Vc: Any
    Vu: Any

    # convenient aliases (numerical)
    objective_cont: float
    objective: float
    compliance_cont: float
    compliance: float
    tv_cont: float
    tv: float
    runtime_total: float

    # FEniCS Functions
    control_cont: Function
    control: Function
    state_cont: Function
    state: Function

    # ---------- convenience ----------
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": ["Objective", "Compliance", "Total Variation"],
            "Continuous": [self.objective_cont, self.compliance_cont, self.tv_cont],
            "Discrete":   [self.objective, self.compliance, self.tv],
        })

    def describe(self) -> Dict[str, List[str]]:
        """
        Returns a compact inventory of what you can access/call.
        (Useful instead of digging through dir() output.)
        """
        attrs = [
            "alpha", "n", "seed",
            "metadata", "objectives", "arrays",
            "mesh", "Vc", "Vu",
            "objective_cont", "objective",
            "compliance_cont", "compliance",
            "tv_cont", "tv",
            "runtime_total",
            "control_cont", "control",
            "state_cont", "state",
        ]
        methods = [
            "summary_df()",
            "describe()",
            "plot_control(cont=False, figsize=(6,5))",
            "plot_state(cont=False, figsize=(6,5))",
            "plot_controls(figsize=(14,5))",
            "plot_states(figsize=(14,5))",
        ]
        return {"attributes": attrs, "methods": methods}

    # ---------- plotting API (discrete default) ----------
    def plot_control(self, cont: bool = False, figsize=(5, 4)) -> None:
        """
        Plot control (DG0). Discrete by default unless cont=True.
        """
        f = self.control_cont if cont else self.control
        title = "Continuous control (DG0)" if cont else "Discrete control (DG0)"

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _plot_fenics_function_on_ax(f, ax, title)
        plt.tight_layout()
        plt.show()

    def plot_state(self, cont: bool = False, figsize=(5, 4)) -> None:
        """
        Plot state (CG1). Discrete by default unless cont=True.
        """
        f = self.state_cont if cont else self.state
        title = "Continuous state (CG1)" if cont else "Discrete state (CG1)"

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _plot_fenics_function_on_ax(f, ax, title)
        plt.tight_layout()
        plt.show()

    def plot_controls(self, figsize=(14, 5)) -> None:
        """
        Side-by-side plot for continuous vs discrete control.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _plot_fenics_function_on_ax(self.control_cont, axes[0], "Continuous control (DG0)")
        _plot_fenics_function_on_ax(self.control, axes[1], "Discrete control (DG0)")
        plt.tight_layout()
        plt.show()

    def plot_states(self, figsize=(14, 5)) -> None:
        """
        Side-by-side plot for continuous vs discrete state.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _plot_fenics_function_on_ax(self.state_cont, axes[0], "Continuous state (CG1)")
        _plot_fenics_function_on_ax(self.state, axes[1], "Discrete state (CG1)")
        plt.tight_layout()
        plt.show()


# ----------------------------
# Factory function
# ----------------------------

def fenicsModel(
    alpha: float,
    n: int,
    seed: int = 0,
    base_dir: str = "fenics_model_tri_1e-06",
) -> FenicsModel:
    """
    Usage:
        fm = fenicsModel(alpha=1e-6, n=16, seed=0)

    Loads:
        {base_dir}/{alpha}/{n}.h5 , group seed_{seed}

    Returns FenicsModel with convenient fields:
        fm.objective_cont, fm.control_cont, fm.state_disc, ...
    """
    h5_path = os.path.join(base_dir, f"{alpha}", f"{n}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        seed_key = f"seed_{seed}"
        
        # Check if seed_key exists, otherwise try 'summary'
        if seed_key not in h5f:
            if 'summary' in h5f:
                seed_key = 'summary'
            else:
                raise KeyError(f"Seed '{seed_key}' not found. Available: {list(h5f.keys())}")

        grp = h5f[seed_key]

        metadata = {
            "dim": grp.attrs.get("dim"),
            "alpha": grp.attrs.get("alpha"),
            "V_frac": grp.attrs.get("V_frac"),
            "runtime_total": grp.attrs.get("runtime_total"),
        }

        objectives = {
            "cont_objective": float(grp["cont_objective"][()]),
            "cont_TV": float(grp["cont_TV"][()]),
            "cont_compliance": float(grp["cont_compliance"][()]),
            "disc_objective": float(grp["disc_objective"][()]),
            "disc_TV": float(grp["disc_TV"][()]),
            "disc_compliance": float(grp["disc_compliance"][()]),
        }

        arrays = {
            "a_opt":  np.array(grp["a_opt"], dtype=float),
            "a_disc": np.array(grp["a_disc"], dtype=float),
            "u_opt":  np.array(grp["u_opt"], dtype=float),
            "u_disc": np.array(grp["u_disc"], dtype=float),
        }

    mesh, Vc, Vu = _build_spaces(n)

    control_cont = _as_control_function(arrays["a_opt"], Vc)      # DG0
    control = _as_control_function(arrays["a_disc"], Vc)     # DG0
    state_cont = _as_state_function(arrays["u_opt"], Vu)       # CG1
    state = _as_state_function(arrays["u_disc"], Vu)      # CG1

    tv_cont = float(objectives["cont_TV"])
    tv_disc = float(objectives["disc_TV"])

    runtime_total = metadata.get("runtime_total")

    return FenicsModel(
        alpha=alpha,
        n=n,
        seed=seed,
        metadata=metadata,
        objectives=objectives,
        arrays=arrays,
        mesh=mesh,
        Vc=Vc,
        Vu=Vu,
        objective_cont=float(objectives["cont_objective"]),
        objective=float(objectives["disc_objective"]),
        compliance_cont=float(objectives["cont_compliance"]),
        compliance=float(objectives["disc_compliance"]),
        tv_cont=tv_cont,
        tv=tv_disc,
        runtime_total=float(runtime_total) if runtime_total is not None else float("nan"),
        control_cont=control_cont,
        control=control,
        state_cont=state_cont,
        state=state,
    )
