# API-style loader + lazy access for ADMM multi-seed HDF5 files.
#
# Usage:
#   from admmviz import ADMM
#   admm = ADMM(alpha=1e-6, dim=16, suffix="tri")   # suffix optional
#
#   # Median (default) summary
#   admm.objective, admm.tv, admm.compliance
#
#   # Median (default) final iterates
#   admm.control, admm.control_1, admm.state
#
#   # Best summary + final iterates
#   admm.objective_best, admm.tv_best, admm.compliance_best
#   admm.control_best, admm.control_1_best, admm.state_best
#
#
#   m = admm.median_seed   # int
#   b = admm.best_seed     # int
#   admm.trial(m).series.objective            # full objective_list
#   admm.trial(m).iters.state                 # full u_list (loaded lazily)
#   admm.trial(b).meta["rho_final"]
#
# Notes:
# - This expects the file structure as:
#     /seed_{i}/objective_list, tv_list, compliance_list, infeas_list, ...
#     /seed_{i}/iters/a_list, b_list, u_list, lambda_list
#     /summary/*  and  /summary/best_iters/*, /summary/median_iters/*
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import re

import h5py
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from dolfin import UnitSquareMesh, FunctionSpace, Function, plot as fenics_plot



SeedLike = Union[int, str]


# ---------------------------
# Utilities
# ---------------------------

_SEED_RE = re.compile(r"^seed_(\d+)$")
_ADMM_ITER_RE = re.compile(r"^admm_iter_(\d+)$")


def _build_tri_points_and_indices(n: int):
    """
    Triangles ordered as: for I=0..n-1, for J=0..n-1: L then U
      L(I,J) = [(I,J),(I+1,J),(I+1,J+1)]
      U(I,J) = [(I,J),(I+1,J+1),(I,J+1)]
    """
    triangles = []
    for I in range(n):
        for J in range(n):
            triangles.append([(I, J), (I + 1, J), (I + 1, J + 1)])   # L
            triangles.append([(I, J), (I + 1, J + 1), (I, J + 1)])   # U

    points = []
    tri_indices = []
    for tri in triangles:
        idxs = []
        for (i, j) in tri:
            points.append([i, j])
            idxs.append(len(points) - 1)
        tri_indices.append(idxs)

    return np.array(points, float), np.array(tri_indices, int)

def _transpose_control_triangles(a: np.ndarray, n: int) -> np.ndarray:
    """
    Fix global diagonal reflection (transpose) for triangle-wise controls.
    Maps (I,J) -> (J,I) for both L and U entries.
    """
    a = np.asarray(a, dtype=float)
    if a.size != 2 * n * n:
        raise ValueError(f"Expected control length {2*n*n}, got {a.size}")

    out = np.empty_like(a)
    for I in range(n):
        for J in range(n):
            src = 2 * (J + I * n)      # swapped
            dst = 2 * (I + J * n)
            out[dst] = a[src]          # L
            out[dst + 1] = a[src + 1]  # U
    return out


def plot_control_field(control_vec, dim: int, *, ax=None, title: str = "Control", figsize=(5, 4), show: bool = True):
    mesh, Vc, _ = _build_fenics_spaces(dim)
    a = np.asarray(control_vec, dtype=float).ravel()

    expected = 2 * dim * dim
    if a.size != expected:
        raise ValueError(f"Control length mismatch: expected {expected} (=2*{dim}*{dim}), got {a.size}")

    f = _as_fenics_function(a, Vc)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    plt.sca(ax)
    ax.clear()
    m = fenics_plot(f, edgecolor="none")
    ax.margins(0)
    ax.autoscale_view()

    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="4%", pad=0.1)
    #cbar = plt.colorbar(m, cax=cax, shrink=0.85)
    #cbar.set_ticks(np.linspace(0, 1, 6))
    #cbar.ax.tick_params(labelsize=14)
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=18)

    if show and created_fig:
        plt.show()

    return ax, m


def plot_state_field(ax, n: int, u: np.ndarray, title: str, figsize=(5, 4), cmap: str = "viridis", show: bool = True):
    u = np.asarray(u, dtype=float)
    expected = (n + 1) * (n + 1)
    if u.size != expected:
        raise ValueError(f"Expected state length {expected}, got {u.size}")

    X, Y = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    u_grid = u.reshape((n + 1, n + 1))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    im = ax.pcolormesh(X, Y, u_grid, shading="gouraud", cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    if show:
        plt.show()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, shrink=0.85)
    cbar.set_ticks(np.linspace(u.min(), u.max(), 6))
    cbar.ax.tick_params(labelsize=14)
    ax.set_axis_off()

    return im


def _seed_int_to_name(seed: int) -> str:
    return f"seed_{int(seed)}"


def _seed_name_to_int(seed_name: str) -> int:
    m = _SEED_RE.match(seed_name)
    if not m:
        raise ValueError(f"Invalid seed group name: {seed_name!r} (expected 'seed_<int>')")
    return int(m.group(1))


def _decode_h5_scalar(x) -> Any:
    """
    Decode HDF5 scalars that might come out as bytes or 0-d arrays.
    """
    # h5py can return numpy scalars, bytes, or arrays
    if isinstance(x, (np.ndarray,)):
        if x.shape == ():
            x = x[()]
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def _read_last(ds) -> float:
    """
    Read last element of 1D dataset robustly.
    """
    # h5py supports negative indexing on datasets, but this keeps it explicit
    n = ds.shape[0]
    return float(ds[n - 1])

def _build_fenics_spaces(dim: int):
    """
    Same spaces as fenicsModel:
      DG0 for control (2*dim*dim dofs on UnitSquareMesh(dim,dim))
      CG1 for state   ((dim+1)^2 dofs)
    """
    mesh = UnitSquareMesh(dim, dim)
    Vc = FunctionSpace(mesh, "DG", 0)
    Vu = FunctionSpace(mesh, "CG", 1)
    return mesh, Vc, Vu


def _as_fenics_function(vec, V: FunctionSpace) -> Function:
    f = Function(V)
    f.vector().set_local(vec)
    f.vector().apply("insert")
    return f


# def plot_control(control_vec, dim: int, *, title: str = "Control", figsize=(5, 4)) -> None:
#     """
#     Generic FEniCS control plotter (DG0).
#     control_vec must be length 2*dim*dim.
#     """
#     mesh, Vc, _ = _build_fenics_spaces(dim)
#     a = np.asarray(control_vec, dtype=float).ravel()

#     expected = 2 * dim * dim
#     if a.size != expected:
#         raise ValueError(f"Control length mismatch: expected {expected} (=2*{dim}*{dim}), got {a.size}")

#     f = _as_fenics_function(a, Vc)

#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plt.sca(ax)
#     ax.clear()
#     m = fenics_plot(f, edgecolor='none')

#     #ax.set_aspect("equal", adjustable="box")
#     ax.margins(0)              # remove internal padding
#     ax.autoscale_view()        # rescale to data

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="4%", pad=0.1)

#     cbar = plt.colorbar(m, cax=cax, shrink=0.85)
#     #cbar = plt.colorbar(m, ax=ax)
#     cbar.set_ticks(np.linspace(0, 1, 6))
#     cbar.ax.tick_params(labelsize=14)
#     #ax.set_title(title)
#     ax.set_axis_off()
#     #plt.tight_layout()
    
#     plt.show()


# def plot_state(state_vec, dim: int, *, title: str = "State", figsize=(5, 4)) -> None:
#     """
#     Generic FEniCS state plotter (CG1).
#     state_vec must be length (dim+1)^2.
#     """
#     mesh, _, Vu = _build_fenics_spaces(dim)
#     u = np.asarray(state_vec, dtype=float).ravel()

#     expected = (dim + 1) * (dim + 1)
#     if u.size != expected:
#         raise ValueError(f"State length mismatch: expected {expected} (=(dim+1)^2), got {u.size}")

#     f = _as_fenics_function(u, Vu)

#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plt.sca(ax)
#     ax.clear()
#     m = fenics_plot(f)
#     ax.set_aspect("equal", adjustable="box")
#     ax.margins(0)              # remove internal padding
#     ax.autoscale_view()        # rescale to data
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="4%", pad=0.1)

#     cbar = plt.colorbar(m, cax=cax, shrink=0.85)
#     #cbar = plt.colorbar(m, ax=ax)
#     cbar.ax.tick_params(labelsize=14)
#     #ax.set_title(title)
#     ax.set_axis_off()
#     #ax.set_title(title)
#     #plt.tight_layout()
#     plt.show()


# ---------------------------
# Views for a seed group
# ---------------------------

@dataclass(frozen=True)
class _SeriesView:
    """
    Access to per-iteration scalar series arrays under seed_{i}/.
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            if key not in grp:
                raise KeyError(f"{self.seed_name}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def objective(self) -> np.ndarray:
        return self._read("objective_list")

    @property
    def tv(self) -> np.ndarray:
        return self._read("tv_list")

    @property
    def compliance(self) -> np.ndarray:
        return self._read("compliance_list")
    
    @property
    def objective_disc(self) -> np.ndarray:
        return self._read("objective_disc_list")

    @property
    def tv_disc(self) -> np.ndarray:
        return self._read("tv_disc_list")

    @property
    def compliance_disc(self) -> np.ndarray:
        return self._read("compliance_disc_list")

    @property
    def infeas(self) -> np.ndarray:
        return self._read("infeas_list")

    @property
    def rho(self) -> np.ndarray:
        return self._read("rho_list")

    @property
    def runtime_sub1(self) -> np.ndarray:
        return self._read("runtime_sub1_list")

    @property
    def runtime_sub2(self) -> np.ndarray:
        return self._read("runtime_sub2_list")

    @property
    def h_tvs(self) -> np.ndarray:
        # Optional metric in some legacy files.
        return self._read("h_tvs")


@dataclass(frozen=True)
class _ItersView:
    """
    Access to iterate matrices under seed_{i}/iters/.
    Convention mapping:
      a_list -> control
      b_list -> control_1
      a_disc_list -> discrete control
      u_list -> state
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/iters"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def control_2(self) -> np.ndarray:
        return self._read("a_list")

    @property
    def control_disc(self) -> np.ndarray:
        return self._read("a_disc_list")

    @property
    def control_1(self) -> np.ndarray:
        return self._read("b_list")

    @property
    def state(self) -> np.ndarray:
        return self._read("u_list")

    @property
    def lam(self) -> np.ndarray:
        return self._read("lambda_list")

    # @property
    # def gradL(self) -> np.ndarray:
    #     return self._read("gradL_list")

    # Convenience finals
    @property
    def control_1_final(self) -> np.ndarray:
        x = self.control_1
        return x[-1].copy()

    @property
    def control_2_final(self) -> np.ndarray:
        x = self.control_2
        return x[-1].copy()
    
    @property
    def control_disc_final(self) -> np.ndarray:
        x = self.control_disc
        return x[-1].copy()

    @property
    def state_final(self) -> np.ndarray:
        x = self.state
        return x[-1].copy()

    @property
    def oc_iters(self) -> "_OCItersView":
        """
        Access to OC-inner-iteration tracking saved under:
          seed_{i}/oc_subproblem1_tracking/admm_iter_k/*

        Examples:
          trial(i).iters.oc_iters.F[k]        -> np.ndarray of F values for ADMM iter k
          trial(i).iters.oc_iters.gradF[k]    -> np.ndarray [n_oc, nele]
          trial(i).iters.oc_iters[k].F        -> same as above via indexed view
        """
        return _OCItersView(self.h5_path, self.seed_name)


@dataclass(frozen=True)
class _OCIterView:
    """
    View into one ADMM iteration's OC tracking group:
      seed_{i}/oc_subproblem1_tracking/admm_iter_k/*
    """
    h5_path: str
    seed_name: str
    admm_iter: int

    def _read(self, key: str) -> np.ndarray:
        base = f"{self.seed_name}/oc_subproblem1_tracking/admm_iter_{int(self.admm_iter)}"
        with h5py.File(self.h5_path, "r") as h5f:
            if base not in h5f:
                raise KeyError(f"{base} group not found in file.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def F(self) -> np.ndarray:
        return self._read("F_list")

    @property
    def gradF(self) -> np.ndarray:
        return self._read("grad_F_list")

    @property
    def gradF_norm(self) -> np.ndarray:
        return self._read("grad_F_norm_list")


@dataclass(frozen=True)
class _OCItersView:
    """
    Collection view over all ADMM-iteration OC traces for one seed.
    """
    h5_path: str
    seed_name: str

    def _base(self) -> str:
        return f"{self.seed_name}/oc_subproblem1_tracking"

    def _iter_ids(self) -> List[int]:
        with h5py.File(self.h5_path, "r") as h5f:
            base = self._base()
            if base not in h5f:
                return []
            out = []
            for name in h5f[base].keys():
                m = _ADMM_ITER_RE.match(name)
                if m:
                    out.append(int(m.group(1)))
            return sorted(out)

    def __len__(self) -> int:
        return len(self._iter_ids())

    def __getitem__(self, admm_iter: int) -> _OCIterView:
        admm_iter = int(admm_iter)
        ids = self._iter_ids()
        if admm_iter not in ids:
            raise IndexError(f"ADMM iter {admm_iter} not found. Available: {ids}")
        return _OCIterView(self.h5_path, self.seed_name, admm_iter)

    @property
    def available_iters(self) -> List[int]:
        return self._iter_ids()

    @property
    def F(self) -> List[np.ndarray]:
        return [self[k].F for k in self._iter_ids()]

    @property
    def gradF(self) -> List[np.ndarray]:
        return [self[k].gradF for k in self._iter_ids()]

    @property
    def gradF_norm(self) -> List[np.ndarray]:
        return [self[k].gradF_norm for k in self._iter_ids()]


@dataclass(frozen=True)
class _PairsView:
    """
    Access to seed_{i}/pair_metrics/*
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/pair_metrics"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file. This run file does not save pair metrics.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def sub1_obj_pairs(self) -> np.ndarray:
        return self._read("sub1_obj_pairs")

    @property
    def compliance_pairs(self) -> np.ndarray:
        return self._read("compliance_pairs")

    @property
    def sub1_penalty_pairs(self) -> np.ndarray:
        return self._read("sub1_penalty_pairs")

    @property
    def sub2_obj_pairs(self) -> np.ndarray:
        return self._read("sub2_obj_pairs")

    @property
    def tv_pairs(self) -> np.ndarray:
        return self._read("tv_pairs")

    @property
    def sub2_penalty_pairs(self) -> np.ndarray:
        return self._read("sub2_penalty_pairs")


@dataclass(frozen=True)
class _TripletsView:
    """
    Access to seed_{i}/triplet_metrics/*
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/triplet_metrics"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file. This run file does not save triplet metrics.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def aug_lagr_triplets(self) -> np.ndarray:
        return self._read("aug_lagr_triplets")


@dataclass(frozen=True)
class Trial:
    """
    A lightweight, lazy view into a single seed_{k} group.
    """
    h5_path: str
    seed: int

    def describe_tree(self) -> Dict[str, Any]:
        """
        JSON tree description for one seed-level Trial view.
        """
        seed_path = self.seed_name

        with h5py.File(self.h5_path, "r") as h5f:
            has_pairs = f"{seed_path}/pair_metrics" in h5f
            has_triplets = f"{seed_path}/triplet_metrics" in h5f
            has_oc_tracking = f"{seed_path}/oc_subproblem1_tracking" in h5f

        oc_iters = self.iters.oc_iters.available_iters if has_oc_tracking else []

        return {
            "class": "Trial",
            "identity": {
                "seed": int(self.seed),
                "seed_name": self.seed_name,
                "h5_path": self.h5_path,
            },
            "attributes": {
                "meta_keys": sorted([str(k) for k in self.meta.keys()]),
                "objective_final": float(self.objective_final),
                "infeas_final": float(self.infeas_final),
            },
            "children": {
                "series": {
                    "type": "_SeriesView",
                    "fields": [
                        "objective",
                        "tv",
                        "compliance",
                        "infeas",
                        "runtime_sub1",
                        "runtime_sub2",
                        "h_tvs",
                    ],
                    "h5_base": f"{seed_path}/",
                },
                "iters": {
                    "type": "_ItersView",
                    "fields": [
                        "control_1",
                        "control_2",
                        "control_disc",
                        "state",
                        "lam",
                        "control_1_final",
                        "control_2_final",
                        "control_disc_final",
                        "state_final",
                        "oc_iters",
                    ],
                    "h5_base": f"{seed_path}/iters",
                },
                "oc_iters": {
                    "type": "_OCItersView",
                    "available": has_oc_tracking,
                    "available_iters": oc_iters,
                    "per_iter_fields": ["F", "gradF", "gradF_norm"],
                    "h5_base": f"{seed_path}/oc_subproblem1_tracking/admm_iter_k",
                },
                "pairs": {
                    "type": "_PairsView",
                    "available": has_pairs,
                    "fields": [
                        "sub1_obj_pairs",
                        "compliance_pairs",
                        "sub1_penalty_pairs",
                        "sub2_obj_pairs",
                        "tv_pairs",
                        "sub2_penalty_pairs",
                    ],
                    "h5_base": f"{seed_path}/pair_metrics",
                },
                "triplets": {
                    "type": "_TripletsView",
                    "available": has_triplets,
                    "fields": ["aug_lagr_triplets"],
                    "h5_base": f"{seed_path}/triplet_metrics",
                },
            },
        }

    @property
    def seed_name(self) -> str:
        return _seed_int_to_name(self.seed)

    @property
    def meta(self) -> Dict[str, Any]:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return dict(grp.attrs.items())

    @property
    def series(self) -> _SeriesView:
        return _SeriesView(self.h5_path, self.seed_name)

    @property
    def iters(self) -> _ItersView:
        return _ItersView(self.h5_path, self.seed_name)

    @property
    def pairs(self) -> _PairsView:
        return _PairsView(self.h5_path, self.seed_name)

    @property
    def triplets(self) -> _TripletsView:
        return _TripletsView(self.h5_path, self.seed_name)

    # Convenience: finals without loading full iters unless asked
    @property
    def objective_final(self) -> float:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return _read_last(grp["objective_list"])
    
    @property
    def objective_disc_final(self) -> float:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return _read_last(grp["objective_disc_list"])

    @property
    def infeas_final(self) -> float:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return _read_last(grp["infeas_list"])


# ---------------------------
# ADMM top-level API
# ---------------------------

class ADMMColin:
    """
    Summary-first API:
      - objective/tv/compliance/control/control_1/state refer to MEDIAN summary
      - *_best fields refer to BEST summary
      - median_seed / best_seed are ints to drill down with trial(seed)
      - trial(seed) returns a lazy view into that seed group

    File lookup:
            base_dir_{suffix}/{alpha}/{dim}.h5 (if suffix is provided)
            base_dir/{alpha}/{dim}.h5 (if suffix is empty)

    If you already have a full path, pass h5_path directly.
    """

    def __init__(
        self,
        alpha: float,
        dim: int,
        *,
        suffix: str = "",
        base_dir: str = "run_data_admm_mergesplit",
        h5_path: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.dim = int(dim)
        self.suffix = str(suffix)

        if h5_path is None:
            base = f"{base_dir}_{suffix}" if suffix else base_dir
            new_dir = os.path.join(base, str(alpha))
            h5_path = os.path.join(new_dir, f"{dim}.h5")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        self.h5_path = h5_path

        # ---- Eager-load ONLY summary essentials (small) ----
        self._load_summary()

    # -----------------
    # Public fields (summary-first)
    # -----------------

    # Scalars (median default)
    objective: float
    tv: float
    compliance: float

    # Final iterates (median default)
    control_2: np.ndarray
    control_1: np.ndarray
    control_disc: np.ndarray
    state: np.ndarray

    # Scalars (best)
    objective_best: float
    tv_best: float
    compliance_best: float

    # Final iterates (best)
    control_2_best: np.ndarray
    control_1_best: np.ndarray
    control_disc_best: np.ndarray
    state_best: np.ndarray

    # Seeds (int default)
    median_seed: int
    best_seed: int

    # Optional names
    median_seed_name: str
    best_seed_name: str

    # -----------------
    # Summary loading
    # -----------------

    def _load_summary(self) -> None:
        with h5py.File(self.h5_path, "r") as h5f:
            if "summary" not in h5f:
                raise KeyError("H5 file has no 'summary' group. Did you create it after runs?")
            s = h5f["summary"]

            # Seed names may be bytes; decode to str
            med_name = _decode_h5_scalar(s["median_seed_name"][()])
            best_name = _decode_h5_scalar(s["best_seed_name"][()])
            self.median_seed_name = str(med_name)
            self.best_seed_name = str(best_name)

            self.median_seed = _seed_name_to_int(self.median_seed_name)
            self.best_seed = _seed_name_to_int(self.best_seed_name)

            # Scalars
            self.objective = float(s["median_objective"][()])
            self.tv = float(s["median_tv"][()])
            self.compliance = float(s["median_compliance"][()])

            self.objective_best = float(s["best_objective"][()])
            self.tv_best = float(s["best_tv"][()])
            self.compliance_best = float(s["best_compliance"][()])

            # Final iterates from summary groups
            self.control_2 = np.array(s["median_iters"]["a_list"][-1], dtype=float)
            self.control_1 = np.array(s["median_iters"]["b_list"][-1], dtype=float)
            self.control_disc = np.array(s["median_iters"]["a_disc_list"][-1], dtype=float)
            self.state = np.array(s["median_iters"]["u_list"][-1], dtype=float)

            self.control_2_best = np.array(s["best_iters"]["a_list"][-1], dtype=float)
            self.control_1_best = np.array(s["best_iters"]["b_list"][-1], dtype=float)
            self.control_disc_best = np.array(s["best_iters"]["a_disc_list"][-1], dtype=float)
            self.state_best = np.array(s["best_iters"]["u_list"][-1], dtype=float)

            # Keep a small metadata dict available (cheap)
            # Prefer summary attrs if present; otherwise leave empty.
            self.metadata = dict(s.attrs.items()) if hasattr(s, "attrs") else {}

    # -----------------
    # Discoverability
    # -----------------

    def describe(self) -> Dict[str, List[str]]:
        """
        Compact listing of core attributes + methods.
        """
        attrs = [
            "alpha", "dim", "suffix", "h5_path",
            "median_seed", "median_seed_name",
            "best_seed", "best_seed_name",
            "objective", "tv", "compliance",
            "control_2", "control_1", "control_disc", "state",
            "objective_best", "tv_best", "compliance_best",
            "control_2_best", "control_1_best", "control_disc_best", "state_best",
            "metadata",
        ]
        methods = [
            "trial(k) -> Trial",
            "seeds() -> List[int]",
            "trials_df() -> pd.DataFrame",
            "reload_summary()",
            "describe()",
        ]
        return {"attributes": attrs, "methods": methods}

    # -----------------
    # Trial access
    # -----------------

    def trial(self, seed: SeedLike) -> Trial:
        """
        Return a lazy view into seed_{k}.
        seed can be int (preferred) or string like 'seed_7'.
        """
        if isinstance(seed, str):
            seed = _seed_name_to_int(seed)
        seed = int(seed)

        # Validate existence quickly without loading data
        with h5py.File(self.h5_path, "r") as h5f:
            name = _seed_int_to_name(seed)
            if name not in h5f:
                raise KeyError(f"Seed group '{name}' not found. Available: {list(h5f.keys())}")
        return Trial(self.h5_path, seed)

    def has_pair_metrics(self, seed: SeedLike) -> bool:
        """Return True if seed_{k}/pair_metrics exists."""
        if isinstance(seed, str):
            seed = _seed_name_to_int(seed)
        seed_name = _seed_int_to_name(int(seed))
        with h5py.File(self.h5_path, "r") as h5f:
            return f"{seed_name}/pair_metrics" in h5f

    def has_triplet_metrics(self, seed: SeedLike) -> bool:
        """Return True if seed_{k}/triplet_metrics exists."""
        if isinstance(seed, str):
            seed = _seed_name_to_int(seed)
        seed_name = _seed_int_to_name(int(seed))
        with h5py.File(self.h5_path, "r") as h5f:
            return f"{seed_name}/triplet_metrics" in h5f

    # -----------------
    # Seed inventory + lightweight summaries
    # -----------------

    def seeds(self) -> List[int]:
        """
        List available seed integers in the file.
        """
        with h5py.File(self.h5_path, "r") as h5f:
            out = []
            for k in h5f.keys():
                if k.startswith("seed_") and _SEED_RE.match(k):
                    out.append(_seed_name_to_int(k))
            return sorted(out)

    def trials_df(self, infeas_thresh: Optional[float] = None) -> pd.DataFrame:
        """
        Build a lightweight per-seed table of final metrics.
        Does NOT load iterates.
        If infeas_thresh is provided, include a boolean column 'feasible'.
        """
        rows = []
        with h5py.File(self.h5_path, "r") as h5f:
            for k in h5f.keys():
                if not (k.startswith("seed_") and _SEED_RE.match(k)):
                    continue
                grp = h5f[k]
                seed = _seed_name_to_int(k)

                obj_last = _read_last(grp["objective_list"])
                tv_last = _read_last(grp["tv_list"])
                comp_last = _read_last(grp["compliance_list"])
                obj_disc_last = _read_last(grp["objective_disc_list"])
                tv_disc_last = _read_last(grp["tv_disc_list"])
                comp_disc_last = _read_last(grp["compliance_disc_list"])
                infeas_last = _read_last(grp["infeas_list"])

                row = {
                    "seed": seed,
                    "objective_last": obj_last,
                    "tv_last": tv_last,
                    "compliance_last": comp_last,
                    "objective_disc_last": obj_last,
                    "tv_disc_last": tv_last,
                    "compliance_disc_last": comp_last,
                    "infeas_last": infeas_last,
                }

                # common attrs if present
                for attr_key in ["final_iter", "rho_init", "rho_final", "zeta", "beta", "idx", "backend", "dim", "alpha"]:
                    if attr_key in grp.attrs:
                        row[attr_key] = grp.attrs[attr_key]

                rows.append(row)

        df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
        if infeas_thresh is not None:
            df["feasible"] = df["infeas_last"] <= float(infeas_thresh)
        return df

    # -----------------
    # Refresh
    # -----------------

    def reload_summary(self) -> None:
        """
        Re-read summary from disk (useful if you update the H5 file).
        """
        self._load_summary()

    
    # ---- plotting API (discrete by default) ----
    def plot_control(
        self,
        control_vec: Optional[np.ndarray] = None,
        which: int = False,
        disc: bool = False,
        best: bool = False,
        ax = None,
        fix_diagonal_reflection: bool = False,
        figsize=(6, 5),
        title: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot control as triangle-wise tripcolor.

        Behavior:
          - If `control_vec` is provided: plot that vector directly.
          - Else:
              best=False -> plot median final control (discrete/continuous)
              best=True  -> plot best   final control (discrete/continuous)

        No title is added unless `title` is explicitly provided.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        if control_vec is not None:
            a = control_vec
        elif best:
            if disc:
                a = self.control_disc_best
            if which == 1:
                a = self.control_1_best
            elif which == 2:
                a = self.control_2_best
            else:
                print("Please enter a valid value for 'which' \in {1, 2}")
        else:
            if disc:
                a = self.control_disc
            if which == 1:
                a = self.control_1
            elif which == 2:
                a = self.control_2
            else:
                print("Please enter a valid value for 'which' \in {1, 2}")

        a = np.asarray(a, dtype=float)

        if fix_diagonal_reflection:
            a = _transpose_control_triangles(a, self.dim)


        plot_control_field(a, dim=self.dim, title=title, figsize=figsize, ax=ax, show=show)


    def plot_state(
    self,
    state_vec: Optional[np.ndarray] = None,
    cont: bool = False,
    best = False,
    ax = None,
    transpose: bool = False,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize=(6, 5),
    show: bool = True,
) -> None:
        """
        Plot state as node-wise pcolormesh.
        Discrete by default unless cont=True.
        If transpose=True, plot u reshaped then transposed.

        Styled to match your compact FEniCS plots:
        - square aspect
        - no axis box/ticks
        - plot fills the axes (no internal padding)
        - colorbar height matches axes and sits close
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # admm_run_random_seeds.py saves only one state iterate stream (u_list).
        # Keep cont for API compatibility, but both branches map to state.
        if state_vec is not None:
            u = state_vec
        elif best:
            u = self.state_best
        else:
            u = self.state

        if transpose:
            u = u.reshape((self.dim + 1, self.dim + 1)).T.ravel()

        # # Use this style (works well with axes_grid1)
        # plt.figure(figsize=figsize)
        # ax = plt.gca()
        # ax.clear()

        default_title = "State u (nodes)"
        plot_title = default_title if title is None else str(title)
        plot_state_field(ax, self.dim, u, title=plot_title, cmap=cmap, figsize = figsize, show=show)
        # if plot_title:
        #     ax.set_title(plot_title)

        # --- Make the plotted field fill the axes tightly ---
        # ax.set_aspect("equal", adjustable="box")
        # ax.margins(0)

        # For pcolormesh, this reliably tightens limits to the mesh extent
        # ax.relim()
        # ax.autoscale_view(tight=True)

        # Remove any extra padding that some helper functions introduce
        # try:
        #     ax.set_xlim(*im.axes.dataLim.intervalx)
        #     ax.set_ylim(*im.axes.dataLim.intervaly)
        # except Exception:
        #     pass

        # --- Colorbar: same height as the axes, small gap ---
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="4%", pad=0.1)

        # cbar = plt.colorbar(im, cax=cax)
        # cbar.ax.tick_params(labelsize=14)
        # #cbar.set_label("u(x)", fontsize=14)

        # # No border/ticks
        # ax.set_axis_off()

        # # Avoid tight_layout (it can fight the appended cax)
        # plt.show()

