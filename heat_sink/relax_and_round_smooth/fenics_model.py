from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import time
import os
import sys
import argparse
import h5py
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, cg, gmres, spilu, LinearOperator
from scipy.sparse.linalg import use_solver  # optional, for legacy
import warnings

set_log_active(False)                  # globally disable all DOLFIN log_output
set_log_level(LogLevel.ERROR)


def build_graph(n_x: int, n_y: int):
    """Build the same edge graph used by the OC implementation for TV/perimeter."""
    N = n_x * n_y
    graph = nx.Graph()
    graph.add_nodes_from(range(N))

    for k in range(0, N, 2):
        if k + 1 < N:
            graph.add_edge(k, k + 1)
        if (k + 2) % n_y != 0 and k + 3 < N:
            graph.add_edge(k, k + 3)
        if (k // n_y) != 0:
            nb = k - (n_y - 1)
            if nb >= 0:
                graph.add_edge(k, nb)

    return graph


def build_scale(graph: nx.Graph):
    """Assign edge weights matching the OC implementation."""
    scale = np.zeros(len(graph.edges()), dtype=float)
    for k, (u, v) in enumerate(graph.edges()):
        scale[k] = math.sqrt(2.0) if abs(int(u) - int(v)) == 1 else 1.0
    return scale


def compute_tv(control: np.ndarray, graph: nx.Graph, scale: np.ndarray) -> float:
    """Compute a graph-based TV/perimeter metric on a discrete control."""
    diffs = []
    for (u, v), s in zip(graph.edges(), scale):
        diffs.append(s * abs(control[u] - control[v]))
    return float(sum(diffs))

import logging

# Suppress logging
logging.getLogger("FFC").setLevel(logging.ERROR)
logging.getLogger("UFL").setLevel(logging.ERROR)

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False
#Next we define some constants, and the Solid Isotropic Material with Penalisation (SIMP) rule.

p = Constant(3)  # power used in the solid isotropic material
# with penalisation (SIMP) rule, to encourage the control
# solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material

def simp_k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
  rule, equation (11)."""
    return eps + (1 - eps) * a ** p

class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""

    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary
    

def forward(a, f, bc ):
    """Solve the forward problem for a given material distribution a(x)."""
    u = Function(P, name="Temperature")
    v = TestFunction(P)

    F = inner(grad(v), simp_k(a) * grad(u)) * dx - f * v * dx
    print(type(F))
    a, L = lhs(F), rhs(F)
    solve(F == 0, u, bc, solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-5,
                    "maximum_iterations": 20,
                    "report": False,            # disable per-step reports
                    "error_on_nonconvergence": False
                }
            })

    return u

class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

    def __init__(self, V):
        self.V = float(V)
        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)
    
    def function(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)
        integral = self.smass.inner(self.tmpvec.vector())
        # if MPI.rank(MPI.comm_world) == 0:
        #     print("Current control integral: ", integral)
        return [self.V - integral]

    def jacobian(self, m):
        return [-self.smass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Topology Optimization with FEniCS")
    parser.add_argument("--alpha", type=str, required=True, help="Comma-separated list of alpha values (e.g., '1e-5,1e-4,1e-3')")
    parser.add_argument("--mesh-list", type=str, required=True, help="Comma-separated list of mesh sizes (e.g., '64,128,256')")
    parser.add_argument("--source_strength", type=float, default=1.0)
    parser.add_argument("--vol_frac", type=float, default=0.4)
    args = parser.parse_args()
    
    seed = 0
    np.random.seed(seed)

    V = Constant(args.vol_frac)  # volume bound on the control
    src = args.source_strength  # source strength
    
    alpha_list = [float(a) for a in args.alpha.split(',')]
    mesh_list = [int(n) for n in args.mesh_list.split(',')]

    # Outer loop over alpha values
    for alpha_val in alpha_list:
        alpha = Constant(alpha_val)
        
        # Inner loop over mesh sizes
        for n in mesh_list:
            print(f"\n{'='*60}")
            print(f"Running: alpha={alpha_val}, mesh_size={n}", flush=True)
            print(f"{'='*60}\n")
            
            mesh = UnitSquareMesh(n, n)
            A = FunctionSpace(mesh, "DG", 0)  # function space for control
            P = FunctionSpace(mesh, "CG", 1)  # function space for solution
            dx = Measure("dx", domain=mesh)
            f = interpolate(Constant(src), P)  # the volume source term for the PDE
            dS = Measure("dS", domain=mesh)
            eps_tv = 1e-6
            bc = [DirichletBC(P, 0.0, WestNorth())]

            t0 = time.perf_counter()
            a = Function(A)
            a_np = args.vol_frac*np.ones(2*n**2)
            a.vector().set_local(a_np)
            u = forward(a, f, bc)  # solve the forward problem once.
            t_forward = (time.perf_counter() - t0)

            lb = 0.0
            ub = 1.0
        
            F_form  = f*u*dx          
            TV_form = alpha * (sqrt((1+2*eps_tv)*jump(a)**2 + eps_tv) - eps_tv) * dS

            AL_form = F_form + TV_form
            J = assemble(AL_form)            # assemble ONCE

            m  = Control(a)
            Jhat = ReducedFunctional(J, m)

            t0 = time.perf_counter()
            problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))

            parameters={
                "acceptable_tol":     1e-3,
                "maximum_iterations": 30,
                "print_level":        0,    
                "file_print_level":   0    
            }
            solver = IPOPTSolver(problem, parameters=parameters)
            a_opt = solver.solve()
            tot_time = (time.perf_counter() - t0)
            t_opt = (time.perf_counter() - t0)
            
            # Discretize a_opt
            K = int(2 * n * n * args.vol_frac)
            cont_a = a_opt.vector().get_local()
            idx_sorted = np.argsort(cont_a)[::-1]
            top_idx = idx_sorted[:K]
            a_disc = np.zeros_like(cont_a, dtype=int)
            a_disc[top_idx] = 1
            
            # Solve forward problem with discrete control
            a_func = Function(A)
            a_func.vector().set_local(a_disc)
            u_disc = forward(a_func, f, bc)

            # Compute a graph-based perimeter on the final discrete control.
            graph = build_graph(n, 2 * n)
            scale = build_scale(graph)
            perimeter = compute_tv(np.asarray(a_disc, dtype=float), graph, scale)
            
            # Compute objectives for continuous and discrete
            u_opt = forward(a_opt, f, bc)
            
            F_form_cont = f * u_opt * dx
            TV_form_cont = alpha * sqrt(jump(a_opt)**2 + eps_tv) * dS

            comp_cont = assemble(F_form_cont)
            tv_cont = assemble(TV_form_cont) / alpha
            obj_cont = comp_cont + assemble(TV_form_cont)
            
            F_form_disc = f * u_disc * dx
            TV_form_disc = alpha * sqrt(jump(a_func)**2 + eps_tv) * dS
            comp_disc = assemble(F_form_disc)
            tv_disc = assemble(TV_form_disc) / alpha
            obj_disc = comp_disc + assemble(TV_form_disc)
            
            # Print results
            print(f"Mesh size: {n} x {n}", flush=True)
            print(f"    Continuous: obj={np.float64(obj_cont):.6f}, comp={np.float64(comp_cont):.6f}, TV={np.float64(tv_cont):.6f}", flush=True)
            print(f"    Discrete:   obj={np.float64(obj_disc):.6f}, comp={np.float64(comp_disc):.6f}, TV={np.float64(tv_disc):.6f}", flush=True)
            print(f"    Runtime: {np.float64(tot_time):.2f}s", flush=True)
            
            # Save data to h5py
            save_dir = f"fenics_model_tri_{eps_tv}/{alpha_val}"
            os.makedirs(save_dir, exist_ok=True)
            
            h5_path = f"{save_dir}/{n}.h5"
            # Save to h5py
            with h5py.File(h5_path, "a") as h5f:
                summary = h5f.require_group("summary")

                # wipe previous content (safe for reruns)
                for key in list(summary.keys()):
                    del summary[key]
                for key in list(summary.attrs.keys()):
                    del summary.attrs[key]

                # Metadata (force native scalars)
                summary.attrs["dim"] = np.int64(n)
                summary.attrs["alpha"] = np.float64(alpha)
                summary.attrs["V_frac"] = np.float64(V)

                # Runtime information
                summary.attrs["runtime_total"] = np.float64(tot_time)
                # summary.attrs["runtime_solver"] = np.float64(t_solver)
                # summary.attrs["runtime_disc"] = np.float64(t_disc)

                # helper
                def save(name, data):
                    if name in summary:
                        del summary[name]
                    summary.create_dataset(name, data=data)

                # Continuous solution values
                save("cont_objective", np.float64(obj_cont))
                save("cont_TV", np.float64(tv_cont))
                save("cont_compliance", np.float64(comp_cont))

                # Discrete solution values
                save("disc_objective", np.float64(obj_disc))
                save("disc_TV", np.float64(tv_disc))
                save("disc_compliance", np.float64(comp_disc))
                save("perimeter", np.float64(perimeter))
                save("tv_disc", np.float64(perimeter))

                # Controls and solutions (ensure numeric arrays)
                save("a_opt",  a_opt.vector().get_local())
                save("a_disc", np.asarray(a_disc, dtype=np.float64))
                save("u_opt",  u_opt.vector().get_local())
                save("u_disc", u_disc.vector().get_local())

            print(f"    Continuous: obj={np.float64(obj_cont):.6f}, comp={np.float64(comp_cont):.6f}, TV={np.float64(tv_cont):.6f}")
            print(f"    Discrete:   obj={np.float64(obj_disc):.6f}, comp={np.float64(comp_disc):.6f}, TV={np.float64(tv_disc):.6f}")
            print(f"    Runtime: total={np.float64(tot_time):.2f}s")
