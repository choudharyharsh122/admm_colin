"""
OC-based PDE-constrained topology optimization solver.

Implements design update for arbitrary meshes and boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def cubic_roots_cardano(a, b, c, d, tol=1e-14):
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    if not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("All coefficient arrays must have the same shape.")
    if np.any(np.abs(a) < tol):
        raise ValueError("Leading coefficient 'a' must be nonzero.")

    A = b / a
    B = c / a
    C = d / a

    p = B - A**2 / 3
    q = 2*A**3 / 27 - A*B / 3 + C

    Delta = (q/2)**2 + (p/3)**3
    sqrtDelta = np.sqrt(Delta)

    z1 = -q/2 + sqrtDelta
    z2 = -q/2 - sqrtDelta

    u = np.power(z1, 1/3)
    v = np.zeros_like(u)

    mask_u = np.abs(u) > tol
    v[mask_u] = -p[mask_u] / (3*u[mask_u])

    mask_small_u = ~mask_u
    if np.any(mask_small_u):
        v_alt = np.power(z2[mask_small_u], 1/3)
        u_alt = np.zeros_like(v_alt)

        mask_v = np.abs(v_alt) > tol
        u_alt[mask_v] = -p[mask_small_u][mask_v] / (3*v_alt[mask_v])

        u[mask_small_u] = u_alt
        v[mask_small_u] = v_alt

    omega = -0.5 + 0.5j*np.sqrt(3)
    omega2 = -0.5 - 0.5j*np.sqrt(3)

    shift = A / 3

    r1 = u + v - shift
    r2 = omega*u + omega2*v - shift
    r3 = omega2*u + omega*v - shift

    return r1, r2, r3


@dataclass
class MeshData:
    """Abstract mesh description for a PDE discretization."""
    coords: np.ndarray  # shape (n_nodes, dim)
    elems: np.ndarray  # shape (n_elems, nodes_per_elem)
    elem_types: Optional[np.ndarray] = None  # optional element type labels
    boundary_edges: Optional[Dict[str, np.ndarray]] = None  # named boundary edge sets


@dataclass
class BoundaryCondition:
    """Abstract boundary condition descriptor."""
    dirichlet_nodes: Optional[np.ndarray] = None
    dirichlet_values: Optional[np.ndarray] = None
    neumann_edges: Optional[np.ndarray] = None
    neumann_values: Optional[np.ndarray] = None


@dataclass
class MaterialInterpolation:
    """Parameters for material interpolation k(b)."""
    penal: float
    eps: float


class Subproblem1Solver:
    """Concrete implementation of the abstract PDE/OC solver."""

    def __init__(
        self,
        mesh: MeshData,
        bc: BoundaryCondition,
        f: np.ndarray,
        volfrac: float,
        material: MaterialInterpolation,
        move: float = 0.2,
        tol: float = 1e-1,
        maxiter: int = 20,
    ) -> None:
        """Initialize the solver with mesh, BCs, material, and load data."""
        self.mesh = mesh
        self.bc = bc
        self.f = np.asarray(f, dtype=float)
        self.volfrac = float(volfrac)
        self.material = material
        self.move = float(move)
        self.tol = float(tol)
        self.maxiter = int(maxiter)

        # Extract mesh info
        self.coords = np.asarray(mesh.coords, dtype=float)
        self.elems = np.asarray(mesh.elems, dtype=int)
        self.elem_types = mesh.elem_types
        
        self.n_nodes = self.coords.shape[0]
        self.n_elems = self.elems.shape[0]
        self.nodes_per_elem = self.elems.shape[1]
        self.v = np.ones(self.n_elems, dtype=float)
        self.last_bisection_history = []

        # Extract boundary condition info
        self.dirichlet_nodes = (
            np.asarray(bc.dirichlet_nodes, dtype=int)
            if bc.dirichlet_nodes is not None
            else np.array([], dtype=int)
        )
        self.dirichlet_values = (
            np.asarray(bc.dirichlet_values, dtype=float)
            if bc.dirichlet_values is not None
            else np.zeros_like(self.dirichlet_nodes, dtype=float)
        )

        # Compute free (unconstrained) DOFs
        all_dofs = np.arange(self.n_nodes, dtype=int)
        self.freedofs = np.setdiff1d(all_dofs, self.dirichlet_nodes)

        # Build stiffness matrices and load vector
        self.build_stiffness_matrices()
        self.F = self.assemble_load_vector(self.f)

    @classmethod
    def from_mesh_generator(
        cls,
        mesh_generator: Callable[..., MeshData],
        bc_builder: Callable[[MeshData], BoundaryCondition],
        *args: Any,
        **kwargs: Any,
    ) -> "Subproblem1Solver":
        """Factory method: build solver from mesh and BC generators.
        
        Parameters
        ----------
        mesh_generator : callable
            Function returning MeshData.
        bc_builder : callable
            Function that takes MeshData and returns BoundaryCondition.
        *args, **kwargs
            Arguments for mesh_generator and solver initialization.
        
        Returns
        -------
        solver : Subproblem1SolverImpl
        """
        # Extract mesh generator args
        mesh_kwargs = {}
        solver_kwargs = {}
        
        # Separate known solver kwargs
        solver_param_names = {"f", "volfrac", "material", "move", "tol", "maxiter"}
        for key, val in kwargs.items():
            if key in solver_param_names:
                solver_kwargs[key] = val
            else:
                mesh_kwargs[key] = val
        
        # Generate mesh and BCs
        mesh = mesh_generator(*args, **mesh_kwargs)
        bc = bc_builder(mesh)
        
        # Create and return solver
        return cls(mesh=mesh, bc=bc, **solver_kwargs)

    def build_stiffness_matrices(self) -> None:
        """Compute element stiffness matrices for the entire mesh.
        
        This method computes the local stiffness matrix for each element
        based on element coordinates and element type.
        
        For simplicity, this assumes linear triangular or bilinear quad elements.
        The stiffness is computed from the gradient of basis functions.
        """
        # Initialize storage for element stiffness matrices
        # For linear triangles/quads, local stiffness is typically 3x3 or 4x4
        self.KE_all = np.zeros((self.n_elems, self.nodes_per_elem, self.nodes_per_elem), dtype=float)

        # Compute local stiffness for each element
        for e in range(self.n_elems):
            elem_nodes = self.elems[e]
            elem_coords = self.coords[elem_nodes]
            
            # Compute element stiffness matrix
            self.KE_all[e] = self._compute_element_stiffness(elem_coords)

    def _compute_element_stiffness(self, elem_coords: np.ndarray) -> np.ndarray:
        """Compute the reference element stiffness matrix.
        
        For a linear triangle with 3 nodes, this uses the standard FEM formula:
        KE_ij = ∫∫ ∇N_i · ∇N_j dA
        
        Parameters
        ----------
        elem_coords : np.ndarray
            Coordinates of element nodes, shape (n_nodes_per_elem, 2)
        
        Returns
        -------
        KE : np.ndarray
            Local stiffness matrix, shape (n_nodes_per_elem, n_nodes_per_elem)
        """
        n = len(elem_coords)
        
        if n == 3:
            # Linear triangular element
            x1, y1 = elem_coords[0]
            x2, y2 = elem_coords[1]
            x3, y3 = elem_coords[2]
            
            # Area of triangle
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area <= 0:
                raise ValueError("Element must have positive area (check orientation)")
            
            # Gradients of linear basis functions (constant on element)
            # For linear triangle: N_i = a_i + b_i*x + c_i*y
            b = np.array([y2 - y3, y3 - y1, y1 - y2])  # dc/dx
            c = np.array([x3 - x2, x1 - x3, x2 - x1])  # dc/dy
            
            # B matrix (gradient matrix)
            B = np.vstack([b, c]) / (2.0 * area)
            
            # Stiffness: KE = B^T B * area
            KE = area * (B.T @ B)
            
        # elif n == 4:
        #     # Bilinear quad element (reference: [-1,1]^2)
        #     # For simplicity, use 2x2 Gauss quadrature
        #     # This is a simplified implementation; full implementation would use
        #     # proper isoparametric mapping
        #     
        #     # Assume reference quad [-1,1]^2 and map to actual coordinates
        #     # Gauss points and weights
        #     gp = 1.0 / np.sqrt(3)
        #     xi_eta = np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])
        #     w = 1.0  # weight for each gauss point
        #     
        #     KE = np.zeros((4, 4), dtype=float)
        #     
        #     for xi, eta in xi_eta:
        #         # Jacobian and B matrix at this Gauss point
        #         J, B = self._isoparametric_quad_basis(elem_coords, xi, eta)
        #         det_J = np.linalg.det(J)
        #         
        #         if det_J <= 0:
        #             raise ValueError("Invalid Jacobian (check element orientation)")
        #         
        #         KE += w * det_J * (B.T @ B)
            
        else:
            raise ValueError(f"Unsupported element type with {n} nodes (currently only triangles supported)")
        
        return KE

    # def _isoparametric_quad_basis(
    #     self, elem_coords: np.ndarray, xi: float, eta: float
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Compute Jacobian and B matrix for bilinear quad at (xi, eta).
    #     
    #     Parameters
    #     ----------
    #     elem_coords : np.ndarray
    #         Quad node coordinates, shape (4, 2)
    #     xi, eta : float
    #         Reference element coordinates in [-1, 1]
    #     
    #     Returns
    #     -------
    #     J : np.ndarray
    #         Jacobian matrix, shape (2, 2)
    #     B : np.ndarray
    #         Gradient matrix, shape (2, 4)
    #     """
    #     # Bilinear basis function derivatives w.r.t. xi, eta
    #     dN_dxi = np.array([
    #         -(1 - eta) / 4,
    #         (1 - eta) / 4,
    #         (1 + eta) / 4,
    #         -(1 + eta) / 4,
    #     ])
    #     dN_deta = np.array([
    #         -(1 - xi) / 4,
    #         -(1 + xi) / 4,
    #         (1 + xi) / 4,
    #         (1 - xi) / 4,
    #     ])
    #     
    #     # Jacobian: J = [dx/dxi, dx/deta; dy/dxi, dy/deta]
    #     dx_dxi = dN_dxi @ elem_coords[:, 0]
    #     dx_deta = dN_deta @ elem_coords[:, 0]
    #     dy_dxi = dN_dxi @ elem_coords[:, 1]
    #     dy_deta = dN_deta @ elem_coords[:, 1]
    #     
    #     J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
    #     
    #     # Inverse Jacobian
    #     J_inv = np.linalg.inv(J)
    #     
    #     # Gradients in physical coordinates: dN/dx = dN/dxi * dxi/dx + ...
    #     dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
    #     dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
    #     
    #     B = np.vstack([dN_dx, dN_dy])
    #     
    #     return J, B

    def assemble_load_vector(self, f: np.ndarray) -> np.ndarray:
        """Assemble the global load vector from distributed source term.
        
        Parameters
        ----------
        f : np.ndarray
            Source term at nodes, shape (n_nodes,)
        
        Returns
        -------
        F : np.ndarray
            Global load vector, shape (n_nodes,)
        """
        f = np.asarray(f, dtype=float)
        F = np.zeros(self.n_nodes, dtype=float)
        
        # Add element contributions
        for e in range(self.n_elems):
            elem_nodes = self.elems[e]
            elem_coords = self.coords[elem_nodes]
            
            # Element area (for simple assembly)
            if self.nodes_per_elem == 3:
                x1, y1 = elem_coords[0]
                x2, y2 = elem_coords[1]
                x3, y3 = elem_coords[2]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            # elif self.nodes_per_elem == 4:
            #     # Approximate area for quad (simplified)
            #     area = self._quad_area(elem_coords)
            else:
                area = 1.0
            
            # Local load vector: f_e = f(elem_nodes) * area / n_nodes
            # This assumes constant source over element and equal distribution to nodes
            f_e = (f[elem_nodes] * area / self.nodes_per_elem)
            
            # Assemble into global vector
            F[elem_nodes] += f_e
        
        return F

    # def _quad_area(self, quad_coords: np.ndarray) -> float:
    #     """Compute area of a bilinear quad element (simplified)."""
    #     # Use Shoelace formula for the quadrilateral
    #     x = quad_coords[:, 0]
    #     y = quad_coords[:, 1]
    #     area = 0.5 * abs(
    #         x[0] * (y[1] - y[3]) +
    #         x[1] * (y[2] - y[0]) +
    #         x[2] * (y[3] - y[1]) +
    #         x[3] * (y[0] - y[2])
    #     )
    #     return area

    def solve_state(self, b: np.ndarray) -> np.ndarray:
        """Solve the PDE state for a given control field b.
        
        Solves: K(b) U = F, with U = 0 on Dirichlet boundaries.
        
        Parameters
        ----------
        b : np.ndarray
            Control/density field per element, shape (n_elems,)
        
        Returns
        -------
        U : np.ndarray
            State vector at all nodes, shape (n_nodes,)
        """
        b = np.asarray(b, dtype=float).flatten()
        
        if len(b) != self.n_elems:
            raise ValueError(f"Control vector size {len(b)} != n_elems {self.n_elems}")
        
        # Material interpolation: k(b) = eps + (1 - eps) * b^penal
        k_b = self.material.eps + (1.0 - self.material.eps) * np.power(
            np.maximum(b, 1e-5), self.material.penal
        )
        
        # Assemble global stiffness matrix K(b)
        K_data = []
        K_row = []
        K_col = []
        
        for e in range(self.n_elems):
            elem_nodes = self.elems[e]
            KE_scaled = k_b[e] * self.KE_all[e]
            
            # Add to COO format
            for i in range(self.nodes_per_elem):
                for j in range(self.nodes_per_elem):
                    K_data.append(KE_scaled[i, j])
                    K_row.append(elem_nodes[i])
                    K_col.append(elem_nodes[j])
        
        K = coo_matrix(
            (K_data, (K_row, K_col)),
            shape=(self.n_nodes, self.n_nodes)
        ).tocsc()
        
        # Solve on free DOFs: K_ff U_f = F_f
        K_ff = K[self.freedofs][:, self.freedofs]
        F_f = self.F[self.freedofs]
        
        U_f = spsolve(K_ff, F_f)
        
        # Construct full solution with boundary conditions
        U = np.zeros(self.n_nodes, dtype=float)
        U[self.freedofs] = U_f
        U[self.dirichlet_nodes] = self.dirichlet_values
        
        return U

    def compute_element_energy(self, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute element-wise strain energy.
        
        For each element: ce_e = U_e^T * KE_e * U_e
        
        Parameters
        ----------
        U : np.ndarray
            State vector, shape (n_nodes,)
        b : np.ndarray
            Control field, shape (n_elems,) [unused here but kept for interface]
        
        Returns
        -------
        ce : np.ndarray
            Per-element energy, shape (n_elems,)
        """
        U = np.asarray(U, dtype=float)
        ce = np.zeros(self.n_elems, dtype=float)
        
        for e in range(self.n_elems):
            elem_nodes = self.elems[e]
            U_e = U[elem_nodes]
            
            # Energy: ce_e = U_e^T * KE_e * U_e
            ce[e] = U_e @ self.KE_all[e] @ U_e
        
        return ce

    def update_design(
        self,
        a: np.ndarray,
        b: np.ndarray,
        lam: np.ndarray,
        rho: float,
        ce: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Perform one OC (optimality criteria) update step.
        
        Uses cubic root formula with volume constraint via Lagrange multiplier.
        
        Parameters
        ----------
        a : np.ndarray
            ADMM primal variable, shape (n_elems,)
        b : np.ndarray
            Current design, shape (n_elems,)
        lam : np.ndarray
            ADMM Lagrange multiplier, shape (n_elems,)
        rho : float
            ADMM penalty parameter
        ce : np.ndarray
            Element energy/sensitivity, shape (n_elems,)
        
        Returns
        -------
        b_new : np.ndarray
            Updated design, shape (n_elems,)
        mu : float
            Lagrange multiplier for volume constraint
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        lam = np.asarray(lam, dtype=float)
        ce = np.asarray(ce, dtype=float)
        
        gprime = (1.0 - self.material.eps) * self.material.penal * np.maximum(b, 1e-5)**(self.material.penal - 1)
        numer = np.maximum(gprime * ce, 0.0)

        mu_low = -1e5
        mu_high = 1e5

        def trial_update(mu):
            c0 = -gprime * ce * b**2
            c1 = np.zeros_like(c0)
            c2 = (lam + mu * self.v - rho * a) / self.n_elems
            c3 = (rho / self.n_elems) * np.ones_like(c0)

            b_candidate = np.real(cubic_roots_cardano(c3, c2, c1, c0)[0])
            b_new = np.maximum(
                self.material.eps,
                np.maximum(
                    b - self.move,
                    np.minimum(1.0, np.minimum(b + self.move, b_candidate)),
                ),
            )
            return b_new

        b_test = trial_update(mu_high)
        while b_test.mean() > self.volfrac:
            mu_high *= 2.0
            b_test = trial_update(mu_high)
            if mu_high > 1e16:
                break

        bisection_history = []
        while (mu_high - mu_low) > 1e-6:
            mu_mid = 0.5 * (mu_low + mu_high)
            b_mid = trial_update(mu_mid)
            b_mid_mean = float(b_mid.mean())
            bisection_history.append((float(mu_mid), b_mid_mean))

            if b_mid_mean > self.volfrac:
                mu_low = mu_mid
            else:
                mu_high = mu_mid

        self.last_bisection_history = bisection_history
        mu = 0.5 * (mu_low + mu_high)
        b_new = trial_update(mu)
        return b_new, mu

    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        lam: np.ndarray,
        rho: float,
        track_oc_convergence: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Solve the control subproblem via OC iterations.
        
        Parameters
        ----------
        a : np.ndarray
            ADMM primal variable
        b : np.ndarray
            Initial design
        lam : np.ndarray
            ADMM Lagrange multiplier
        rho : float
            ADMM penalty parameter
        track_oc_convergence : bool
            If True, track convergence metrics
        
        Returns
        -------
        b : np.ndarray
            Final design
        U : np.ndarray
            Final state
        track_data : dict or None
            Convergence tracking data
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float).copy()
        lam = np.asarray(lam, dtype=float)
        
        track_data_dict = {
            "F_list": [],
            "grad_F_list": [],
            "grad_F_norm_list": [],
        }
        
        converged = False
        loop = 0
        change = np.inf
        while change > self.tol and loop < self.maxiter:
            b_old = b.copy()
            
            # Solve PDE state
            U = self.solve_state(b)
            
            # Compute element energy
            ce = self.compute_element_energy(U, b)
            
            # Update design
            b, mu = self.update_design(a, b, lam, rho, ce)
            
            # Check convergence
            change = np.max(np.abs(b - b_old))
            loop += 1
            
            if track_oc_convergence:
                # Track convergence metrics
                obj_val, _, _, _ = self.compute_objective(a, b, lam, rho)
                track_data_dict["F_list"].append(float(obj_val))
            
            if change < self.tol:
                converged = True
                break
        
        # Final state solve
        U = self.solve_state(b)
        
        # Prepare return data
        if track_oc_convergence:
            track_data = {
                "F_list": np.array(track_data_dict["F_list"], dtype=float),
                "grad_F_list": np.zeros((0, len(b))),  # Simplified
                "grad_F_norm_list": np.array([], dtype=float),
            }
        else:
            track_data = None
        
        return b, U, track_data

    def compute_objective(
        self,
        a: np.ndarray,
        b: np.ndarray,
        lam: np.ndarray,
        rho: float,
    ) -> Tuple[float, float, float, Optional[float]]:
        """Compute the augmented objective.
        
        obj = compliance + lambda^T (b - a) + (rho/2) ||b - a||^2
        
        Parameters
        ----------
        a, b, lam : np.ndarray
            ADMM variables
        rho : float
            ADMM penalty parameter
        
        Returns
        -------
        obj : float
            Full augmented objective
        compliance : float
            PDE compliance term F^T U
        penalty : float
            Quadratic penalty term
        extra : float or None
            Additional term (unused, None)
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        lam = np.asarray(lam, dtype=float)
        
        # Solve PDE
        U = self.solve_state(b)
        
        # Compliance: F^T U
        compliance = float(self.F @ U)
        
        # ADMM penalty: lambda^T (b - a) + (rho/2) ||b - a||^2
        diff = b - a
        lag_term = float(lam @ diff)
        pen_term = float((rho / 2.0) * (diff @ diff))
        
        obj = compliance + lag_term + pen_term
        
        return obj, compliance, pen_term, None


# ============================================================================
# HELPER FUNCTIONS: Mesh Generation and BC Builder
# ============================================================================

def generate_unit_square_mesh(dim: int) -> MeshData:
    """Generate a unit square triangular mesh.
    
    Parameters
    ----------
    dim : int
        Number of elements per side.
    
    Returns
    -------
    mesh : MeshData
        Mesh object with coordinates, connectivity, and boundary edges.
    """
    h = 1.0 / dim
    nn = dim + 1  # number of nodes per side
    
    # Node coordinates
    coords = np.zeros((nn * nn, 2), dtype=float)
    for j in range(nn):
        for i in range(nn):
            coords[i + j * nn] = [i * h, j * h]
    
    # Element connectivity (2 triangles per square cell)
    elems = []
    elem_types = []
    boundary_edges = {
        'north': [],
        'south': [],
        'east': [],
        'west': [],
    }
    
    node_ids = np.arange(nn * nn).reshape((nn, nn))
    
    for j in range(dim):
        for i in range(dim):
            n0 = node_ids[j, i]
            n1 = node_ids[j, i + 1]
            n2 = node_ids[j + 1, i]
            n3 = node_ids[j + 1, i + 1]
            
            # Triangle 1
            elems.append([n0, n1, n3])
            elem_types.append(0)
            
            # Triangle 2
            elems.append([n0, n3, n2])
            elem_types.append(1)
            
            # Track boundary edges
            if j == 0:
                boundary_edges['south'].append([n0, n1])
            if j == dim - 1:
                boundary_edges['north'].append([n2, n3])
            if i == 0:
                boundary_edges['west'].append([n0, n2])
            if i == dim - 1:
                boundary_edges['east'].append([n1, n3])
    
    return MeshData(
        coords=np.array(coords, dtype=float),
        elems=np.array(elems, dtype=int),
        elem_types=np.array(elem_types, dtype=int),
        boundary_edges={k: np.array(v, dtype=int) for k, v in boundary_edges.items()},
    )


def build_dirichlet_bc_from_config(
    mesh: MeshData,
    dirichlet_boundaries: list,
    bc_values: dict,
) -> BoundaryCondition:
    """Build BC from mesh and specified Dirichlet boundaries.
    
    Parameters
    ----------
    mesh : MeshData
        Mesh object with boundary_edges dict.
    dirichlet_boundaries : list of str
        e.g., ['north', 'south', 'west']
    bc_values : dict
        e.g., {'north': 0.0, 'south': 0.0, 'west': 0.0}
    
    Returns
    -------
    bc : BoundaryCondition
        Boundary condition object.
    """
    dirichlet_nodes = []
    dirichlet_values = []
    
    for boundary_name in dirichlet_boundaries:
        if mesh.boundary_edges and boundary_name in mesh.boundary_edges:
            boundary_node_indices = np.unique(mesh.boundary_edges[boundary_name].flatten())
            value = bc_values.get(boundary_name, 0.0)
            
            dirichlet_nodes.extend(boundary_node_indices)
            dirichlet_values.extend([value] * len(boundary_node_indices))
    
    dirichlet_nodes = np.array(list(set(dirichlet_nodes)), dtype=int)
    
    # Map values back to nodes
    node_to_value = {}
    for node, val in zip(dirichlet_nodes, dirichlet_values):
        node_to_value[node] = val
    
    dirichlet_values = np.array([node_to_value[n] for n in dirichlet_nodes], dtype=float)
    
    return BoundaryCondition(
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=dirichlet_values,
    )
