import os
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyenv
import math
import argparse
import networkx as nx
import h5py
import time
from dolfin import *
import sys
sys.stdout.reconfigure(line_buffering=True)



def cell_idx_2d_to_1d(I, J, tri, n):
    """
    Convert (I, J, tri) into a 1D index.
    tri ∈ {'L','U'}
    """
    base = I + J * n
    if tri == 'L':
        return 2 * base
    else:  # 'U'
        return 2 * base + 1

def cell_idx_1d_to_2d(idx, n):
    """
    Convert 1D triangle index into (I, J, tri).
    tri ∈ {'L','U'}
    """
    tri = 'L' if (idx % 2) == 0 else 'U'
    k = idx // 2
    I = k % n
    J = k // n
    return (I, J, tri)

def node_idx_2d_to_1d(i, j, n):
    """Map node (i,j) → global 1D node index."""
    return i + j * (n+1)

def node_idx_1d_to_2d(idx, n):
    """Map global 1D node index → (i, j) 2D grid index."""
    i = idx % (n + 1)          # column / x-index
    j = idx // (n + 1)         # row / y-index
    return i, j

def nbhd(i, j, n_max):
    nodes = []
    for i_ in [i-1, i, i+1]:
        for j_ in [j-1, j, j+1]:
            if i_ < 0 or i_ > n_max or j_ < 0 or j_ > n_max:
                continue
            nodes.append((i_, j_))
    return nodes

def nbhd_tri(i, j, n):
    """
    Return the valid neighboring nodes of (i,j) including itself.

    Node grid is (n+1) x (n+1).
    """
    nbrs = []
    candidates = [
        (i, j),
        (i-1, j),
        (i+1, j),
        (i, j-1),
        (i, j+1),
        (i+1, j+1),
        (i-1, j-1)
    ]

    for ip, jp in candidates:
        if 0 <= ip <= n and 0 <= jp <= n:
            nbrs.append((ip, jp))

    return nbrs

def get_cell_nbhd(i, j, ip, jp, n):
    """
    Return all triangular cells (I, J, tri) that touch the entity
    defined by the two nodes (i,j) and (ip,jp) in an n x n triangular mesh.

    tri \in {'L','U'}

    Cases:
      1. Vertex:     (i,j) == (ip,jp)
      2. Diagonal:   |i-ip|=1 and |j-jp|=1 (only along y=x)
      3. Vertical:   |i-ip|=1 and j=jp
      4. Horizontal: i=ip and |j-jp|=1
    """

    cells = []

    # Case 1: Vertex case
    if i == ip and j == jp:
        # 6 candidate nodes to select from:
        candidates = [
            (i-1, j-1, 'L'),
            (i-1, j-1, 'U'),
            (i-1, j,   'L'),
            (i,   j-1, 'U'),
            (i,   j,   'L'),
            (i,   j,   'U'),
        ]

        for I, J, tri in candidates:
            if 0 <= I < n and 0 <= J < n:
                cells.append((I, J, tri))

        return cells

    # Differences
    di = ip - i
    dj = jp - j

    # Case 2: Diagonal edge
    # |di|=1 and |dj|=1 and diagonal along y=x
    if abs(di) == 1 and abs(dj) == 1:
        if di == dj:
            I = min(i, ip)
            J = min(j, jp)
            if 0 <= I < n and 0 <= J < n:
                cells.append((I, J, 'L'))
                cells.append((I, J, 'U'))
        return cells

    # Case 3: Vertical edge
    # |di|=1 and j == jp
    if abs(dj) == 1 and i == ip:
        j0 = min(j, jp)
        i0 = i

        # Two cells attached to this vertical edge:
        #   L(i0, j0-1),  U(i0, j0)
        if 0 <= i0-1 < n and 0 <= j0 < n:
            cells.append((i0-1, j0, 'L'))

        if 0 <= i0 < n and 0 <= j0 < n:
            cells.append((i0, j0, 'U'))

        return cells

    # Case 4: Horizontal edge
    # i == ip and |dj|=1
    if j == jp and abs(di) == 1:
        j0 = j
        i0 = min(i, ip)

        # Two cells touching this horizontal edge:
        #   U(i0-1, j0),  L(i0, j0)
        if 0 <= j0-1 < n and 0 <= i0 < n:
            cells.append((i0, j0-1, 'U'))

        if 0 <= i0 < n and 0 <= j0 < n:
            cells.append((i0, j0, 'L'))

        return cells

    # No valid shared entity
    return cells


def cell_nbhd(i, j, n_max):
    n_s_domain = []
    if i > 0 and j > 0:
        n_s_domain.append((i-1, j-1))
    if i < n_max and j > 0:
        n_s_domain.append((i, j-1))
    if i > 0 and j < n_max:
        n_s_domain.append((i-1, j))
    if i < n_max and j < n_max:
        n_s_domain.append((i, j))
    return n_s_domain

def cell_nbhd_tri(i, j, n):
    """
    Return all triangular cells (I, J, tri) that touch the node
    (i,j) in an n x n triangular mesh.

    tri \in {'L','U'}

    """
    cells = []

    # 6 candidate nodes to select from:
    candidates = [
        (i-1, j-1, 'L'),
        (i-1, j-1, 'U'),
        (i-1, j,   'L'),
        (i,   j-1, 'U'),
        (i,   j,   'L'),
        (i,   j,   'U'),
    ]

    for I, J, tri in candidates:
        if 0 <= I < n and 0 <= J < n:
            cells.append((I, J, tri))

    return cells


def A(i, j, i_, j_):
    dist = abs(i - i_) + abs(j - j_)
    if dist == 0:
        return 2/3
    elif dist == 1:
        return -1/6
    elif dist == 2:
        return -1/3
    else:
        return 0

def A_tri(cell, i, j, ip, jp, h):
    
    I, J, tri = cell
    
    if (((ip,jp)) not in nbhd_tri(i, j, n)):
        #print(f"({i},{j}), ({ip},{jp}) not neighbors")
        return 0
    
    else:
        di = abs(i - ip)
        dj = abs(j - jp)

        # Case 1: same node pairing
        if i == ip and j == jp:
            if I == i and J == j - 1:
                return 1.0 
            elif I == i - 1 and J == j:
                return 1.0 
            else:
                return 0.5 

        # Case 2: vertical adjacent nodes pairing: |i-i'|=1, |j-j'|=0
        if di == 1 and dj == 0:
            return -0.5

        # Case 3: horizontal adjacent nodes pairing: |i-i'|=0, |j-j'|=1
        if di == 0 and dj == 1:
            return -0.5 

        # Case 4: diagonal adjacent nodes paring: |i-i'|=1 and |j-j'|=1
        if di == 1 and dj == 1:
            return 0.0

        return 0.0

    
    

def overlap(i, j, i_, j_, n_max):
    return set(cell_nbhd(i, j, n_max)) & set(cell_nbhd(i_, j_, n_max))

def overlap_tri(i, j, i_, j_, n):
    return set(cell_nbhd_tri(i, j, n)) & set(cell_nbhd_tri(i_, j_, n))

def get_model_compliance(u, n, f, h):
    obj_expr = 0.0
    tri_area_weight = f * (h*h / 6.0)   # f * (A_T/3)

    for I in range(n):
        for J in range(n):

            # -------- Lower triangle L(I,J) --------
            # vertices: (I,J), (I+1,J), (I+1,J+1)
            vL = [
                (I,   J),
                (I+1, J),
                (I+1, J+1)
            ]

            for (i, j) in vL:
                k = node_idx_2d_to_1d(i, j, n)
                obj_expr += tri_area_weight * u[k]

            # -------- Upper triangle U(I,J) --------
            # vertices: (I,J), (I+1,J+1), (I,J+1)
            vU = [
                (I,   J),
                (I+1, J+1),
                (I,   J+1)
            ]

            for (i, j) in vU:
                k = node_idx_2d_to_1d(i, j, n)
                obj_expr += tri_area_weight * u[k]
    return obj_expr

def get_model_TV(model, edges, n, alpha):

    return (alpha/n) * sum(w * model.d[c_idx, c_idx_, w] for (c_idx, c_idx_, w) in edges)

def get_edge_list(n):
    edges = []
    for i in range(n):
        for j in range(n):
            # 1. Internal adjacency
            c_idx  = cell_idx_2d_to_1d(i, j, 'L', n)
            c_idx_ = cell_idx_2d_to_1d(i, j, 'U', n)
            edges.append((c_idx, c_idx_, math.sqrt(2)))

            # 2. Vertical adjacency: L(i,j) ↔ L(i, j+1)
            if j + 1 < n:
                c_idx  = cell_idx_2d_to_1d(i, j, 'U', n)
                c_idx_ = cell_idx_2d_to_1d(i, j+1, 'L', n)
                edges.append((c_idx, c_idx_, 1))

            if i + 1 < n:
                c_idx  = cell_idx_2d_to_1d(i, j, 'L', n)
                c_idx_ = cell_idx_2d_to_1d(i+1, j, 'U', n)
                edges.append((c_idx, c_idx_, 1))
    return edges

def build_rhs(i, j, n, h):
    
    idx = node_idx_2d_to_1d(i, j, n)

    # --- interior nodes ---
    if 1 <= i <= n-1 and 1 <= j <= n-1:
        #b[i, j] = f * h**2
        b_val = f * h**2

    # --- edge nodes (not corners) ---
    elif (i == 0 or i == n) and (1 <= j <= n-1):
        #b[i, j] = (f * h**2) / 2
        b_val = f * h**2 / 2
    elif (j == 0 or j == n) and (1 <= i <= n-1):
        #b[i, j] = (f * h**2) / 2
        b_val = f * h**2 / 2

    # --- two-triangle corners ---
    elif (i == 0 and j == 0) or (i == n and j == n):
        #b[i, j] = (f * h**2) / 3
        b_val = f * h**2 / 3

    # --- one-triangle corners ---
    elif (i == n and j == 0) or (i == 0 and j == n):
        #b[i, j] = (f * h**2) / 6
        b_val = f * h**2 / 6
    
    return b_val
    
def create_model(n, f, alpha, V, beta, gamma, p=3, continuous=True):
    h = 1/n

    # This may not be big enough, it seemed to work for the parameters I tried though
    M = 10
    
    # This is the domain for the z variables
    n_s_domain = []
    for i in range(n+1):
        for j in range(n+1):
            # To consider: (i-1, j-1), (i-1, j), (i, j-1), (i, j)
            #n_s_domain.extend([(i, j) + cell for cell in cell_nbhd(i, j, n)])
            n_s_domain.extend([(i, j) + cell for cell in cell_nbhd_tri(i, j, n)])

    # The domain for the d variables
    edges = get_edge_list(n)

    # Creating model and variables
    model = pyenv.ConcreteModel()
    model.nodes = [(i,j) for i in range(n+1) for j in range(n+1)]
    model.cells = [(i, j) for i in range(n) for j in range(n)]
    # coefficients in finite element expansion
    #model.c = pyenv.Var(range(n+1), range(n+1), within=pyenv.NonNegativeReals)
    model.c = pyenv.Var(range((n+1)*(n+1)), within=pyenv.NonNegativeReals)
    # controls\
    if continuous:
        #model.a = pyenv.Var(range(n), range(n), within=pyenv.NonNegativeReals, bounds=(0,1))
        model.a = pyenv.Var(range(2*n*n), within=pyenv.NonNegativeReals, bounds=(0,1))
    else:
        model.a = pyenv.Var(range(2*n*n), range(n), within=pyenv.Binary)

    # linearization of total variation terms
    model.d = pyenv.Var(edges, within=pyenv.NonNegativeReals)

    node_dict = {idx: (i,j) for idx, (i,j) in enumerate(model.nodes)} 
    inv_node_dict = {(i,j): idx for idx, (i,j) in enumerate(model.nodes)}

    # Ensures that u solves the pde
    model.finite_element_constraints = pyenv.ConstraintList()
    for i, j in model.nodes:
        expr = 0
        if i == 0 or j == n:
            # Boundary conditions
            idx = node_idx_2d_to_1d(i, j, n)
            model.finite_element_constraints.add(model.c[idx] == 0)            
        else:
            for i_, j_ in nbhd(i, j, n):
                for ((cell_i, cell_j, tri)) in overlap_tri(i, j, i_, j_,n):
                    if continuous:
                        c_idx = cell_idx_2d_to_1d(cell_i, cell_j, tri, n)
                        n_idx = node_idx_2d_to_1d(i_, j_, n)
                        expr += (gamma*model.c[n_idx] + model.c[n_idx]*model.a[c_idx]**p * (beta - gamma)) * A_tri((cell_i, cell_j, tri),i,j,i_,j_,h)
                    else:
                        expr += (gamma*model.c[i_,j_] + model.z[i_,j_,cell_i,cell_j] * (beta - gamma)) * A(i,j,i_,j_)
            #rhs = h**2/4 * len(cell_nbhd(i,j,n))*f
            rhs = build_rhs(i, j, n, h)
            model.finite_element_constraints.add(expr==rhs)


    if not continuous:
        # linearization of term c * a
        model.z = pyenv.Var(n_s_domain, within=pyenv.NonNegativeReals)
        model.z_constraints = pyenv.ConstraintList()
        
        for ni, nj, si, sj in n_s_domain:
            model.z_constraints.add(model.z[ni,nj,si,sj] <= M* model.a[si,sj])
            model.z_constraints.add(model.z[ni,nj,si,sj] >= model.c[ni,nj] - M*(1-model.a[si,sj]))
            model.z_constraints.add(model.z[ni,nj,si,sj] <= model.c[ni,nj])

    model.tv_constraints = pyenv.ConstraintList()
    
    for (c_idx, c_idx_, w) in edges:
        model.tv_constraints.add(model.a[c_idx] - model.a[c_idx_] <= model.d[c_idx, c_idx_, w])
        model.tv_constraints.add(model.a[c_idx] - model.a[c_idx_] >= -model.d[c_idx, c_idx_, w])

    #model.volume_constraint = pyenv.Constraint(expr=sum(model.a[ci,cj] for ci in range(n) for cj in range(n)) <= V*n**2)
    model.volume_constraint = pyenv.Constraint(expr=sum(model.a[k] for k in range(2*n*n)) <= V * (2*n*n))

    

    # model.obj = pyenv.Objective(
    #     expr=obj_expr + (alpha/n) * sum(model.d[e] for e in edges),
    #     sense=pyenv.minimize
    # )
    compliance = get_model_compliance(model.c, n, f, h)
    TV = get_model_TV(model, edges, n, alpha)

    model.obj = pyenv.Objective(
    expr=compliance + TV, sense=pyenv.minimize)

    return model

def create_and_solve_continuous_model(n, alpha, V=0.4, beta=1, gamma=1e-3, f=1e-2, p=5):
    """
    Create and solve the continous relaxation of the model on an nxn rectangular grid. Then threshold
    the solution so that the proportion of cells where a = 1 is <= V.

    Args:
        n (int): The dimensions of the mesh
        f (float): right hand side of pde
        alpha (float): Total variation regularization parameter
        V (float): Upper limit on proportion of cells with a=1
        beta (float): conductivity of material 1
        gamma (float): conductivity of material 2
        p (int): power in the SIMP penalty
    """
    cont_model = create_model(n, f, alpha, V, beta, gamma, p, continuous=True)

    for k in range(2*n*n):
        cont_model.a[k].value = V
        cont_model.a[k].fixed = True

    cont_solver = pyenv.SolverFactory('ipopt')
    cont_solver.options['print_level'] = 5
    cont_solver.solve(cont_model, tee=True)

    for k in range(2*n*n):
        cont_model.a[k].fixed = False

    cont_solver.solve(cont_model, tee=True)

    cont_a = np.array([cont_model.a[k].value for k in range(2*n*n)])
    cont_c = np.array([cont_model.c[k].value for k in range((n+1)*(n+1))])

    threshold = np.sort(cont_a.flatten())[int(n**2 * V)]
    disc_a = (cont_a > threshold).astype(int)
    return cont_a, disc_a
    

def compute_TV(a, n, alpha):
    edges = get_edge_list(n)
    return (alpha / n) * sum(w * abs(a[u] - a[v]) for (u, v, w) in edges)

def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]

def parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def evaluate_at_a(model, a_point, solver_name="ipopt"):
    # 1) Fix a
    for k in model.a:
        model.a[k].fix(a_point[k])   # a_point is dict-like: {k: val}

    # Solve the rest of the problem for fixed a to determine u(a) and J(a)  
    solver = pyenv.SolverFactory(solver_name)
    solver.solve(model, tee=True)

    # 4) Read back c and objective
    c_vals = np.array([model.c[k].value for k in range((n + 1) * (n + 1))])
    obj_val = pyenv.value(model.obj) if hasattr(model, "obj") else None  # see note below
    return obj_val, c_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_list", required=True, type=parse_int_list)     # e.g. 16,32,64
    parser.add_argument("--alpha_list", required=True, type=parse_float_list)  # e.g. 1e-5,1e-6
    parser.add_argument("--source_strength", type=float, default=1.0)
    parser.add_argument("--vol_frac", type=float, default=0.4)
    args = parser.parse_args()

    f = args.source_strength
    V = args.vol_frac
    beta = 1
    gamma = 1e-3

    solver = pyenv.SolverFactory("ipopt")
    solver.options['print_level'] = 5  # Enable IPOPT iteration logs
    seed = 0  # Seed counter for naming h5 groups

    for alpha in args.alpha_list:
        print(f"Running for alpha = {alpha}")
        for n in args.mesh_list:
            h = 1/n
            eps = 1e-3
            p = 5
            print(f"  Mesh size: {n} x {n}")
            t_start = time.time()
            
            # Create h5py file for this mesh
            h5_dir = f"fem_model_tri/{alpha}"
            os.makedirs(h5_dir, exist_ok=True)
            h5_path = f"{h5_dir}/{n}.h5"
            
            print(f"[DEBUG] Creating model for alpha={alpha}, n={n}", flush=True)
            model = create_model(n, f, alpha, V, beta, gamma)
            
            # Time the solver
            t_solver_start = time.time()
            print(f"[DEBUG] Starting IPOPT solve for alpha={alpha}, n={n}", flush=True)
            solver.solve(model, tee=True)
            t_solver = time.time() - t_solver_start

            # Extract continuous solution
            a_opt = np.array([model.a[k].value for k in range(2 * n**2)])
            u_opt = np.array([model.c[k].value for k in range((n + 1) * (n + 1))])
            #comp_cont = get_model_compliance(u_opt, n, f, h)

            obj = pyenv.value(model.obj)

            # Create discrete solution
            K = int(2 * n * n * V)                  # number of 1's allowed
            idx_sorted = np.argsort(a_opt)[::-1]   # sort indices by value descending
            top_idx = idx_sorted[:K]
            a_disc = np.zeros_like(a_opt, dtype=int)
            a_disc[top_idx] = 1

            # Time the discrete solution computation
            t_disc_start = time.time()
            obj_disc, u_disc = evaluate_at_a(model, a_disc, solver_name="ipopt")
            t_disc = time.time() - t_disc_start

            edges = get_edge_list(n)
            #TV_cont = pyenv.value(get_model_TV(model, edges, n, alpha))
            
            # Compute objective values
            comp_cont = get_model_compliance(u_opt, n, f, h)
            comp_disc = get_model_compliance(u_disc, n, f, h)
            
            tv_cont = compute_TV(a_opt, n, alpha)
            tv_disc = compute_TV(a_disc, n, alpha)
            
            obj_cont = comp_cont + tv_cont
            obj_disc = comp_disc + tv_disc

            # Total runtime
            t_total = time.time() - t_start

            # Save to h5py
            with h5py.File(h5_path, "a") as h5f:
                summary = h5f.require_group("summary")

                # (optional) wipe old contents if rerunning
                for k in list(summary.keys()):
                    del summary[k]
                for k in list(summary.attrs.keys()):
                    del summary.attrs[k]
                
                # Metadata
                summary.attrs["dim"] = n
                summary.attrs["alpha"] = alpha
                summary.attrs["V_frac"] = V

                # Runtime information
                summary.attrs["runtime_total"] = t_total
                summary.attrs["runtime_solver"] = t_solver
                summary.attrs["runtime_disc"] = t_disc

                # helper to overwrite datasets
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

                # Save controls and solutions
                save("a_opt", a_opt)
                save("a_disc", a_disc)
                save("u_opt", u_opt)
                save("u_disc", u_disc)
            
            print(f"    Continuous: obj={np.float64(obj_cont):.6f}, comp={np.float64(comp_cont):.6f}, TV={np.float64(tv_cont):.6f}")
            print(f"    Discrete:   obj={np.float64(obj_disc):.6f}, comp={np.float64(comp_disc):.6f}, TV={np.float64(tv_disc):.6f}")
            print(f"    Runtime: total={t_total:.2f}s, solver={t_solver:.2f}s, disc={t_disc:.2f}s")