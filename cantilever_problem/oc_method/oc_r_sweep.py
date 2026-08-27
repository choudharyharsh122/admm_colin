import argparse
import math
import time
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def tri3_stiffness(xy: np.ndarray, E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """P1 triangle plane-stress elasticity stiffness matrix for a CCW triangle."""
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area <= 0:
        raise ValueError("Triangle must be counterclockwise with positive area")

    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    B = np.zeros((3, 6), dtype=float)
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    B /= 2.0 * area

    D = (E / (1.0 - nu * nu)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
        dtype=float,
    )

    return area * (B.T @ D @ B)

def build_unitsquaremesh_right_tri(n_x, n_y):
    """Build a triangular mesh on the unit rectangle [0,1]x[0,1] using right diagonals."""
    hx = 1.0 / n_x
    hy = 1.0 / n_y
    nnx = n_x + 1
    nny = n_y + 1

    nodenrs = np.arange(nnx * nny).reshape((nny, nnx), order="C")

    coords = np.zeros((nnx * nny, 2), dtype=float)
    for j in range(nny):
        for i in range(nnx):
            coords[nodenrs[j, i]] = [i * hx, j * hy]

    tris = []
    tri_type = []
    bottom_edges = []
    right_edges = []

    for j in range(n_y):
        for i in range(n_x):
            bl = nodenrs[j, i]
            br = nodenrs[j, i + 1]
            tl = nodenrs[j + 1, i]
            tr = nodenrs[j + 1, i + 1]

            # default "right" diagonal: bl -> tr
            tris.append([bl, br, tr]); tri_type.append(0)
            tris.append([bl, tr, tl]); tri_type.append(1)

            if j == 0:
                bottom_edges.append([bl, br])
            if i == n_x - 1:
                right_edges.append([br, tr])

    tris = np.asarray(tris, dtype=int)
    tri_type = np.asarray(tri_type, dtype=int)
    bottom_edges = np.asarray(bottom_edges, dtype=int)
    right_edges = np.asarray(right_edges, dtype=int)

    west_nodes = nodenrs[:, 0]
    north_nodes = nodenrs[-1, :]
    dirichlet_nodes = np.unique(np.r_[west_nodes, north_nodes]).astype(int)

    return coords, tris, tri_type, bottom_edges, right_edges, dirichlet_nodes


def prepare_filter_from_centroids(centroids: np.ndarray, rmin: float):
    """Build sparse sensitivity filter matrix H and row sums Hs."""
    nele = centroids.shape[0]
    rows, cols, vals = [], [], []

    for i in range(nele):
        xi = centroids[i]
        for j in range(nele):
            dist = np.linalg.norm(xi - centroids[j])
            w = max(0.0, rmin - dist)
            if w > 0:
                rows.append(i)
                cols.append(j)
                vals.append(w)

    H = coo_matrix((vals, (rows, cols)), shape=(nele, nele)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def build_exact_filter_template(centroids: np.ndarray, rmax: float):
    """Build neighbor template once for all rmin <= rmax."""
    nele = centroids.shape[0]
    rows, cols, dists = [], [], []

    for i in range(nele):
        diff = centroids - centroids[i]
        dist_i = np.linalg.norm(diff, axis=1)
        mask = dist_i < float(rmax)
        idx = np.where(mask)[0]

        rows.extend([i] * idx.size)
        cols.extend(idx.tolist())
        dists.extend(dist_i[idx].tolist())

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(dists, dtype=float),
        nele,
    )


def prepare_filter_from_template(neighbor_template, rmin: float):
    """Build filter matrix for a specific rmin for a mesh."""
    rows, cols, dists, nele = neighbor_template
    vals = rmin - dists
    mask = vals > 0.0

    H = coo_matrix((vals[mask], (rows[mask], cols[mask])), shape=(nele, nele)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


# # H, Hs = prepare_filter_from_centroids( ), 1)
# coords, tris, tri_type, bottom, right, dirichlet_nodes = build_unitsquaremesh_right_tri(2)
# centroids = coords[tris].mean(axis=1)
# H, Hs = prepare_filter_from_centroids(centroids, 0.38)

def apply_sensitivity_filter(H, Hs, a: np.ndarray, dc: np.ndarray, gamma: float = 1e-6):
    """
    Sensitivity filter:
    dcf_e = [sum_i H_ei * a_i * dc_i] / [Hs_e * max(gamma, a_e)]
    """
    num = H @ (a * dc)
    denom = Hs * np.maximum(gamma, a)
    denom = np.maximum(denom, 1e-12)
    return num / denom

def oc_update(a: np.ndarray, volfrac: float, dc: np.ndarray, move: float = 0.2) -> np.ndarray:
    """Optimality criteria update under box and volume constraints."""
    l1, l2 = 0.0, 1e5
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        ratio = np.maximum(1e-30, -dc / lmid)
        a_candidate = a * np.sqrt(ratio)

        anew = np.maximum(
            0.001,
            np.maximum(a - move, np.minimum(1.0, np.minimum(a + move, a_candidate))),
        )

        if anew.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid

    return anew

def build_graph(n_x: int, n_y: int):
    """Graph builder, this is hardcoded for the right triangular mesh and single element
    indexing is linear and 1D and starts bottom left to top right"""
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
    scale = np.zeros(len(graph.edges()), dtype=float)
    for k, (u, v) in enumerate(graph.edges()):
        scale[k] = math.sqrt(2.0) if abs(int(u) - int(v)) == 1 else 1.0
    return scale

def compute_tv(control: np.ndarray, graph: nx.Graph, scale: np.ndarray) -> float:
    diffs = []
    for (u, v), s in zip(graph.edges(), scale):
        diffs.append(s * abs(control[u] - control[v]))
    return float(sum(diffs))

def solve_state_and_compliance(
    a: np.ndarray,
    coords: np.ndarray,
    tris: np.ndarray,
    tri_type: np.ndarray,
    fixeddofs: np.ndarray,
    iK: np.ndarray,
    jK: np.ndarray,
    F: np.ndarray,
    penal: float,
    eps: float,
):
    """Solve PDE for control a and return (U, compliance)."""
    nele = tris.shape[0]
    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    hx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    hy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0

    KE0 = tri3_stiffness(
        np.array([[0.0, 0.0], [hx, 0.0], [hx, hy]], dtype=float)
    )
    KE1 = tri3_stiffness(
        np.array([[0.0, 0.0], [hx, hy], [0.0, hy]], dtype=float)
    )

    KE_all = np.zeros((nele, 6, 6), dtype=float)
    KE_all[tri_type == 0] = KE0
    KE_all[tri_type == 1] = KE1

    kvec = eps + (1.0 - eps) * a ** penal
    sK = (KE_all.reshape(nele, 36) * kvec[:, None]).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    alldofs = np.arange(ndof, dtype=int)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    U = np.zeros(ndof, dtype=float)
    U[freedofs] = spsolve(K[freedofs][:, freedofs], F[freedofs])
    U[fixeddofs] = 0.0

    compliance = float(F @ U)
    return U, compliance


def run_topology_optimization(
    mesh_dim_x: int,
    mesh_dim_y: int,
    volfrac: float,
    penal: float,
    rmin: float,
    eps: float,
    f0: float,
    tol: float,
    maxiter: int,
    neighbor_template=None,
):
    """Run TO with sensitivity filtering and return final continuous control."""
    coords, tris, tri_type, bottom_edges, right_edges, dirichlet_nodes = build_unitsquaremesh_right_tri(
        mesh_dim_x, mesh_dim_y
    )

    # print("Fixed DOFs:", sorted(dirichlet_nodes))          # or fixeddofs
    # print("Number of fixed DOFs:", len(dirichlet_nodes))

    nele = tris.shape[0]
    n_nodes = coords.shape[0]
    ndof = 2 * n_nodes
    hx = 1.0 / mesh_dim_x
    hy = 1.0 / mesh_dim_y

    KE0 = tri3_stiffness(
        np.array([[0.0, 0.0], [hx, 0.0], [hx, hy]], dtype=float)
    )
    KE1 = tri3_stiffness(
        np.array([[0.0, 0.0], [hx, hy], [0.0, hy]], dtype=float)
    )

    KE_all = np.zeros((nele, 6, 6), dtype=float)
    KE_all[tri_type == 0] = KE0
    KE_all[tri_type == 1] = KE1

    edofs = np.empty((nele, 6), dtype=int)
    edofs[:, 0::2] = 2 * tris
    edofs[:, 1::2] = 2 * tris + 1

    # Cantilever-like boundary conditions from the reference implementation.
    # Fix left-edge horizontal displacements, plus the y displacement at the top-right node.
    nodenrs = np.arange((mesh_dim_x + 1) * (mesh_dim_y + 1)).reshape(
        (mesh_dim_y + 1, mesh_dim_x + 1), order="C"
    )
    left_nodes = nodenrs[:, 0]
    top_right_node = nodenrs[-1, -1]
    fixeddofs = np.unique(
        np.concatenate([2 * left_nodes, [2 * top_right_node + 1]])
    )

    # Classic left-edge clamp (both components)
    left_nodes = nodenrs[:, 0]
    fixeddofs = np.unique(np.concatenate([2 * left_nodes, 2 * left_nodes + 1]))

    # Classic force on right edge mid-height
    load_node = nodenrs[mesh_dim_y // 2, -1]   # right edge, middle
    F = np.zeros(ndof, dtype=float)
    F[2 * load_node + 1] = -f0

    # print("Force vector non-zeros:", np.where(np.abs(F) > 1e-12)[0])
    # print("Force values:", F[np.abs(F) > 1e-12])

    iK = np.repeat(edofs, 6, axis=1).ravel()
    jK = np.tile(edofs, (1, 6)).ravel()

    t_filter_start = time.perf_counter()
    if neighbor_template is None:
        centroids = coords[tris].mean(axis=1)
        H, Hs = prepare_filter_from_centroids(centroids, rmin)
    else:
        H, Hs = prepare_filter_from_template(neighbor_template, rmin)
    runtime_filter = time.perf_counter() - t_filter_start

    a = np.full(nele, volfrac, dtype=float)

    # print("n_x, n_y =", mesh_dim_x, mesh_dim_y)
    # print("n_nodes =", coords.shape[0])
    # print("coords of nodes with x≈0:\n", coords[np.isclose(coords[:,0], 0)])
    # print("Fixed DOFs:", sorted(fixeddofs))
    # print("Force non-zeros:", np.where(np.abs(F) > 1e-12)[0])

    loop = 0
    change = 1.0
    t_algo_start = time.perf_counter()
    while change > tol and loop < maxiter:
        loop += 1
        a_old = a.copy()

        kvec = eps + (1.0 - eps) * a ** penal
        sK = (KE_all.reshape(nele, 36) * kvec[:, None]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

        alldofs = np.arange(ndof, dtype=int)
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        U = np.zeros(ndof, dtype=float)
        U[freedofs] = spsolve(K[freedofs][:, freedofs], F[freedofs])
        U[fixeddofs] = 0.0

        Ue = U[edofs]
        ce = np.einsum("ei,eij,ej->e", Ue, KE_all, Ue)
        dkda = (1.0 - eps) * penal * np.maximum(a, 1e-12) ** (penal - 1)
        dc = -(dkda * ce)

        dcf = apply_sensitivity_filter(H, Hs, a, dc, gamma=1e-3)
        a = oc_update(a, volfrac, dcf)

        change = float(np.max(np.abs(a - a_old)))
        c = float(F @ U)
        print(
            f"It.: {loop:4d} Obj.: {c:12.6e} Vol.: {a.mean():6.3f} "
            f"ch.: {change:6.3f} rmin={rmin:.6f}"
        )

    runtime_algorithm = time.perf_counter() - t_algo_start

    return a, coords, tris, tri_type, fixeddofs, iK, jK, F, runtime_filter, runtime_algorithm

def discretize_control(a_cont: np.ndarray, volfrac: float) -> np.ndarray:
    """ volume preserving rounding """
    a_disc = np.zeros_like(a_cont)
    k = int(np.floor(volfrac * a_cont.size))
    idx = np.argsort(a_cont)[::-1]
    a_disc[idx[:k]] = 1.0
    return a_disc

def save_result(
    output_path: Path,
    control_cont: np.ndarray,
    control_disc: np.ndarray,
    compliance_disc: float,
    tv_disc: float,
    mesh_dim: int,
    r: float,
    rmin: float,
    runtime_filter: float,
    runtime_algorithm: float,
    runtime_total: float,
    runtime_template_prep: float,
    runtime_template_filter_algo: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["mesh_dim"] = int(mesh_dim)
        h5f.attrs["r"] = float(r)
        h5f.attrs["rmin"] = float(rmin)
        h5f.attrs["runtime_filter"] = float(runtime_filter)
        h5f.attrs["runtime_algorithm"] = float(runtime_algorithm)
        h5f.attrs["runtime_total"] = float(runtime_total)
        h5f.attrs["runtime_template_prep"] = float(runtime_template_prep)
        h5f.attrs["runtime_template_filter_algo"] = float(runtime_template_filter_algo)
        h5f.create_dataset("control_cont", data=control_cont)
        h5f.create_dataset("control_disc", data=control_disc)
        h5f.create_dataset("compliance_disc", data=np.float64(compliance_disc))
        h5f.create_dataset("tv_disc", data=np.float64(tv_disc))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep sensitivity-filter TO over r values and save results."
    )
    parser.add_argument("--mesh-dim", type=int, default=64)
    parser.add_argument("--r-start", type=float, default=2)
    parser.add_argument("--r-end", type=float, default=2)
    parser.add_argument("--num-r", type=int, default=1)
    parser.add_argument("--volfrac", type=float, default=0.4)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--f0", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--output-root", type=str, default="OC_results_mew")
    args = parser.parse_args()

    mesh_dim_x = args.mesh_dim
    mesh_dim_y = args.mesh_dim

    r_values = np.linspace(args.r_start, args.r_end, args.num_r)
    rmin_values = r_values / mesh_dim_x

    # Build centroid-distance model for the 2:1 cantilever mesh.
    t_template_start = time.perf_counter()
    base_coords, base_tris, _, _, _, _ = build_unitsquaremesh_right_tri(
        mesh_dim_x, mesh_dim_y
    )
    base_centroids = base_coords[base_tris].mean(axis=1)
    neighbor_template = build_exact_filter_template(
        base_centroids, rmax=float(np.max(rmin_values))
    )
    runtime_template_prep = time.perf_counter() - t_template_start
    
    # Builds the graph of mesh for Total variation calculation on the solution later.
    # Control graph size must match the number of triangular elements.
    graph = build_graph(mesh_dim_y, 2 * mesh_dim_x)

    # assigns scale to each edge based on above graph
    scale = build_scale(graph)

    for run_idx, r in enumerate(r_values, start=1):
        t_run_start = time.perf_counter()
        rmin = float(r / mesh_dim_x)
        print(
            f"\n=== Run {run_idx}/{len(r_values)}: r={r:.6f}, "
            f"rmin={rmin:.6f} ==="
        )

        control_cont, coords, tris, tri_type, fixeddofs, iK, jK, F, runtime_filter, runtime_algorithm = run_topology_optimization(
            mesh_dim_x=mesh_dim_x,
            mesh_dim_y=mesh_dim_y,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=rmin,
            eps=args.eps,
            f0=args.f0,
            tol=args.tol,
            maxiter=args.maxiter,
            neighbor_template=neighbor_template,
        )

        control_disc = discretize_control(control_cont, args.volfrac)

        _, compliance_disc = solve_state_and_compliance(
            control_disc,
            coords,
            tris,
            tri_type,
            fixeddofs,
            iK,
            jK,
            F,
            args.penal,
            args.eps,
        )

        tv_disc = compute_tv(control_disc, graph, scale)
        runtime_total = time.perf_counter() - t_run_start
        runtime_template_filter_algo = runtime_template_prep + runtime_filter + runtime_algorithm

        r_folder = f"{r:.2f}"
        output_path = Path(args.output_root) / r_folder / f"{args.mesh_dim}.h5"
        save_result(
            output_path,
            control_cont,
            control_disc,
            compliance_disc,
            tv_disc,
            args.mesh_dim,
            r,
            rmin,
            runtime_filter,
            runtime_algorithm,
            runtime_total,
            runtime_template_prep,
            runtime_template_filter_algo,
        )
        print(
            f"Runtimes: filter={runtime_filter:.3f}s, "
            f"algorithm={runtime_algorithm:.3f}s, total={runtime_total:.3f}s, "
            f"template+filter+algo={runtime_template_filter_algo:.3f}s"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()