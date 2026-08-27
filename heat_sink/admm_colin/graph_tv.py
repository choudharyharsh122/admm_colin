import numpy as np


def chambolle_pock_graph_tv(
    n_vertices,
    edges,
    a,
    b,
    budget=None,
    alpha=0.01,
    x_lo=0.0,
    max_iter=50_000,
    tol=1e-7,
    callback=None,
    x_init=None,
    edge_weights=None,
):
    """
    Solve

        min_x sum_i (a_i*x_i**2 + b_i*x_i)
              + alpha*sum_e w_e*abs(x_u(e) - x_v(e))

        subject to x_lo <= x_i <= 1
                   sum_i x_i <= budget, if budget is not None.

    The box and budget constraints are handled together in the primal proximal
    map. Thus, the Chambolle-Pock linear operator contains only graph TV and
    does not become ill-conditioned merely because the mesh has more vertices.
    """
    n = int(n_vertices)
    if n <= 0:
        raise ValueError("n_vertices must be positive")

    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge vertex indices are out of range")

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != (n,) or b.shape != (n,):
        raise ValueError("a and b must have shape (n_vertices,)")
    if not np.all(np.isfinite(a)) or not np.all(a > 0.0):
        raise ValueError("all quadratic coefficients must be finite and positive")
    if not np.all(np.isfinite(b)):
        raise ValueError("all linear coefficients must be finite")
    if not 0.0 <= x_lo <= 1.0:
        raise ValueError("x_lo must lie in [0, 1]")
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if budget is not None and budget < n * x_lo:
        raise ValueError("budget is incompatible with the lower bound")

    n_edges = edges.shape[0]
    u_idx = edges[:, 0]
    v_idx = edges[:, 1]

    if edge_weights is None:
        weights = np.ones(n_edges, dtype=float)
    else:
        weights = np.asarray(edge_weights, dtype=float)
        if weights.shape != (n_edges,):
            raise ValueError("edge_weights must have shape (n_edges,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("edge_weights must be finite and nonnegative")

    # K_tv x = weights * (x[u] - x[v]).
    rows = np.concatenate((u_idx, v_idx))

    def apply_adjoint(y):
        """Apply K_tv.T without explicitly constructing a sparse matrix."""
        edge_flux = weights * y
        return (
            np.bincount(u_idx, weights=edge_flux, minlength=n)
            - np.bincount(v_idx, weights=edge_flux, minlength=n)
        )

    # Diagonal preconditioning based on absolute row/column sums of K_tv.
    # This avoids forcing the entire mesh to use the step size required by
    # only its largest edge weight or highest-degree vertex.
    vertex_weight_sum = np.bincount(
        rows,
        weights=np.concatenate((weights, weights)),
        minlength=n,
    )
    safety = 0.99
    tau = np.ones(n, dtype=float)
    connected = vertex_weight_sum > 0.0
    tau[connected] = safety / vertex_weight_sum[connected]

    sigma = np.ones(n_edges, dtype=float)
    positive_edges = weights > 0.0
    sigma[positive_edges] = safety / (2.0 * weights[positive_edges])

    denom = 1.0 + 2.0 * tau * a

    if x_init is None:
        x = np.full(n, np.clip(0.5, x_lo, 1.0), dtype=float)
    else:
        x_init = np.asarray(x_init, dtype=float)
        if x_init.shape != (n,):
            raise ValueError("x_init must have shape (n_vertices,)")
        x = np.clip(x_init, x_lo, 1.0)

    y_tv = np.zeros(n_edges, dtype=float)
    x_bar = x.copy()
    converged = False
    primal_residual = np.inf
    dual_residual = np.inf
    budget_violation = 0.0

    def primal_prox(z):
        """prox of quadratic + box + volume indicator."""
        def candidate(multiplier):
            return np.clip(
                (z - tau * (b + multiplier)) / denom,
                x_lo,
                1.0,
            )

        result = candidate(0.0)
        if budget is None or result.sum() <= budget:
            return result

        # sum(candidate(mu)) is continuous and nonincreasing in mu.
        mu_lo = 0.0
        mu_hi = 1.0
        while candidate(mu_hi).sum() > budget:
            mu_hi *= 2.0

        # A fixed iteration count gives near machine-precision feasibility.
        for _ in range(60):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            if candidate(mu_mid).sum() > budget:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        return candidate(mu_hi)

    for k in range(max_iter):
        x_old = x.copy()
        y_tv_old = y_tv.copy()

        weighted_difference = weights * (
            x_bar[u_idx] - x_bar[v_idx]
        )
        y_tv = np.clip(
            y_tv + sigma * weighted_difference,
            -alpha,
            alpha,
        )

        prox_argument = x_old - tau * apply_adjoint(y_tv)
        x = primal_prox(prox_argument)
        x_bar = 2.0 * x - x_old

        # Fixed-point residuals are divided by their step sizes, so a tiny
        # step size cannot by itself trigger convergence.
        primal_residual = np.linalg.norm(
            (x - x_old) / tau
        ) / np.sqrt(n)
        tv_dual_residual = (
            np.linalg.norm((y_tv - y_tv_old) / sigma)
            / np.sqrt(n_edges)
            if n_edges
            else 0.0
        )
        dual_residual = tv_dual_residual
        budget_violation = (
            max(0.0, float(x.sum() - budget))
            if budget is not None
            else 0.0
        )

        if callback is not None:
            callback(x.copy(), k)

        residual_scale = max(1.0, np.linalg.norm(x) / np.sqrt(n))
        budget_tol = tol * max(1.0, float(budget or 0.0))
        if (
            primal_residual <= tol * residual_scale
            and dual_residual <= tol
            and budget_violation <= budget_tol
        ):
            converged = True
            break

    info = {
        "n_iter": k + 1,
        "converged": converged,
        "primal_residual": float(primal_residual),
        "dual_residual": float(dual_residual),
        "budget_violation": float(budget_violation),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "sigma_min": float(sigma.min()) if n_edges else None,
        "sigma_max": float(sigma.max()) if n_edges else None,
    }
    return x, info


def objective(x, edges, a, b, alpha=1.0, edge_weights=None):
    """Evaluate the same weighted objective used by the solver."""
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=int)

    weights = (
        np.ones(edges.shape[0], dtype=float)
        if edge_weights is None
        else np.asarray(edge_weights, dtype=float)
    )
    local = np.sum(np.asarray(a) * x**2 + np.asarray(b) * x)
    tv = alpha * np.sum(
        weights * np.abs(x[edges[:, 0]] - x[edges[:, 1]])
    )
    return float(local + tv)


def run_chambolle_pock_admm(
    graph,
    alpha,
    edge_weights,
    a_previous,
    b_admm,
    lambda_unscaled,
    rho,
    volume_fraction,
    max_iter=10_000,
    tol=1e-7,
):
    """
    Solve the ADMM y-subproblem

      (1/n) [lambda.T (b-y) + (rho/2)||b-y||^2]
      + alpha/sqrt(n/2) * weighted_graph_TV(y).

    lambda_unscaled is the ordinary (unscaled) ADMM multiplier.
    """
    b_admm = np.asarray(b_admm, dtype=float)
    lambda_unscaled = np.asarray(lambda_unscaled, dtype=float)
    n = b_admm.size
    if lambda_unscaled.shape != (n,):
        raise ValueError("lambda_unscaled must have the same shape as b_admm")

    edges = np.asarray(list(graph.edges()), dtype=int)
    budget = float(volume_fraction) * n

    a_quad = np.full(n, rho / (2.0 * n), dtype=float)
    b_lin = (-rho * b_admm - lambda_unscaled) / n
    alpha_scaled = alpha / np.sqrt(n / 2.0)

    solution, info = chambolle_pock_graph_tv(
        n_vertices=n,
        edges=edges,
        a=a_quad,
        b=b_lin,
        budget=budget,
        alpha=alpha_scaled,
        x_lo=0.0,
        max_iter=max_iter,
        tol=tol,
        x_init=a_previous,
        edge_weights=edge_weights,
    )
    status = "OK" if info["converged"] else "MAX_ITER"
    return solution, status, info
