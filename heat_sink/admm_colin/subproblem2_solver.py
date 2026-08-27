from graph_tv import chambolle_pock_graph_tv
import gurobipy as gp
from gurobipy import GRB
import mergesplit.mergesplit as ms
import networkx as nx
import numpy as np
#from fenics import *
import math
import pyomo.environ as pyo
from scipy import sparse
import osqp

class Subproblem2Solver:
    def __init__(self, n_x, n_y, alpha, seed, use_mip=True, cutoff_time=None):
        """
        n_x, n_y : ints
            dimensions of your 2D grid
        alpha : float
            TV weight
        seed : int
            RNG seed for mergesplit
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n = n_x * n_y
        self.alpha = alpha
        self.seed = seed
        self.use_mip = bool(use_mip)
        self.cutoff_time = cutoff_time

        # build the graph once
        #self.graph = self._build_graph(n_x, n_y)
        self.graph = self.build_graph(n_x, n_y)

        # precompute the per-edge scale factors
        # (so you dont recompute abs(u-v)==1 each iteration)
        self.scale = np.zeros(len(self.graph.edges()))
        for k, (u, v) in enumerate(self.graph.edges()):
            self.scale[k] = math.sqrt(2) if abs(int(u) - int(v)) == 1 else 1.0  

    def compute_TV(self, a, b, lam, rho):
        """Total variation term at (a,b,lam,rho)"""
        # note: lam, rho arent used here  but signature stays same
        diffs = []
        for (u, v), s in zip(self.graph.edges(), self.scale):
            diffs.append(s * abs(a[u] - a[v]))
        Gg = sum(diffs)
        return (1.0 / self.n_x) * Gg * self.alpha

    def computeF(self, a, b, lam, rho):
        """Quadratic penalty term"""
        # Regular ADMM form: lambda.(b-a) + (rho/2)*|b-a|^2
        diff = b - a
        return (lam @ diff + (rho/2) * (diff**2).sum()) / len(b)

    # def _build_graph(self, mesh):
    #     G = nx.Graph()
    #     num_cells = mesh.num_cells()
    #     G.add_nodes_from(range(num_cells))

    #     # get the connectivity: cell → facets → neighboring cells
    #     mesh.init(2, 1)
    #     mesh.init(1, 2)

    #     for cell in cells(mesh):
    #         cid = cell.index()
    #         for facet in facets(cell):
    #             for neighbor in facet.entities(2):
    #                 if neighbor != cid:
    #                     G.add_edge(cid, neighbor)

    #     return G

    def build_graph(self, n_x, n_y):
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

    def run(self, a, b, lam, rho, V_max, seed, k, backend):
        """
        Solve the subproblem with the selected backend.

        Parameters
        ----------
        backend : {'mergesplit','gurobi'}
            Which implementation to use.
        return_raw : bool
            If True and backend='mergesplit', also return the raw updown object.

        Returns
        -------
        x : np.ndarray or None
            Binary solution (0/1) when available. None if no feasible solution.
        status : int or str
            Backend-specific status (e.g., Gurobi status code, 'OK'/'FAIL' for mergesplit).
        raw : object (optional)
            Only returned if return_raw=True for 'mergesplit'; the PyUpDownMergeSplit object.
        """
        if backend == "mergesplit":
            x, status = self._run_mergesplit(a, b, lam, rho, V_max, seed)
            return x, status
        elif backend == "gurobi":
            x, status = self._run_gurobi(a, b, lam, rho, V_max, seed)
            return x, status
        elif backend == "chambolle-pock":
            x, status = self._run_chambolle_pock(a, b, lam, rho, V_max)
            return x, status
        elif backend == "osqp":
            x, status = self._run_osqp(a, b, lam, rho, V_max, seed)
            return x, status
        elif backend in ["scip", "cplex"]:
            solver = pyo.SolverFactory(backend)
            model = self.build_pyomo_model(a, b, lam, rho, V_max)
            solver.options['time'] = 60
            solver.solve(model, tee=True)
            x = np.array([pyo.value(model.w[i]) for i in model.nodes])
            return x, 'OK'
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'mergesplit', 'gurobi', or 'chambolle-pock'.")

    # ---------- backend implementations ----------
    def _run_mergesplit(self, a, b, lam, rho, V_max, seed):
        """
        Original mergesplit implementation. Tries to return an np.array solution too.
        """
        # Regular ADMM form: lambda.(b-x) + rho*|b-x|^2
        F = lambda x: (lam * (b - x) + (rho/2) * (b - x)**2) / len(b)
        G = lambda y: (self.alpha * self.scale * np.abs(y)) / math.sqrt(len(b)/2)
        H = lambda x: x.flatten()

        updown = ms.PyUpDownMergeSplit(
            self.graph, F, G, H, 1,
            trust_region_active=True,
            delta=V_max * len(b),
            seed=seed,
            efficiency_ordering=True
        )
        updown.initialize(a.astype(np.int32))
        updown.optimize()
        
        sol = updown.x
        
        status = "OK" if sol is not None else "FAIL"

        F = self.computeF(a, b, lam, rho)
        TV = self.compute_TV(a, b, lam, rho)
        
        print(f"Quadratic penalty F(a) = {F}")
        print(f"TV term G(a) = {TV}")
        
        return sol, status

    def _run_gurobi(self, a, b, lam, rho, V_max, seed):
        """
        Original Gurobi implementation. Returns (x, status).
        """
        N = len(self.graph.nodes)
        E = list(self.graph.edges())

        m = gp.Model("graph_binary_opt")
        m.Params.OutputFlag = 0
        m.Params.Seed = int(seed)
        if self.cutoff_time is not None and float(self.cutoff_time) > 0:
            m.Params.TimeLimit = float(self.cutoff_time)

        # Decision vars: binary if MIP mode is enabled, otherwise continuous in [0, 1].
        w_vtype = GRB.BINARY if self.use_mip else GRB.CONTINUOUS
        w = m.addMVar(N, vtype=w_vtype, lb=0.0, ub=1.0, name="w")
        # Start from 'a' if provided
        try:
            w.Start = a
        except Exception:
            pass

        # Budget constraint
        m.addConstr(w.sum() <= V_max * N, name="budget")

        # Regular ADMM penalty term
        diff = b - w
        lag = lam @ diff + (rho/2) * (diff @ diff)
        quad_term = lag / len(b)

        # TV term using absolute differences on edges
        tv_terms = []
        for k, (i, j) in enumerate(E):
            d = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"d_{i}_{j}")
            m.addConstr(d >=  w[i] - w[j])
            m.addConstr(d >=  w[j] - w[i])
            tv_terms.append(self.alpha * self.scale[k] * d)
        tv_term = gp.quicksum(tv_terms) / math.sqrt(len(b)/2)

        m.setObjective(quad_term + tv_term, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount > 0:
            x_dtype = int if self.use_mip else float
            x = np.asarray(w.X, dtype=x_dtype)
            return x, m.Status
        else:
            return None, m.Status

    import math



    def _run_osqp(self, a, b, lam, rho, V_max, seed=None):
        """
        Solve the continuous graph-TV QP using OSQP.

        Returns
        -------
        x : np.ndarray or None
            Continuous solution w in [0, 1]^N.
        status : str
            OSQP status string.

        Notes
        -----
        OSQP does not support binary variables. Therefore, this method solves only
        the continuous relaxation corresponding to self.use_mip=False.
        """
        if self.use_mip:
            raise ValueError(
                "OSQP cannot solve mixed-integer problems. "
                "Use Gurobi when self.use_mip=True."
            )

        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges())

        N = len(nodes)
        M = len(edges)

        if N == 0:
            raise ValueError("The graph contains no nodes.")

        b = np.asarray(b, dtype=float).reshape(-1)
        lam = np.asarray(lam, dtype=float).reshape(-1)

        if b.size != N or lam.size != N:
            raise ValueError(
                f"Expected b and lam to have length {N}, "
                f"got {b.size} and {lam.size}."
            )

        if rho < 0:
            raise ValueError("rho must be nonnegative for the QP to be convex.")

        scale = np.asarray(self.scale, dtype=float).reshape(-1)
        if scale.size != M:
            raise ValueError(
                f"Expected self.scale to contain one value per edge ({M}), "
                f"but got {scale.size}."
            )

        # If graph node labels are not necessarily 0, ..., N-1, map them to indices.
        node_to_index = {node: idx for idx, node in enumerate(nodes)}

        # ------------------------------------------------------------
        # Graph incidence matrix B:
        #
        # For edge k = (i, j):
        #     (B w)_k = w_i - w_j
        # ------------------------------------------------------------
        if M > 0:
            edge_rows = np.repeat(np.arange(M), 2)

            edge_cols = np.empty(2 * M, dtype=int)
            edge_data = np.empty(2 * M, dtype=float)

            for k, (node_i, node_j) in enumerate(edges):
                i = node_to_index[node_i]
                j = node_to_index[node_j]

                edge_cols[2 * k] = i
                edge_cols[2 * k + 1] = j

                edge_data[2 * k] = 1.0
                edge_data[2 * k + 1] = -1.0

            B = sparse.csc_matrix(
                (edge_data, (edge_rows, edge_cols)),
                shape=(M, N),
            )
        else:
            B = sparse.csc_matrix((0, N))

        # Decision vector:
        #
        #     z = [w_1, ..., w_N, d_1, ..., d_M]
        #
        n_variables = N + M

        # ------------------------------------------------------------
        # Objective:
        #
        # (1/N) [lamᵀ(b-w) + rho/2 ||b-w||²]
        # + alpha / sqrt(N/2) * scaleᵀ d
        #
        # Removing constants:
        #
        # rho/(2N) wᵀw
        # - (lam + rho*b)ᵀw/N
        # + alpha / sqrt(N/2) * scaleᵀd
        #
        # OSQP form:
        #
        #     1/2 zᵀ P z + qᵀ z
        # ------------------------------------------------------------
        P_w = (rho / N) * sparse.eye(N, format="csc")
        P_d = sparse.csc_matrix((M, M))

        P = sparse.block_diag(
            (P_w, P_d),
            format="csc",
        )

        q_w = -(lam + rho * b) / N
        q_d = self.alpha * scale / math.sqrt(N / 2.0)

        q = np.concatenate((q_w, q_d))

        # ------------------------------------------------------------
        # Constraints use OSQP's form:
        #
        #                lower <= A z <= upper
        #
        # 1. 0 <= w <= 1
        # 2. 0 <= d <= 1
        # 3. sum(w) <= V_max * N
        # 4. B w - d <= 0
        # 5. -B w - d <= 0
        #
        # Constraints 4 and 5 imply
        #
        #     d >= |B w|.
        # ------------------------------------------------------------
        I_w = sparse.eye(N, format="csc")
        I_d = sparse.eye(M, format="csc")

        zero_wd = sparse.csc_matrix((N, M))
        zero_dw = sparse.csc_matrix((M, N))

        A_w_bounds = sparse.hstack(
            (I_w, zero_wd),
            format="csc",
        )

        A_d_bounds = sparse.hstack(
            (zero_dw, I_d),
            format="csc",
        )

        A_budget = sparse.hstack(
            (
                sparse.csc_matrix(np.ones((1, N))),
                sparse.csc_matrix((1, M)),
            ),
            format="csc",
        )

        A_tv_positive = sparse.hstack(
            (B, -I_d),
            format="csc",
        )

        A_tv_negative = sparse.hstack(
            (-B, -I_d),
            format="csc",
        )

        A = sparse.vstack(
            (
                A_w_bounds,
                A_d_bounds,
                A_budget,
                A_tv_positive,
                A_tv_negative,
            ),
            format="csc",
        )

        lower = np.concatenate(
            (
                np.zeros(N),            # w >= 0
                np.zeros(M),            # d >= 0
                np.array([-np.inf]),    # no budget lower bound
                np.full(M, -np.inf),    # B w - d has no lower bound
                np.full(M, -np.inf),    # -B w - d has no lower bound
            )
        )

        upper = np.concatenate(
            (
                np.ones(N),                 # w <= 1
                np.ones(M),                 # d <= 1
                np.array([V_max * N]),      # sum(w) <= V_max*N
                np.zeros(M),                # B w - d <= 0
                np.zeros(M),                # -B w - d <= 0
            )
        )

        solver = osqp.OSQP()

        settings = {
            "verbose": False,
            "warm_starting": True,
            "polishing": True,
            "adaptive_rho": True,
            "max_iter": 50_000,
            "eps_abs": 1e-2,
            "eps_rel": 1e-2,
            "scaled_termination": True,
        }

        if self.cutoff_time is not None and float(self.cutoff_time) > 0:
            settings["time_limit"] = float(self.cutoff_time)

        solver.setup(
            P=P,
            q=q,
            A=A,
            l=lower,
            u=upper,
            **settings,
        )

        # Warm-start from a.
        if a is not None:
            a = np.asarray(a, dtype=float).reshape(-1)

            if a.size == N:
                w_start = np.clip(a, 0.0, 1.0)

                if M > 0:
                    d_start = np.abs(B @ w_start)
                else:
                    d_start = np.empty(0, dtype=float)

                z_start = np.concatenate((w_start, d_start))
                solver.warm_start(x=z_start)

        result = solver.solve()

        status = result.info.status
        status_val = result.info.status_val

        # Current OSQP status values:
        # 1 = solved
        # 2 = solved inaccurate
        if result.x is not None and status_val in (1, 2):
            w_solution = np.asarray(result.x[:N], dtype=float)

            # Remove tiny numerical violations, e.g. -1e-9 or 1 + 1e-9.
            w_solution = np.clip(w_solution, 0.0, 1.0)

            return w_solution, status

        return None, status

    def build_pyomo_model(self, a, b, lam, rho, V_max):
        """
        Generic Pyomo model corresponding to the given Gurobi model.
    
        Parameters
        ----------
        a : array-like or None
            Optional initial guess for w.
        b : array-like
        lam : array-like
        rho : float
        V_max : float
        alpha : float
    
        Returns
        -------
        model : pyo.ConcreteModel
        """
        model = pyo.ConcreteModel()
    
        # Sets
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges())
        N = len(nodes)
    
        model.nodes = pyo.Set(initialize=nodes)
        model.edges = pyo.Set(initialize=edges, dimen=2)
    
        # Parameters
        b_dict = {i: float(b[i]) for i in nodes}
        lam_dict = {i: float(lam[i]) for i in nodes}
    
        model.b = pyo.Param(model.nodes, initialize=b_dict)
        model.lam = pyo.Param(model.nodes, initialize=lam_dict)
        model.rho = pyo.Param(initialize=float(rho))
        model.V_max = pyo.Param(initialize=float(V_max))
        model.alpha = pyo.Param(initialize=float(self.alpha))
        model.N_total = pyo.Param(initialize=N)
    
        # Variables
        w_domain = pyo.Binary if self.use_mip else pyo.UnitInterval
        model.w = pyo.Var(model.nodes, domain=w_domain, bounds=(0,1))
        model.d = pyo.Var(model.edges, domain=pyo.NonNegativeReals)
    
        # Optional warm start
        if a is not None:
            for i in nodes:
                try:
                    model.w[i].value = int(a[i])
                except Exception:
                    pass
    
        # Budget constraint: sum(w) <= V_max * N
        def budget_rule(m):
            return sum(m.w[i] for i in m.nodes) <= (m.V_max * m.N_total)
        model.budget = pyo.Constraint(rule=budget_rule)
    
        # Absolute-difference constraints on edges:
        # d[i,j] >= w[i] - w[j]
        # d[i,j] >= w[j] - w[i]
        def abs1_rule(m, i, j):
            return m.d[i, j] >= m.w[i] - m.w[j]
        model.abs1 = pyo.Constraint(model.edges, rule=abs1_rule)
    
        def abs2_rule(m, i, j):
            return m.d[i, j] >= m.w[j] - m.w[i]
        model.abs2 = pyo.Constraint(model.edges, rule=abs2_rule)
    
        # Objective:
        # quad_term = (rho/2) * sum_i (b_i - w_i + lam_i)^2 / len(b)
        # tv_term   = alpha * sum_(i,j) scale_(i,j) * d_(i,j) / sqrt(len(b))
        def obj_rule(m):
            quad_term = (
                (m.rho / 2.0)
                * sum((m.b[i] - m.w[i] + m.lam[i])**2 for i in m.nodes)
                / N
            )
            tv_term = (
                sum(m.alpha * self.scale[k] * m.d[i, j] for k, (i, j) in enumerate(m.edges))
                / self.n_x
            )
            return quad_term + tv_term
    
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
        return model
    
    def _run_chambolle_pock(self, a, b, lam, rho, V_max):
        n = len(b)
        budget = V_max * n
        edges = np.asarray(list(self.graph.edges()), dtype=int)

        # ADMM quadratic term in y-update (scaled by 1/n to match subproblem1):
        # (1/n) * [lambda.(b-y) + (rho/2)||b-y||^2]
        #   = (rho/(2n)) y^2 + ((-rho*b - lambda)/n) y + const
        # TV term scaled by 1/sqrt(n/2) to match.
        a_quad = np.full(n, rho / (2.0 * n), dtype=float)
        b_lin = (-rho * np.asarray(b, dtype=float) + rho * np.asarray(lam, dtype=float)) / n
        alpha_scaled = self.alpha / np.sqrt(n / 2.0)

        x_init = np.asarray(a, dtype=float) if a is not None else None
        sol, info = chambolle_pock_graph_tv(
            n_vertices=n,
            edges=edges,
            a=a_quad,
            b=b_lin,
            budget=budget,
            alpha=alpha_scaled,
            x_lo=0.0,
            max_iter=2000,
            tol=1e-8,
            x_init=x_init,
            edge_weights=self.scale,
        )

        status = "OK" if info.get("converged", False) else "MAX_ITER"
        return np.asarray(sol, dtype=float), status