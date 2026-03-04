import numpy as np
from gurobipy import Model, GRB

time_lapse = 900
obbt_time = 180


######################
# Auxiliary Functions
######################

def obj_assign(centers, X):
    d, n = X.shape
    k = centers.shape[1]

    dmat = np.zeros((k, n))

    for j in range(n):
        for i in range(k):
            dmat[i, j] = np.sum((X[:, j] - centers[:, i]) ** 2)

    assign = np.empty(n, dtype=int)
    costs = np.empty(n)

    for j in range(n):
        a = np.argmin(dmat[:, j])
        assign[j] = a
        costs[j] = dmat[a, j]

    return np.sum(costs), assign


def init_bound(X, d, k, lower=None, upper=None):
    """
    X shape: (d, n)
    Returns lower, upper of shape (d, k)
    """

    d, n = X.shape

    lower_data = np.min(X, axis=1).reshape(d, 1)
    upper_data = np.max(X, axis=1).reshape(d, 1)

    lower_data = np.repeat(lower_data, k, axis=1)
    upper_data = np.repeat(upper_data, k, axis=1)

    if lower is None or upper is None:
        lower = lower_data.copy()
        upper = upper_data.copy()
    else:
        lower = np.maximum(lower, lower_data)
        upper = np.minimum(upper, upper_data)

    return lower, upper


######################
# Distance Functions
######################

def max_dist(X, d, k, n, lower, upper):
    """
    X: (d, n)
    Returns: (k, n)
    """

    dmat_max = np.zeros((k, n))

    for j in range(n):
        for i in range(k):
            total = 0.0
            for t in range(d):
                total += max(
                    (X[t, j] - lower[t, i]) ** 2,
                    (X[t, j] - upper[t, i]) ** 2
                )
            dmat_max[i, j] = total

    return dmat_max


######################
# Local Optimization
######################

def local_OPT(X, k, lower=None, upper=None):
    """
    Local continuous refinement using SciPy (Ipopt-style behavior).
    X: (d, n)
    lower, upper: (d, k) box bounds
    Returns:
        centers_val: (d, k)
        assign: (n,)
        objv: float
    """

    import numpy as np
    from scipy.optimize import minimize

    d, n = X.shape

    # ------------------------------------------------------
    # Initialize bounds properly
    # ------------------------------------------------------
    lower, upper = init_bound(X, d, k, lower, upper)

    # Flatten centers to vector of length d*k
    def pack(C):
        return C.reshape(d * k)

    def unpack(x):
        return x.reshape((d, k))

    # ------------------------------------------------------
    # Objective: true k-means objective
    # ------------------------------------------------------
    def objective(x):
        C = unpack(x)  # (d, k)

        # Compute squared distances (k, n)
        dists = np.zeros((k, n))
        for i in range(k):
            diff = X - C[:, i:i+1]
            dists[i, :] = np.sum(diff * diff, axis=0)

        # Assign each point to closest center
        min_dists = np.min(dists, axis=0)

        return np.sum(min_dists)

    # ------------------------------------------------------
    # Initial point: project KMeans solution into box
    # ------------------------------------------------------
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X.T)
    C0 = km.cluster_centers_.T  # (d, k)

    # Clip into bounds
    C0 = np.minimum(np.maximum(C0, lower), upper)

    x0 = pack(C0)

    # ------------------------------------------------------
    # Bounds for SciPy
    # ------------------------------------------------------
    bounds = []
    for t in range(d):
        for i in range(k):
            bounds.append((float(lower[t, i]), float(upper[t, i])))

    # ------------------------------------------------------
    # Symmetry constraint:
    # centers[0,j] <= centers[0,j+1]
    # ------------------------------------------------------
    constraints = []

    for j in range(k - 1):

        def constr_factory(j):
            return {
                'type': 'ineq',
                'fun': lambda x, j=j: unpack(x)[0, j + 1] - unpack(x)[0, j]
            }

        constraints.append(constr_factory(j))

    # ------------------------------------------------------
    # Run local optimization (SLSQP handles bounds + constraints)
    # ------------------------------------------------------
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 200,
            "ftol": 1e-6,
            "disp": False
        }
    )

    # If solver fails, fallback to initial guess
    if not result.success:
        centers_val = C0
    else:
        centers_val = unpack(result.x)

    # ------------------------------------------------------
    # Compute final assignment + objective consistently
    # ------------------------------------------------------
    objv, assign = obj_assign(centers_val, X)

    return centers_val, assign, objv


######################
# Global Optimization Base
######################

def global_OPT_base(X, k, lower=None, upper=None, mute=False):
    d, n = X.shape

    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)

    m = Model()

    if mute:
        m.Params.OutputFlag = 0

    m.Params.Threads = 1
    m.Params.TimeLimit = time_lapse

    centers = m.addVars(d, k,
                        lb=lambda t, i: lower[t, i],
                        ub=lambda t, i: upper[t, i],
                        name="centers")

    dmat = m.addVars(k, n, lb=0,
                     ub=lambda i, j: dmat_max[i, j],
                     name="dmat")

    lambda_vars = m.addVars(k, n, vtype=GRB.BINARY, name="lambda")
    costs = m.addVars(n, lb=0, name="costs")

    for j in range(k - 1):
        m.addConstr(centers[0, j] <= centers[0, j + 1])

    for i in range(k):
        for j in range(n):
            m.addConstr(
                dmat[i, j] >=
                sum((X[t, j] - centers[t, i]) ** 2 for t in range(d))
            )

            m.addConstr(costs[j] - dmat[i, j] >=
                        -dmat_max[i, j] * (1 - lambda_vars[i, j]))

            m.addConstr(costs[j] - dmat[i, j] <=
                        dmat_max[i, j] * (1 - lambda_vars[i, j]))

    for j in range(n):
        m.addConstr(sum(lambda_vars[i, j] for i in range(k)) == 1)

    m.setObjective(sum(costs[j] for j in range(n)), GRB.MINIMIZE)
    m.optimize()

    centers_val = np.array([
        [centers[i, j].X for j in range(k)]
        for i in range(d)
    ])

    objv, _ = obj_assign(centers_val, X)

    return centers_val, objv, m.NodeCount, m.MIPGap