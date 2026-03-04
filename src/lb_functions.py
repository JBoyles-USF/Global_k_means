import numpy as np
from sklearn.cluster import KMeans
import random
import math


tol = 1e-6
mingap = 1e-3


# -----------------------------
# Median of 3 values
# -----------------------------
def med(a, b, c):
    return a + b + c - max(a, b, c) - min(a, b, c)


# -----------------------------
# getGlobalLowerBound
# -----------------------------
def getGlobalLowerBound(nodeList):

    best_value = float("inf")
    best_id = -1

    for idx, node in enumerate(nodeList):

        if node.LB is None:
            continue

        if node.LB < best_value:
            best_value = node.LB
            best_id = idx

    # If all nodes have LB None → just return first
    if best_id == -1:
        return None, 0

    return best_value, best_id


# -----------------------------
# unique_inverse (faithful)
# -----------------------------
def unique_inverse(A):
    out = []
    out_idx = []
    seen = {}

    for idx, x in enumerate(A):
        if x not in seen:
            seen[x] = len(seen)
            out.append(x)
            out_idx.append([])
        out_idx[seen[x]].append(idx)

    return out, out_idx


# -----------------------------
# strGrp_nofill (faithful)
# -----------------------------
def strGrp_nofill(assign, ngroups):
    l, c = unique_inverse(assign)
    groups = [[] for _ in range(ngroups)]
    ng = 0

    for i in range(len(l)):
        perm = np.random.permutation(len(c[i]))
        p = [c[i][j] for j in perm]

        for j in range(len(p)):
            if ng == ngroups:
                ng = 0
            groups[ng].append(p[j])
            ng += 1

    return groups


# -----------------------------
# kmeans_group (FIXED ORIENTATION)
# -----------------------------
def kmeans_group(X, assign, ngroups):
    """
    X must be (d, n)
    """

    np.random.seed(123)

    _, clst_idx = unique_inverse(assign)
    groups = [[] for _ in range(ngroups)]

    for i in range(len(clst_idx)):

        if len(clst_idx[i]) == 1:
            groups[0].append(clst_idx[i][0])
        else:
            k_sub = math.floor(len(clst_idx[i]) / ngroups)
            if k_sub <= 1:
                k_sub = 2

            # FIX: select columns, then transpose for sklearn
            X_sub = X[:, clst_idx[i]].T  # (m, d)

            n_trial = 5
            mini_cost = float("inf")
            best_assign = None

            for _ in range(n_trial):
                kmeans = KMeans(n_clusters=k_sub, n_init=10, random_state=None)
                kmeans.fit(X_sub)

                cost = kmeans.inertia_
                if cost <= mini_cost:
                    mini_cost = cost
                    best_assign = kmeans.labels_

            clst_groups = strGrp_nofill(best_assign, ngroups)

            for j in range(len(clst_groups)):
                for idx_local in clst_groups[j]:
                    groups[j].append(clst_idx[i][idx_local])

    return groups


# -----------------------------
# getLowerBound_analytic (STABLE VERSION)
# -----------------------------
def getLowerBound_analytic(X, k, lower=None, upper=None):

    d, n = X.shape

    if lower is None or upper is None:
        from opt_functions import init_bound
        lower, upper = init_bound(X, d, k)

    LB = 0.0

    centers_gp = np.zeros((d, k, n))

    for s in range(n):

        x = X[:, s]

        # replicate x across clusters
        x_mat = np.tile(x.reshape(d, 1), (1, k))

        # elementwise median(lower, x, upper)
        mu = lower + x_mat + upper \
             - np.maximum(np.maximum(lower, x_mat), upper) \
             - np.minimum(np.minimum(lower, x_mat), upper)

        centers_gp[:, :, s] = mu

        min_dist = np.inf

        for i in range(k):
            diff = x - mu[:, i]
            dist = np.sum(diff * diff)
            if dist <= min_dist:
                min_dist = dist

        LB += min_dist

    return LB, centers_gp