# import necessary modules
import numpy as np
from sklearn.cluster import KMeans  # used instead of Julia's Clustering.kmeans
from opt_functions import local_OPT  # your Python module

def getUpperBound(X, k, lower=None, upper=None, tol=0):
    """
    Compute an upper bound on clustering cost.

    If lower/upper bounds are not given, uses standard k-means trials.
    Otherwise, uses local optimization (Gurobi).

    Parameters:
        X: np.array of shape (d, n)
        k: number of clusters
        lower, upper: optional bounds for cluster centers
        tol: tolerance for improvement
    
    Returns:
        centers: np.array of shape (d, k)
        assign: np.array of shape (n,)
        UB: float, total cost
    """
    n_trial = 10
    d, n = X.shape

    if lower is None or upper is None:
        UB = np.inf
        best_result = None
        for _ in range(n_trial):
            # sklearn KMeans expects (n_samples, n_features) => transpose X
            kmeans_rlt = KMeans(n_clusters=k, n_init=1, init='random', random_state=None).fit(X.T)
            total_cost = kmeans_rlt.inertia_  # sum of squared distances
            if total_cost <= UB - tol:
                UB = total_cost
                best_result = kmeans_rlt

        centers = best_result.cluster_centers_.T  # shape (d, k)
        assign = best_result.labels_
    else:
        # use Gurobi local optimization
        centers, assign, UB = local_OPT(X, k, lower, upper)

    return centers, assign, UB
