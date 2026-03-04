import numpy as np
from sklearn.cluster import KMeans
from opt_functions import local_OPT


# ----------------------------------------------------------
# Upper Bound computation
# ----------------------------------------------------------
def getUpperBound(X, k, lower=None, upper=None, tol=0):
    """
    X must be (d, n)
    Returns:
        centers: (d, k)
        assign: (n,)
        UB: float
    """

    d, n = X.shape

    n_trial = 10
    UB = np.inf
    result = None

    #print(f"Starting upper bound calculation with {n_trial} trials.")

    # ------------------------------------------------------
    # Case 1: No box bounds → run random k-means
    # ------------------------------------------------------
    if lower is None:

        X_km = X.T  # sklearn expects (n_samples, n_features)

        for tr in range(n_trial):
            print(f"Trial {tr + 1} of {n_trial}...")

            kmeans = KMeans(
                n_clusters=k,
                init="random",
                n_init=1,
                random_state=None
            ).fit(X_km)

            total_cost = kmeans.inertia_

            print(f"Trial {tr + 1} - Total cost: {total_cost}")

            if total_cost <= UB - tol:
                UB = total_cost
                result = kmeans

        print(f"Best UB found: {UB}")

        centers = result.cluster_centers_.T  # (d, k)
        assign = result.labels_              # (n,)

    # ------------------------------------------------------
    # Case 2: Box bounds given → solve local NLP
    # ------------------------------------------------------
    else:
        centers, assign, UB = local_OPT(X, k, lower, upper)

    #print("Returning centers, assignments, and UB.")
    return centers, assign, UB