# data_process.py
# Faithful Python translation of original Julia data_process module

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score as randindex


# ------------------------
# Data preprocessing
# ------------------------
def data_preprocess(dataname, datapackage="datasets", path=None,
                    missingchar=None, header=False):
    """
    Read and preprocess data.
    Returns:
        X : (d, n) feature matrix
        y : 1-based label vector
    """
    import pandas as pd
    from pydataset import data as r_dataset

    if path is None:
        df = r_dataset(dataname, package=datapackage)
    else:
        df = pd.read_csv(path,
                         header=0 if header else None,
                         na_values=missingchar)
        df = df.dropna()

    # Features
    X = df.iloc[:, :-1].to_numpy().T   # transpose → (d, n)

    # Labels
    y = df.iloc[:, -1].to_numpy()

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y) + 1  # make 1-based
    else:
        unique_vals = np.unique(y)
        mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        y = np.array([mapping[val] for val in y])

    return X, y


# ------------------------
# Sigma generator
# ------------------------
def sig_gen(eigvals):
    """
    Generate random covariance matrix with specified eigenvalues.
    """
    n = len(eigvals)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    D = np.diag(eigvals)
    return Q @ D @ Q.T


# ------------------------
# Update centers (k-means style)
# ------------------------
def update_centers(X, assign, k):
    """
    X shape: (d, n)
    assign: 1-based labels
    """
    d, n = X.shape
    centers = np.zeros((d, k))
    counts = np.zeros(k)

    for j in range(n):
        cj = int(assign[j]) - 1
        centers[:, cj] += X[:, j]
        counts[cj] += 1

    for i in range(k):
        if counts[i] > 0:
            centers[:, i] /= counts[i]

    return centers


# ------------------------
# Get centers and cost
# ------------------------
def get_center_cost(X, assign, k):
    """
    Compute cluster centers and total SSE cost.
    """
    d, n = X.shape
    centers = np.zeros((d, k))
    cost = 0.0

    for i in range(k):
        idx = np.where(assign == i + 1)[0]

        if len(idx) == 0:
            continue

        ci = X[:, idx]
        centers[:, i] = np.mean(ci, axis=1)

        diff = ci - centers[:, i:i + 1]
        cost += np.sum(diff ** 2)

    return centers, cost


# ------------------------
# Convert label function
# ------------------------
def convertlabel(label_range, labels):
    """
    Map labels to consecutive integers starting from 1.
    """
    labels = np.array(labels)
    label_map = {old: new for new, old in enumerate(label_range, start=1)}
    return np.array([label_map[l] for l in labels])


# ------------------------
# Compute normalized mutual information
# ------------------------
def compute_nmi(z1, z2):
    z1 = np.array(z1)
    z2 = np.array(z2)
    n = len(z1)

    unique1 = np.unique(z1)
    unique2 = np.unique(z2)

    pk1 = np.array([np.sum(z1 == val) / n for val in unique1])
    pk2 = np.array([np.sum(z2 == val) / n for val in unique2])

    nk12 = np.zeros((len(unique1), len(unique2)))

    for i, val1 in enumerate(unique1):
        for j, val2 in enumerate(unique2):
            nk12[i, j] = np.sum((z1 == val1) & (z2 == val2))

    pk12 = nk12 / n

    eps = np.finfo(float).eps

    Hx = -np.sum(pk1 * np.log(pk1 + eps))
    Hy = -np.sum(pk2 * np.log(pk2 + eps))
    Hxy = -np.sum(pk12 * np.log(pk12 + eps))

    MI = Hx + Hy - Hxy

    return MI / (0.5 * (Hx + Hy))


# ------------------------
# Clustering evaluation
# ------------------------
def cluster_eval(z1, z2):
    z1 = np.array(z1)
    z2 = np.array(z2)

    nmi = compute_nmi(z1, z2)
    print("nmi:", nmi)

    from sklearn.metrics import mutual_info_score
    from scipy.stats import entropy

    MI = mutual_info_score(z1, z2)

    Hz1 = entropy(np.bincount(z1) / len(z1))
    Hz2 = entropy(np.bincount(z2) / len(z2))

    vi = Hz1 + Hz2 - 2 * MI

    ari = randindex(z1, z2)
    print("ari:", ari)

    return nmi, vi, ari


# ------------------------
# Nested evaluation
# ------------------------
def nestedEval(X, label, centers, objv, result):
    """
    Compare k-means result and branch-and-bound result.
    """
    k = centers.shape[1]

    # Cost of true labels
    centers_o, objv_o = get_center_cost(X, label, k)

    # Cost of BB assignments
    centers_bb, objv_f = get_center_cost(X, result.assignments, k)
    assign_bb = result.assignments

    print("kmeans:", result.assignments)
    print("opt_km:", assign_bb)

    nmi_kb, vi_kb, ari_kb = cluster_eval(result.assignments, assign_bb)
    nmi_km, vi_km, ari_km = cluster_eval(result.assignments, label)
    nmi_bb, vi_bb, ari_bb = cluster_eval(assign_bb, label)

    print("km_cost:", result.totalcost)
    print("bb_cost:", objv, "objective cost:", objv_f)
    print("real_cost:", objv_o)

    nestedEvalRlt = np.array([
        [nmi_kb, nmi_bb],
        [vi_kb, vi_bb],
        [ari_kb, ari_bb],
        [objv_o, objv]
    ])

    return nestedEvalRlt