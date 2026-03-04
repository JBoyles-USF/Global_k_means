import sys
import os
import time
import numpy as np
import random
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Argument structure (faithful to Julia ARGS)
# arg1: number of cores OR "HPC"
# arg2: number of clusters to solve
# arg3: number of points per cluster
# arg4: lower bound method (base-BB, CF, SG, LD+SG)
# ------------------------------------------------------------

if len(sys.argv) < 5:
    raise ValueError(
        "Usage: python testing_toy.py <cores|HPC> <k> <clst_n> <method>"
    )

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]

# ------------------------------------------------------------
# Parallel setup (structural equivalent to Julia addprocs)
# ------------------------------------------------------------

if arg1 == "HPC":
    num_cores = int(os.environ.get("SLURM_NTASKS", "1"))
else:
    num_cores = int(arg1)

print(f"Running {num_cores} processes")

# ------------------------------------------------------------
# Add src/ to path (faithful to LOAD_PATH push!)
# ------------------------------------------------------------

if "src" not in sys.path:
    sys.path.append("src")

from data_process import convertlabel
from bb_functions import branch_bound
from opt_functions import global_OPT_base


# ------------------------------------------------------------
# Helper function to generate covariance matrix (approximate Julia's sig_gen)
# ------------------------------------------------------------

def sig_gen(sampled):
    """
    Generate a full covariance matrix based on the sampled values for each dimension.
    In Julia, this might correspond to generating a full covariance matrix where 
    each value in the `sampled` vector influences both the diagonal and off-diagonal elements.
    """
    d = len(sampled)
    # Create a full covariance matrix with random off-diagonal elements (simulating the full matrix behavior)
    sig = np.random.rand(d, d) * sampled
    sig = (sig + sig.T) / 2  # Ensure the covariance matrix is symmetric
    np.fill_diagonal(sig, sampled)  # Ensure the diagonal values are from `sampled`
    return sig


# ------------------------------------------------------------
# Main Process
# ------------------------------------------------------------

def main():

    random.seed(1)
    np.random.seed(1)

    clst_n = int(arg3)
    nclst = 3
    d = 2

    # Julia: Array{Float64}(undef, d, clst_n*nclst)
    data = np.empty((d, clst_n * nclst))
    label = np.empty(clst_n * nclst)

    # Julia: mu = reshape(sample(1:30, nclst*d), nclst, d)
    mu = np.reshape(np.random.choice(np.arange(1, 31), size=nclst * d, replace=True), (nclst, d))

    for i in range(nclst):

        # Sample the sig (covariance) matrix
        sampled = np.random.choice(
            np.arange(1, 11),
            size=d,
            replace=True
        )

        sig = np.round(sig_gen(sampled))  # Use the custom sig_gen function
        print(sig)

        mvn = multivariate_normal(mean=mu[i, :], cov=sig)
        clst = mvn.rvs(clst_n).T   # shape d × clst_n

        start = i * clst_n
        end = (i + 1) * clst_n

        data[:, start:end] = clst
        label[start:end] = i + 1   # keep 1-based labels

    # Julia: label = convertlabel(1:nclst, vec(label))
    label = convertlabel(range(1, nclst + 1), label.flatten())

    k = int(arg2)

    # ------------------------------------------------------------
    # Branch-and-Bound or Base-BB
    # ------------------------------------------------------------

    if arg4 == "base-BB":

        start_time = time.time()
        centers_g, objv_g, iter_g, gap_g = global_OPT_base(data, k)
        t_g = time.time() - start_time

        print(
            f"Toy-{nclst}-{clst_n}:\t"
            f"{round(objv_g,2)}\t"
            f"{round(t_g,2)}\t"
            f"{round(gap_g,4)}%\t"
            f"{iter_g}"
        )

    else:

        start_time = time.time()
        centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = \
            branch_bound(data, k, arg4, "fixed", "Gurobi")

        t_adp_LD = time.time() - start_time

        final_gap = calcInfo_adp_LD[-1][-1]

        if final_gap <= 0.001:
            iter_count = len(calcInfo_adp_LD) - 1
        else:
            iter_count = len(calcInfo_adp_LD)

        print(
            f"Toy-{nclst}-{clst_n}:\t"
            f"{round(objv_adp_LD,2)}\t"
            f"{round(t_adp_LD,2)}\t"
            f"{round(final_gap*100,4)}%\t"
            f"{iter_count}"
        )


if __name__ == "__main__":
    main()