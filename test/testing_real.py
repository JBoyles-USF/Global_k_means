# testing_real.py
import sys
import os
import time
import numpy as np
import random
from sklearn import datasets
import pandas as pd

# ------------------------------------------------------------
# Argument structure
# ------------------------------------------------------------

if len(sys.argv) < 5:
    raise ValueError(
        "Usage: python testing_real.py <cores|HPC> <k> <dataset_name> <method>"
    )

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]

# ------------------------------------------------------------
# Parallel setup
# ------------------------------------------------------------

if arg1 == "HPC":
    num_cores = int(os.environ.get("SLURM_NTASKS", "1"))
else:
    num_cores = int(arg1)

print(f"Running {num_cores} processes")

# ------------------------------------------------------------
# Add src/ to path
# ------------------------------------------------------------

if "src" not in sys.path:
    sys.path.append("src")

from data_process import convertlabel
from bb_functions import branch_bound
from opt_functions import global_OPT_base

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_data(dataset_name):
    if dataset_name == "iris":
        iris = datasets.load_iris()
        data = iris.data              # (n, d)
        label = iris.target
    else:
        data = pd.read_csv(os.path.join("data", f"{dataset_name}.csv"))
        label = data.pop("label")
        data = data.values            # (n, d)
        label = label.values

    # 🔥 CRITICAL FIX: convert to (d, n)
    data = data.T

    return data, label


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    random.seed(123)
    np.random.seed(123)

    dataset_name = arg3
    k = int(arg2)

    data, label = load_data(dataset_name)

    print("Data shape (d, n):", data.shape)

    # --------------------------------------------------------
    # Base global optimization
    # --------------------------------------------------------
    if arg4 == "base-BB":

        start_time = time.time()
        centers_g, objv_g, iter_g, gap_g = global_OPT_base(data, k)
        t_g = time.time() - start_time

        print(
            f"{dataset_name}:\t"
            f"{round(objv_g, 2)}\t"
            f"{round(t_g, 2)}\t"
            f"{round(gap_g, 4)}%\t"
            f"{iter_g}"
        )

    # --------------------------------------------------------
    # Branch and bound variants
    # --------------------------------------------------------
    else:

        start_time = time.time()

        centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = branch_bound(
            data, k, arg4, "fixed", "Gurobi"
        )

        t_adp_LD = time.time() - start_time

        final_gap = calcInfo_adp_LD[-1][-1]

        if final_gap <= 0.001:
            iter_count = len(calcInfo_adp_LD) - 1
        else:
            iter_count = len(calcInfo_adp_LD)

        print(
            f"{dataset_name}:\t"
            f"{round(objv_adp_LD, 2)}\t"
            f"{round(t_adp_LD, 2)}\t"
            f"{round(final_gap * 100, 4)}%\t"
            f"{iter_count}"
        )


if __name__ == "__main__":
    main()