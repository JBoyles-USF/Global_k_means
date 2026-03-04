import numpy as np
import time
import ub_functions
import lb_functions
import branch
from Nodes import Node

maxiter = 5000000
tol = 1e-6
mingap = 1e-3
time_lapse = 4 * 3600  # 4 hours


def time_finish(seconds):
    return int(seconds * 1e9 + time.time_ns())


# ----------------------------------------------------------
# TRUE best-bound node selection
# ----------------------------------------------------------
def getGlobalLowerBound(nodeList):

    best_value = float("inf")
    best_id = -1

    for idx, node in enumerate(nodeList):

        # Skip nodes without LB (should not happen after fix)
        if node.LB is None:
            continue

        if node.LB < best_value:
            best_value = node.LB
            best_id = idx

    # Fallback safety
    if best_id == -1:
        return None, 0

    return best_value, best_id


# ----------------------------------------------------------
# MAIN BRANCH & BOUND
# ----------------------------------------------------------
def branch_bound(X, k, method="CF", mode="fixed", solver="Gurobi"):

    d, n = X.shape

    # Optional scaling
    x_max = X.max()
    tnsf_max = False
    if x_max >= 20:
        tnsf_max = True
        X = X / (x_max * 0.05)

    # ----------------------------------------------------------
    # Root bounds initialization
    # ----------------------------------------------------------
    data_min = np.min(X, axis=1)
    data_max = np.max(X, axis=1)

    lower = np.zeros((d, k))
    upper = np.zeros((d, k))

    for i in range(k):
        lower[:, i] = data_min
        upper[:, i] = data_max

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------
    UB = 1e15
    centers = None
    assign = None
    w_sos = None

    root = Node(lower.copy(), upper.copy(), 0, None, None, None, None)
    nodeList = [root]

    iter = 0
    global_LB = 0.0

    print(f"{'iter':<6} {'left':<6} {'lev':<6} "
          f"{'LB':<14} {'UB':<14} {'gap':<10}")

    end_time = time_finish(time_lapse)
    calcInfo = []

    # ----------------------------------------------------------
    # INITIAL UPPER BOUND
    # ----------------------------------------------------------
    centers, assign, UB = ub_functions.getUpperBound(
        X, k, None, None, tol
    )

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------
    while nodeList:

        # --------------------------------------------------
        # 1 - Compute LB for ALL open nodes (no lazy eval)
        # --------------------------------------------------
        for n in nodeList:
            if n.LB is None:

                if method == "SG":
                    node_LB, groups, _, group_centers = (
                        lb_functions.getLowerBound_adptGp_LD(
                            X, k, w_sos, assign,
                            n, UB, mode, solver, False
                        )
                    )
                    n.LB = node_LB
                    n.groups = groups
                    n.group_centers = group_centers

                else:  # CF analytic
                    node_LB, group_centers = (
                        lb_functions.getLowerBound_analytic(
                            X, k, n.lower, n.upper
                        )
                    )
                    n.LB = node_LB
                    n.group_centers = group_centers

        # --------------------------------------------------
        # 2 - Select true best-bound node
        # --------------------------------------------------
        _, nodeid = getGlobalLowerBound(nodeList)
        node = nodeList.pop(nodeid)

        # --------------------------------------------------
        # Compute global LB
        # --------------------------------------------------
        if nodeList:
            global_LB = min([n.LB for n in nodeList] + [node.LB])
        else:
            global_LB = node.LB

        # --------------------------------------------------
        # Compute gap
        # --------------------------------------------------
        if UB != 0:
            gap_percentage = (UB - global_LB) / abs(UB) * 100
        else:
            gap_percentage = 0.0

        print(f"{iter:<6} {len(nodeList):<6} {node.level:<6} "
              f"{global_LB:<14.4f} {UB:<14.4f} {gap_percentage:<10.4f} %")

        calcInfo.append(
            [iter, len(nodeList), node.level,
             global_LB, UB, gap_percentage]
        )

        # Stopping conditions
        if iter >= maxiter or time.time_ns() >= end_time:
            break

        if gap_percentage <= mingap:
            break

        iter += 1

        # --------------------------------------------------
        # Prune
        # --------------------------------------------------
        if node.LB >= UB:
            continue

        # --------------------------------------------------
        # UB refinement (CF mode)
        # --------------------------------------------------
        if method == "CF":

            node_centers, node_assign, node_UB = (
                ub_functions.getUpperBound(
                    X, k, node.lower, node.upper, tol
                )
            )

            if node_UB < UB:
                UB = node_UB
                centers = node_centers
                assign = node_assign

        # --------------------------------------------------
        # Branch
        # --------------------------------------------------
        if method == "SG":
            bVarIdx, bVarIdy = branch.SelectVardMaxLBCenterRange(
                node.group_centers
            )
        else:
            bVarIdx, bVarIdy = branch.SelectVarMaxRange(node)

        if bVarIdx is None:
            continue

        print(f"branching on {bVarIdx} {bVarIdy}")

        bValue = (
            node.upper[bVarIdx, bVarIdy]
            + node.lower[bVarIdx, bVarIdy]
        ) / 2

        branch.branch(
            X,
            nodeList,
            bVarIdx,
            bVarIdy,
            bValue,
            node,
            node.LB,
            k
        )

    # ----------------------------------------------------------
    # FINISH
    # ----------------------------------------------------------
    if not nodeList:
        print("all nodes solved")

    print(f"solved nodes: {iter}")
    print(f"{iter:<52} {global_LB:<14.4e} {UB:<14.4e}")
    print("centers", centers)

    if tnsf_max:
        UB = UB * (x_max * 0.05) ** 2

    return centers, UB, calcInfo