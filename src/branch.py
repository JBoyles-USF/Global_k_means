import numpy as np
from Nodes import Node


# ----------------------------------------------------------
# Select variable with maximum box width (CF / analytic)
# Julia equivalent: SelectVarMaxRange(node)
# ----------------------------------------------------------
def SelectVarMaxRange(node):
    """
    Select (dimension, cluster) with largest box width.
    Avoid branching on zero-width boxes.
    """
    dif = node.upper - node.lower

    # Ignore nearly-zero widths
    mask = dif > 1e-12

    if not np.any(mask):
        return None, None

    # Set invalid entries to -inf so argmax ignores them
    dif_masked = np.where(mask, dif, -np.inf)

    ind = np.unravel_index(np.argmax(dif_masked), dif.shape)
    return ind[0], ind[1]


# ----------------------------------------------------------
# Select variable with maximum LB center range (SG method)
# Julia equivalent: SelectVardMaxLBCenterRange(group_centers)
# group_centers shape must be (d, k, g)
# ----------------------------------------------------------
def SelectVardMaxLBCenterRange(group_centers):

    d, k, g = group_centers.shape
    dif = np.zeros((d, k))

    for dim in range(d):
        for clst in range(k):
            dif[dim, clst] = (
                np.max(group_centers[dim, clst, :]) -
                np.min(group_centers[dim, clst, :])
            )

    ind = np.unravel_index(np.argmax(dif), dif.shape)
    return ind[0], ind[1]


# ----------------------------------------------------------
# Branching function
# Julia equivalent: branch!
# ----------------------------------------------------------
def branch(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k):

    # NOTE:
    # X is (d, n) but we do NOT use it here.
    # Branching only modifies box bounds.

    # ======================================================
    # LEFT CHILD  (upper bound tightened)
    # ======================================================
    lower_left = node.lower.copy()
    upper_left = node.upper.copy()

    upper_left[bVarIdx, bVarIdy] = bValue

    # Feasibility check
    if np.all(lower_left <= upper_left):

        left_node = Node(
            lower_left,
            upper_left,
            node.level + 1,
            None,   # groups
            None,   # lambda
            None,   # group_centers
            None    # LB must be None (recompute later)
        )

        nodeList.append(left_node)

    # ======================================================
    # RIGHT CHILD  (lower bound tightened)
    # ======================================================
    lower_right = node.lower.copy()
    upper_right = node.upper.copy()

    lower_right[bVarIdx, bVarIdy] = bValue

    # Feasibility check
    if np.all(lower_right <= upper_right):

        right_node = Node(
            lower_right,
            upper_right,
            node.level + 1,
            None,
            None,
            None,
            None
        )

        nodeList.append(right_node)