# module branch

# using Nodes
# export branch!

# import Node class from your Nodes module
# from Nodes import Node
import numpy as np
from Nodes import Node

# function SelectVarMaxRange(node)
#     dif = node.upper -node.lower
#     ind = findmax(dif)[2]
#     return ind[1], ind[2]
def SelectVarMaxRange(node):
    dif = node.upper - node.lower
    idx = np.unravel_index(np.argmax(dif, axis=None), dif.shape)
    return idx  # returns (row_idx, col_idx) in Python

# function SelectVardMaxLBCenterRange(group_centers)
#     d, k, ~ = size(group_centers)
#     dif = zeros(d, k)
#     for dim in 1:d
#         for clst in 1:k
#             dif[dim, clst] = maximum(group_centers[dim, clst,:]) - minimum(group_centers[dim, clst,:])
#         end
#     end
#     ind = findmax(dif)[2]
#     return ind[1], ind[2]
def SelectVardMaxLBCenterRange(group_centers):
    d, k, _ = group_centers.shape
    dif = np.zeros((d, k))
    for dim in range(d):
        for clst in range(k):
            dif[dim, clst] = np.max(group_centers[dim, clst, :]) - np.min(group_centers[dim, clst, :])
    idx = np.unravel_index(np.argmax(dif, axis=None), dif.shape)
    return idx  # (dim_idx, cluster_idx)

# function branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k)
def branch_(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k):
    d, n = X.shape
    lower = node.lower.copy()
    upper = node.upper.copy()

    # split from this variable at bValue
    upper[bVarIdx, bVarIdy] = bValue
    # bound tightening to avoid symmetric solutions
    for j in range(1, k):
        if upper[0, k-j-1] >= upper[0, k-j]:
            upper[0, k-j-1] = upper[0, k-j]

    if np.sum(lower <= upper) == d * k:
        left_node = Node(lower.copy(), upper.copy(), node.level+1, node_LB,
                         node.groups, node.lambda_, node.group_centers)
        nodeList.append(left_node)
        # print("left_node:", lower, upper)

    # create right node
    lower = node.lower.copy()
    upper = node.upper.copy()
    lower[bVarIdx, bVarIdy] = bValue
    for j in range(1, k):
        if lower[0, j] <= lower[0, j-1]:
            lower[0, j] = lower[0, j-1]

    if np.sum(lower <= upper) == d * k:
        right_node = Node(lower.copy(), upper.copy(), node.level+1, node_LB,
                          node.groups, node.lambda_, node.group_centers)
        nodeList.append(right_node)
        # print("right_node:", lower, upper)
