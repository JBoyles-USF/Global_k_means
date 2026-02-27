# module Nodes

# using Printf
# export Node

# struct Node
#     lower
#     upper
#     level::Int
#     LB::Float64
#     groups
#     lambda
#     group_centers
# end
#
# Node() = Node(nothing, nothing, -1, -1e15, nothing, nothing, nothing)

class Node:
    """
    Node class equivalent to Julia struct Node
    """
    def __init__(self, lower=None, upper=None, level=-1, LB=-1e15, groups=None, lambda_=None, group_centers=None):
        self.lower = lower
        self.upper = upper
        self.level = level
        self.LB = LB
        self.groups = groups
        self.lambda_ = lambda_   # `lambda` is a Python keyword
        self.group_centers = group_centers

# function to print the node in a neat form
# function printNodeList(nodeList)
#     for i in 1:length(nodeList)
#         println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:lower))) # reserve 3 decimal precision
#         println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:upper)))
#         println(getfield(nodeList[i],:level)) # integer
#         println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:LB)))
#     end
# end

def printNodeList(nodeList):
    for node in nodeList:
        if node.lower is not None:
            print([f"{x:.3f}" for x in node.lower])
        else:
            print(None)
        if node.upper is not None:
            print([f"{x:.3f}" for x in node.upper])
        else:
            print(None)
        print(node.level)
        print(f"{node.LB:.3f}")
