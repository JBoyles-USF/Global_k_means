import numpy as np


class Node:
    def __init__(self, lower, upper, level,
                 groups=None, lambda_=None,
                 group_centers=None, LB=None):

        self.lower = lower          # (d, k) numpy array
        self.upper = upper          # (d, k) numpy array
        self.level = level

        self.groups = groups
        self.lambda_ = lambda_
        self.group_centers = group_centers

        # MUST be None until evaluated
        self.LB = LB

    def __repr__(self):
        return (
            f"Node(level={self.level}, "
            f"LB={self.LB}, "
            f"lower_shape={None if self.lower is None else self.lower.shape}, "
            f"upper_shape={None if self.upper is None else self.upper.shape})"
        )


# ----------------------------------------------------------
# Default Node constructor
# ----------------------------------------------------------
def default_node():
    """
    Create an uninitialized node.
    LB must be None.
    """
    return Node(
        lower=None,
        upper=None,
        level=-1,
        groups=None,
        lambda_=None,
        group_centers=None,
        LB=None
    )


# ----------------------------------------------------------
# Print Node list (safe version)
# ----------------------------------------------------------
def print_node_list(node_list):
    for node in node_list:
        print("Level:", node.level)
        print("LB:", node.LB)

        if node.lower is not None:
            print("Lower bounds shape:", node.lower.shape)
            print(node.lower)

        if node.upper is not None:
            print("Upper bounds shape:", node.upper.shape)
            print(node.upper)

        print("-" * 40)


# ----------------------------------------------------------
# Simple test
# ----------------------------------------------------------
if __name__ == "__main__":

    lower = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

    upper = np.array([[5.0, 6.0],
                      [7.0, 8.0]])

    node1 = Node(lower, upper, level=1)
    node2 = Node(lower, upper, level=2, LB=10.5)

    nodes = [node1, node2]

    print_node_list(nodes)