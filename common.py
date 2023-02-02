from load_tree import Node

import pandas as pd
import numpy as np
import joblib
from bitarray import bitarray

# new Imports I (Angel) added
from load_tree import forestConversion

# ------------------------------Algorithm 1 ----------------------------------------
# get the number of leaf nodes of decision tree
def leaf(tree: Node):
    if tree is None:
        return 0
    elif tree.left_child is None and tree.right_child is None:
        return 1
    else:
        return leaf(tree.left_child) + leaf(tree.right_child)


# True if internal node
def is_leaf(node):
    if node.right_child is None and node.left_child is None:
        return False
    return True

# create a binary tree
# def Creat_Tree(Root: Node, vals):
#     if len(vals) == 0:
#         return Root
#     if vals[0] != '#':
#         Root = Node(vals[0])
#         vals.pop(0)
#         Root.left_child = Creat_Tree(Root.left_child, vals)
#         Root.right_child = Creat_Tree(Root.right_child, vals)
#         return Root
#     else:
#         Root = None
#         vals.pop(0)
#         return Root

# output the bitvector of internal node
def generate_bitvector(tree: Node) -> list[bitarray]:
    bit_list = []
    if tree is None:
        # tree.bitVector = bit_list
        return tree
    stack = []
    #tree.left_leaf_node_num = 0 #temporarily remove this line
    stack.append(tree)
    N = leaf(tree)
    while len(stack) != 0:
        node = stack.pop()
        x = leaf(node.left_child)
        s = '1' * node.left_leaf_node_num + '0' * x + '1' * (           #left_leaf_node_num is probably # of leaf nodes
                N - node.left_leaf_node_num - x)
        if is_leaf(node):
            bit_list.append(s)
        left_node = node.left_child
        right_node = node.right_child
        if right_node is not None:
            right_node.left_leaf_node_num = node.left_leaf_node_num + x
            stack.append(right_node)
        if left_node is not None:
            left_node.left_leaf_node_num = node.left_leaf_node_num
            stack.append(left_node)
    # Return the bitvector list for the tree
    tree.bitVector = bit_list
    return tree

# ----------------------------------------end-----------------------------------------
def sort_by_feature_threhold(
    feature: pd.DataFrame,
    thresholds: float,
    tree_ids: int,
    bitvectors: list[list[int]],
    offsets: list[int],
    v: list[list[int]],
    leaves: list[int],
) -> int:
    pass


def quickscorer(forest, feature) -> int:
    pass


def test():
    def get_test_model():
        sklearn_model = joblib.load("test_utils/models/higgs_randomforest_10_8.pkl")
        forest = None
        return (forest, sklearn_model)

    forest, sklearn_model = get_test_model()

    features = pd.read_csv(
        "test_utils/test_data/test_samples.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )

    for feature in features:
        prediction1 = quickscorer(forest, feature)
        prediction2 = int(sklearn_model.predict())
        assert (
            prediction1 == prediction2
        ), f"Prediction from our implementation is {prediction1} and prediction from sklearn model is {prediction2}"


if __name__ == "__main__":
    #test()
    # create the tree and store it
    myTree=forestConversion()

    # create the bitvector and store it
    i=0
    myBitVector=[]
    while i < len(myTree):
        myBitVector = generate_bitvector(myTree[i])


    myBitVector
