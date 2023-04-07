from load_tree import Node
import load_tree
import pandas as pd
import numpy as np
import joblib
from bitarray import bitarray


# get the number of leaf node
def leaf(tree: Node):
    if tree is None:
        return 0
    elif tree.left_child is None and tree.right_child is None:
        return 1
    else:
        return leaf(tree.left_child) + leaf(tree.right_child)


# True if not leaf node
def is_leaf(node):
    if node.right_child is None and node.left_child is None:
        return False
    return True


# generate the bitvector
def generate_bitvector(tree: Node) -> list[bitarray]:
    bit_list = []  # a list to store bitvector
    if tree is None:  # empty if only root node
        tree.bitvector = bit_list
        return bit_list
    stack = []
    tree.left_leaf_node_num = 0
    stack.append(tree)  # append the root node
    N = leaf(tree)  # get the number of leaf nodes
    while len(stack) != 0:
        node = stack.pop()
        x = leaf(node.left_child)
        s = '1' * node.left_leaf_node_num + '0' * x + '1' * (
                N - node.left_leaf_node_num - x)
        if is_leaf(node):
            b = bitarray(s)
            bit_list.append(b)
            node.bitVector = b
        left_node = node.left_child
        right_node = node.right_child
        if right_node is not None:
            right_node.left_leaf_node_num = node.left_leaf_node_num + x
            stack.append(right_node)
        if left_node is not None:
            left_node.left_leaf_node_num = node.left_leaf_node_num
            stack.append(left_node)
    tree.bitvector = bit_list
    return bit_list


# find the false node
def Find_false(tree: Node, x: list):
    result_list = []
    if tree is None:
        return result_list
    stack = []
    stack.append(tree)
    while len(stack) != 0:
        node = stack.pop()
        if is_leaf(node):
            i = node.feature_name
            if type(i) == str:
                j = i[1:]
                k = int(j)
            elif type(i) == int:
                k = i
            if x[k] > float(node.threhold):
                result_list.append(node.bitVector)
        left_node = node.left_child
        right_node = node.right_child
        if right_node is not None:
            stack.append(right_node)
        if left_node is not None:
            stack.append(left_node)
    return result_list


# apply the AND operator
def And(re_list):
    if re_list is None:
        return None
    i = len(re_list[0])
    s = '1' * i
    bit = bitarray(s)
    for x in re_list:
        bit = bit & x
    return bit


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

# get prediction
def bit_prediction(node, bit_array):
    prediction = 0.0
    i = 0
    if node is None:
        return 0.0
    stack = []
    stack.append(node)
    while len(stack) != 0:
        node = stack.pop()
        if node.left_child is None and node.right_child is None:
            if int(bit_array[i]) == 1:
                prediction += node.gain
                return prediction
            i = i + 1
        left_node = node.left_child
        right_node = node.right_child
        if right_node is not None:
            stack.append(right_node)
        if left_node is not None:
            stack.append(left_node)
    return prediction

def quickscorer(forest, feature) -> int:
    pass

def quickscorer_without_sorting(forest, feature) -> int:
    prediction = 0.0
    for node in forest:
        generate_bitvector(node)
        false_list = Find_false(node, feature)
        bit_array = And(false_list)
        prediction = bit_prediction(node, bit_array) + prediction

    return 1 if prediction > 0 else 0

# test the bitvector
def bit_test():
    forest, sklearn_model = load_tree.parse_from_pickle()

    features = pd.read_csv(
        "test_utils/test_data/test_samples.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )

    for i, feature in features.iterrows():
        feature = feature.to_numpy()
        prediction1 = quickscorer_without_sorting(forest, feature)
        prediction2 = int(sklearn_model.predict(np.expand_dims(feature, 0)))
        print(f"Test{i}: Predition1 {prediction1}, Prediction2 {prediction2}")
        assert prediction1 == prediction2

# def test():
#     def get_test_model():
#         sklearn_model = joblib.load("test_utils/models/higgs_xgboost_10_8.pkl")
#         forest = None
#         return (forest, sklearn_model)

#     forest, sklearn_model = get_test_model()

#     features = pd.read_csv(
#         "test_utils/test_data/test_samples.csv",
#         dtype=np.float32,
#         usecols=range(1, 29),
#         header=None,
#     )

#     for feature in features:
#         prediction1 = quickscorer(forest, feature)
#         prediction2 = int(sklearn_model.predict())
#         assert (
#             prediction1 == prediction2
#         ), f"Prediction from our implementation is {prediction1} and prediction from sklearn model is {prediction2}"


if __name__ == "__main__":
#    test()
    bit_test()