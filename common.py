from load_tree import Node, parse_from_pickle
import pandas as pd
import numpy as np
from bitarray import bitarray
from dataclasses import dataclass

@dataclass
class BitvectorNode:
    feature_name: int
    threshold: float
    bitvector: bitarray

@dataclass
class BitvectorTree:
    nodes : list[BitvectorNode]
    output_values: list[int]

# get the number of leaf node
# def leaf(tree: Node):
#     if tree is None:
#         return 0
#     elif tree.left_child is None and tree.right_child is None:
#         return 1
#     else:
#         return leaf(tree.left_child) + leaf(tree.right_child)


# # True if not leaf node
# def is_leaf(node):
#     if node.right_child is None and node.left_child is None:
#         return False
#     return True


# generate the bitvector
def generate_BitvectorTree(tree_root: Node) -> BitvectorTree:
    if tree_root is None:
        raise ValueError('Tree is None')

    bitvectors = []
    output_values = []
    
    get_leaf_count_and_output_values(tree_root, output_values)
    generate_bitvector_helper(tree_root, 0, tree_root.leaf_count, bitvectors)
    return BitvectorTree(bitvectors, output_values)


def get_leaf_count_and_output_values(node, output_values):
    if node.left_child is None and node.right_child is None:
        output_values.append(node.gain)
        node.leaf_count = 1
        return

    if node.left_child:
        get_leaf_count_and_output_values(node.left_child, output_values)
        
    if node.right_child:
        get_leaf_count_and_output_values(node.right_child, output_values)

    node.leaf_count = node.left_child.leaf_count + node.right_child.leaf_count


def generate_bitvector_helper(node, begin, length, bitvectors):
    if node.left_child:
        generate_bitvector_helper(node.left_child, begin, length, bitvectors)
    if node.right_child:
        N = node.left_child.leaf_count if node.left_child else 0
        generate_bitvector_helper(node.right_child, begin + N, length, bitvectors)
    
    if node.left_child or node.right_child:
        bitvector = bitarray('1' * begin + '0' * N + '1' * (length - begin - N))
        # assert len(bitvector) == length, f'{begin}, {N}, {length}'
        bitvectors.append(BitvectorNode(
            node.feature_name,
            node.threshold,
            bitvector
        ))

def sort_by_feature_threshold(
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


def aggregate_vectors(tree: BitvectorTree, features: list[float], offset: int = 0):
    and_vector = bitarray('1' * len(tree.nodes[0].bitvector))
    for node in tree.nodes:
        if 0 <= (index := node.feature_name - offset) < len(features):
            if features[index] > node.threshold:
                and_vector &= node.bitvector
    return and_vector

def tree_prediction(tree: BitvectorTree, features: list[float]) -> float:
    and_vector = aggregate_vectors(tree, features)
    for i, bit in enumerate(and_vector):
        if bit == 1:
            return tree.output_values[i]

def quickscorer_without_sorting(bitvector_trees, features) -> int:
    prediction = 0.0

    for tree in bitvector_trees:
        prediction += tree_prediction(tree, features)
        # false_list = Find_false(node, feature)
        # bit_array = And(false_list)
        # prediction = bit_prediction(node, bit_array) + prediction

    return 1 if prediction > 0 else 0

def get_forest_model():
    forest, _ = parse_from_pickle() 
    bitvector_trees = []
    for tree_root in forest:
        bitvector_trees.append(generate_BitvectorTree(tree_root)) 
    return bitvector_trees

# test the bitvector
def bit_test():

    features = pd.read_csv(
        "test_utils/test_data/HIGGS_test.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    ).values
    
    forest, sklearn_model = parse_from_pickle()
    bitvector_trees = []
    for node in forest:
        bitvector_trees.append(generate_BitvectorTree(node))

    for i, feature in enumerate(features):
        prediction1 = quickscorer_without_sorting(bitvector_trees, feature)
        prediction2 = int(sklearn_model.predict(np.expand_dims(feature, 0)))
        print(f"Test{i}: Predition1 {prediction1}, Prediction2 {prediction2}")
        assert prediction1 == prediction2


if __name__ == "__main__":
    bit_test()