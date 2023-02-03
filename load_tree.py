from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import csv


class Node:
    def __init__(self):
        self.id = None
        self.feature_name = None  # feature number
        self.threhold = None  # split
        self.left_child = None  # yes
        self.right_child = None  # no
        self.site_name = None
        self.gain = None
        self.bitVector = None


def parse_from_pickle(
        model_path: Optional[str] = "test_utils/models/higgs_randomforest_10_8.pkl",
) -> tuple[list[Node], RandomForestClassifier]:
    sklearn_model = joblib.load(model_path)
    forest = []
    # TODO: extract trees from sklearn model and store them to forest

    return (forest, sklearn_model)


def load_from_protobuf():
    pass


# --------------------------------------- converting to forest from csv-----------------------------------------------
# ------- print tree -------------
# simply used for verifying that all cells were used
# print methods accreddited to this source https://www.geeksforgeeks.org/level-order-tree-traversal/
def printLevelOrder(root: Node):
    h = height(root)
    for i in range(1, h + 1):
        printCurrentLevel(root, i)


def printCurrentLevel(root: Node, level):
    if root is None:
        return
    if level == 1:
        print(
            "node id: " + root.id + "\t feature name: " + root.feature_name + "\t\tthreshold: " + root.threhold + "\t\tgain: "
            + root.gain)
    elif level > 1:
        printCurrentLevel(root.left_child, level - 1)
        printCurrentLevel(root.right_child, level - 1)


def height(node: Node):
    if node is None:
        return 0
    else:
        lheight = height(node.left_child)
        rheight = height(node.right_child)

        if lheight > rheight:
            return lheight + 1
        else:
            return rheight + 1


# ----------- build tree
def searchTree(root: Node, nodeId) -> Node:
    tempNode = Node
    tempNode2 = Node
    if root is None or root.id == nodeId:
        return root
    tempNode = searchTree(root.left_child, nodeId)
    # print()
    tempNode2 = searchTree(root.right_child, nodeId)
    # print()
    if tempNode is not None:
        return tempNode
    elif tempNode2 is not None:
        return tempNode2


def createNewNode():
    newNode = Node()
    return newNode


def insertNode(forest: list, treeID, row):
    # print("")
    temp = len(forest)
    # check if forest is empty
    if len(forest) == int(treeID):  # if len of forest is equal to treeID, that tree doesn't exist; create new tree
        newRoot = createNewNode()
        newRoot.id = row[3]
        newRoot.feature_name = row[4]
        newRoot.threhold = row[5]
        newRoot.gain = row[9]

        # initialize children as well
        leftNode = Node()
        leftNode.id = row[6]
        rightNode = Node()
        rightNode.id = row[7]

        newRoot.left_child = leftNode
        newRoot.right_child = rightNode

        # add new root to forest
        forest.append(newRoot)
    elif forest and row[6] and row[7]:  # forest is not empty and node is not a leaf node (has child/children)
        newNode = Node()
        newNode = searchTree(forest[int(treeID)], row[3])
        newNode.feature_name = row[4]
        newNode.threhold = row[5]
        newNode.gain = row[9]

        # initialize children as well
        leftNode = Node()
        leftNode.id = row[6]
        rightNode = Node()
        rightNode.id = row[7]

        newNode.left_child = leftNode
        newNode.right_child = rightNode

    elif forest and not row[6] and not row[7]:  # accounts for leaf nodes
        newNode = Node()
        newNode = searchTree(forest[int(treeID)], row[3])
        newNode.feature_name = row[4]
        newNode.threhold = row[5]
        newNode.gain = row[8]


def forestConversion():
    forest = []
    rootNode = None
    # row 1 is the header for the columns
    # the rows start with row 1, not row 0
    # the columns start with column 0
    with open('test_utils/models/treeModel (copy).csv') as csvObject:
        data = csv.reader(csvObject)
        print("hello world")
        count = 0
        treeID = None
        for row in data:

            # skip the header row
            if count == 0:
                count = count + 1
                continue
            # append node to tree
            insertNode(forest, row[1], row)
            count = count + 1

            # check the end of each tree
            # treeID=int(row[1])
            # if (count == ((512*(treeID+1))-treeID)):
            #     print()

        for x in forest:
            printLevelOrder(x)
        return forest


# --------------------------------------------- conversion ends here  ------------------------------------------
def test():
    def tree_traverse_predict(forest: list[Node], feature: pd.DataFrame) -> int:
        pass

    forest, sklearn_model = parse_from_pickle()

    features = pd.read_csv(
        "test_utils/test_data/test_samples.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )

    for feature in features:
        prediction1 = tree_traverse_predict(forest, feature)
        prediction2 = int(sklearn_model.predict())
        assert (
                prediction1 == prediction2
        ), f"Prediction from our implementation is {prediction1} and prediction from sklearn model is {prediction2}"


if __name__ == "__main__":
    # test()
    forestConversion()