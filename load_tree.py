from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from typing import TypeAlias

class Node:
    def __init__(self):
        self.id = None
        self.feature_name = None    #feature number
        self.threhold = None        #split
        self.left_child = None      #yes
        self.right_child = None     #no
        self.site_name = None
        self.gain = None
        self.bitVector = None
        self.is_leaf = False

Forest: TypeAlias = list[Node]

def parse_from_pickle(
    model_name: Optional[str] = "higgs_xgboost_10_8",
) -> tuple[Forest, RandomForestClassifier]:
    # Get Sklearn model
    model_dir = 'test_utils/models'
    pkl_path = f'{model_dir}/{model_name}.pkl'
    sklearn_model = joblib.load(pkl_path)

    # Get our own representation
    csv_path = f'{model_dir}/{model_name}.csv'
    if Path(csv_path).is_file():
        forest = forestConversion(pd.read_csv(csv_path))
    else:
        model_df = sklearn_model.get_booster().trees_to_dataframe()
        model_df.to_csv(csv_path)
        forest = forestConversion(model_df)
    
    return (forest, sklearn_model)


def load_from_protobuf():
    pass
# --------------------------------------- converting to forest from csv-----------------------------------------------
# ------- print tree -------------
# simply used for verifying that all cells were used
# print methods accreddited to this source https://www.geeksforgeeks.org/level-order-tree-traversal/
def printLevelOrder(root:Node):
    h= height(root)
    for i in range(1,h+1):
        printCurrentLevel(root,i)
def printCurrentLevel(root:Node, level):
    if root is None:
        return
    if level==1:
        print("node id: "+root.id+ "\t feature name: " + root.feature_name + "\t\tthreshold: " + root.threhold +"\t\tgain: "
              +root.gain)
    elif level>1:
        printCurrentLevel(root.left_child, level-1)
        printCurrentLevel(root.right_child, level-1)
def height(node:Node):
    if node is None:
        return 0
    else:
        lheight=height(node.left_child)
        rheight=height(node.right_child)

        if lheight>rheight:
            return lheight+1
        else:
            return rheight+1

# ----------- build tree
def searchTree(root:Node, nodeId)->Node:
    tempNode=Node
    tempNode2=Node
    if root is None or root.id==nodeId:
        return root
    tempNode=searchTree(root.left_child, nodeId)
    # print()
    tempNode2=searchTree(root.right_child, nodeId)
    # print()
    if tempNode is not None:
        return tempNode
    elif tempNode2 is not None:
        return tempNode2

def createNewNode():
    newNode=Node()
    return newNode

def insertNode(forest: list, treeID, row):
    # print("")
    if len(forest)==treeID:  # if len of forest is equal to treeID, that tree doesn't exist; create new tree
        newRoot=createNewNode()
        newRoot.id = row[2]
        newRoot.feature_name = int(row[3]) - 1
        newRoot.threhold = float(row[4])
        newRoot.gain=float(row[8])
        assert newRoot.threhold is not None

        # initialize children as well
        leftNode=Node()
        leftNode.id=row[5]
        rightNode=Node()
        rightNode.id=row[6]

        newRoot.left_child=leftNode
        newRoot.right_child=rightNode

        # add new root to forest
        forest.append(newRoot)
    elif forest and row[3] != 'Leaf':   # forest is not empty and node is not a leaf node (has child/children)
        newNode=searchTree(forest[treeID], row[2])
        newNode.feature_name = int(row[3]) - 1
        newNode.threhold = float(row[4])
        newNode.gain= float(row[8])
        assert newNode.threhold is not None

        # initialize children as well
        leftNode = Node()
        leftNode.id = row[5]
        rightNode = Node()
        rightNode.id = row[6]

        newNode.left_child = leftNode
        newNode.right_child = rightNode

    elif forest and row[3] == 'Leaf': # accounts for leaf nodes
        newNode = searchTree(forest[treeID], row[2])
        newNode.gain = float(row[8])
        newNode.is_leaf = True
    else:
        print("Something else")
        raise


def forestConversion(model):
    forest = []
    # row 1 is the header for the columns
    # the rows start with row 1, not row 0
    # the columns start with column 0
    # with open('test_utils/models/treeModel (copy).csv') as csvObject:
    #     data = csv.reader(csvObject)
        # print("hello world")

    count = 0
    for _, row in model.iterrows():
        # # skip the header row
        # if count==0:
        #     count=count+1
        #     continue
        # append node to tree
        treeID = row[0]
        insertNode(forest, treeID ,row)
        count = count + 1

        # check the end of each tree

        # if (count == ((512*(treeID+1))-treeID)):
        #     print()

        # for x in forest:
        #     printLevelOrder(x)
    return forest

# --------------------------------------------- conversion ends here  ------------------------------------------
def tree_traverse_predict(forest: list[Node], feature: np.array) -> int:
    prediction = 0.0
    for tree_node in forest:
        # Each tree makes a prediction
        while not tree_node.is_leaf:
            feature_value = feature[tree_node.feature_name]
            if feature_value < tree_node.threhold:
                tree_node = tree_node.left_child
            else:
                tree_node = tree_node.right_child
        
        prediction += tree_node.gain

    # Aggregate predictions
    return 1 if prediction > 0 else 0
    # return prediction

def test():
    forest, sklearn_model = parse_from_pickle()

    features = pd.read_csv(
        "test_utils/test_data/test_samples.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )

    for i, feature in features.iterrows():
        feature = feature.to_numpy()
        prediction1 = tree_traverse_predict(forest, feature)
        prediction2 = int(sklearn_model.predict(np.expand_dims(feature,0)))
        print(f"Test{i}: Predition1 {prediction1}, Prediction2 {prediction2}")
        assert prediction1 == prediction2

if __name__ == "__main__":
    test()
    # forest = forestConversion()
