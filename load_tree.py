from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


class Node:
    def __init__(self):
        self.id = None
        self.feature_name = None
        self.threhold = None
        self.left_child = None
        self.right_child = None
        self.site_name = None


def parse_from_pickle(
    model_path: Optional[str] = "test_utils/models/higgs_randomforest_10_8.pkl",
) -> tuple[list[Node], RandomForestClassifier]:
    sklearn_model = joblib.load(model_path)
    forest = []
    # TODO: extract trees from sklearn model and store them to forest

    return (forest, sklearn_model)


def load_from_protobuf():
    pass


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
    test()
