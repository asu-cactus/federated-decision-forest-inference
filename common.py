from load_tree import Node

import pandas as pd
import numpy as np
import joblib


def generate_bitvector(tree: Node) -> list[list[int]]:
    pass


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
    test()
