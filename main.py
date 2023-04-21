from host import Host
from client import Client
from load_tree import parse_from_pickle
import pandas as pd
import numpy as np

def test():
    def federated_inference(forest, feature, feature_division):
        client1 = Client()
        client2 = Client()
        host = Host()

        host.local_compute()
        bitvectorss_from_client1 = client1.local_compute()
        bitvectorss_from_client2 = client2.local_compute()
        return host.aggregate([bitvectorss_from_client1, bitvectorss_from_client2])

    feature_division = [tuple(range(0, 16)), tuple(range(16, 24)), tuple(range(24, 29))]
    forest, sklearn_model = parse_from_pickle() 
    features = pd.read_csv(
        "test_utils/test_data/test_samples.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )

    for feature in features:
        prediction1 = federated_inference(forest, feature, feature_division)
        prediction2 = int(sklearn_model.predict())
        assert (
            prediction1 == prediction2
        ), f"Prediction from our implementation is {prediction1} and prediction from sklearn model is {prediction2}"


if __name__ == "__main__":
    test()
