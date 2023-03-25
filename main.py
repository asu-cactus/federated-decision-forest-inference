from client import Client
from load_tree import parse_from_pickle
import pandas as pd
import numpy as np
import time

def partition_dataset():
    dataset = pd.read_csv(
        "test_utils/test_data/HIGG_test.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )
    feature_division = [(0, 10), (10, 19), (19, 28)]
    partitions = []
    for i, j in feature_division:
        partitions.append(dataset.iloc[i:j, :])
    return partitions

def join_and_inference():
    forest, _ = parse_from_pickle() 
    partitions = partition_dataset()

    # TODO: Implement join and inference. 
    # You can add an id column to each partition and join by id.
    # Please use pandas join instead of simply concatenate the partitions.


    # Join
    start = time.time()
    ## join code
    end = time.time()
    join_time = end - start

    # Inference
    start = time.time()
    ## inference code (can import from common.py)
    end = time.time()
    infer_time = end - start

    print('Inference after join using Pandas:')
    print(f'Join time: {join_time}s, Inference time: {infer_time}s, total time: {join_time + infer_time}s')



def federated_inference():
    partitions = partition_dataset()
    clients = [Client() for _ in range(len(partitions))]

    start = time.time()
    for all_features in zip(*partitions):
        # TODO: implement local_compute for Client class
        local_results = [client.local_compute(feature_partition) for client, feature_partition in zip(clients, all_features)]
        # TODO: Aggregate local_results

    total_time = time.time() - start
    print('Federated inference:')
    print(f'Total time: {total_time}s')
    


if __name__ == "__main__":
    join_and_inference()
    federated_inference()
