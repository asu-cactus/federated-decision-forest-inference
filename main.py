from client import Client
import pandas as pd
import numpy as np
import time
from common import quickscorer_without_sorting, get_forest_model

feature_offsets = [0, 10, 19]

def partition_dataset():
    dataset = pd.read_csv(
        "test_utils/test_data/HIGGS_test.csv",
        dtype=np.float32,
        usecols=range(1, 29),
        header=None,
    )
    offset0, offset1, offset2 = feature_offsets
    feature_division = [(offset0, offset1), (offset1, offset2), (offset2, 28)]
    partitions = []
    for i, j in feature_division:
        partitions.append(dataset.iloc[:, i:j])
    return partitions



def join_and_inference():
    forest = get_forest_model()
    partitions = partition_dataset()

    for i, partition in enumerate(partitions):
        partition['key'] = range(len(partition))
        partitions[i] = partition.set_index('key')

    # Join
    start = time.time()
    df = partitions[0].join(partitions[1]).join(partitions[2])
    end = time.time()
    join_time = end - start

    df = df.reset_index()

    # Inference
    start = time.time()
    for i, feature in df.iterrows():
        quickscorer_without_sorting(forest, feature.tolist())
    end = time.time()
    infer_time = end - start

    print('Inference after join using Pandas:')
    print(f'Join time: {join_time}s, Inference time: {infer_time}s, total time: {join_time + infer_time}s')
    return join_time + infer_time


def federated_inference():
    partitions = partition_dataset()
    bitvector_trees = get_forest_model()
    clients = [Client(bitvector_trees) for _ in range(len(partitions))]
    start = time.time()
    
    local_results = [client.local_compute(partition.values, offset) for client, partition, offset in zip(clients, partitions, feature_offsets)]

    for l1, l2, l3 in zip(*local_results): # For each feature
        prediction = 0.0
        
        for tree, vec1, vec2, vec3 in zip(bitvector_trees, l1, l2, l3): # For each tree
            and_vector = vec1 & vec2 & vec3
            for i, bit in enumerate(and_vector):
                if bit == 1:
                    prediction += tree.output_values[i]
                break
            
        prediction = 1 if prediction > 0 else 0
            
    total_time = time.time() - start
    print('Federated inference:')
    print(f'Total time: {total_time}s')
    return total_time
    


if __name__ == "__main__":
    join_and_inference()
    federated_inference()
