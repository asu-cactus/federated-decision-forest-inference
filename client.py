from common import aggregate_vectors

class Client:
    def __init__(self, bitvector_trees):
        self.bitvector_trees = bitvector_trees

    def local_compute(self, features, feature_offset):
        """
        Return a list of lists, size [data_size, tree_size]
        """
        and_vectorss = []
        for _, feature in enumerate(features):
            and_vectors = []
            for tree in self.bitvector_trees:
                and_vectors.append(aggregate_vectors(tree, feature, feature_offset))
            and_vectorss.append(and_vectors)

        return and_vectorss
            
        



