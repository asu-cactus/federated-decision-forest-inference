from load_tree import parse_from_pickle

class Client:
    def __init__(self):
        self.forest, _ = parse_from_pickle()

    def local_compute(self, feature_partition):
        pass
