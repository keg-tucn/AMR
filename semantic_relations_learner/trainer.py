import math

from sklearn.neural_network import MLPClassifier

import DatasetExtractor


def train(train_split_percentage=0.8):
    data = DatasetExtractor.get_concepts_relations_pairs()

    train_size = math.floor(len(data) * train_split_percentage)

    train = data[0:train_size]
    test = data[train_size:]


X = [[0., 0.], [1., 1.]]
y = [0, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)
