from __future__ import annotations
import numpy as np


class KNN:
    """
    K Nearest Neighbors Classifier
    """
    dist_fn_map = {
        'euclidean': lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)),
        'manhattan': lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1),
        'hamming': lambda x1, x2: np.sum((x1 != x2).astype(int), axis=-1)
    }

    def __init__(self,
                 k: int = 1,
                 dist_fn: str = 'euclidean'
                 ) -> None:
        """
        Initialises KNN model with choice of k & distance function, to calculate distance between points
        :param k: Default k is 1
        :param dist_fn: Default is euclidean distance
        """
        self.dist_fn = self.dist_fn_map[dist_fn]
        self.k = k
        self.x = None
        self.y = None
        self.c = None
        return

    def fit(self, x: np.ndarray, y: np.ndarray) -> KNN:
        """
        Stores training set
        :param x: Training set for X
        :param y: Training set for Y
        :return: Self
        """
        self.x = x
        self.y = y
        self.c = np.max(y) + 1
        return self

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Makes prediction on testing set of X using initialised distance function
        :param x_test: Testing set of X
        :return: Tuple containing 1: Probabilities of each class, 2: Nearest neighbors, 3: Predicted class
        """
        n = x_test.shape[0]
        distances = self.dist_fn(self.x[None, :, :], x_test[:, None, :])

        neighbors = np.zeros((n, self.k), dtype=int)
        y_prob = np.zeros((n,  self.c))
        for i in range(n):
            neighbors[i, :] = np.argsort(distances[i])[:self.k]
            y_prob[i, :] = np.bincount(self.y[neighbors[i, :]], minlength=self.c)

        y_prob /= self.k
        y_pred = np.argmax(y_prob, axis=1)
        return y_prob, neighbors, y_pred
