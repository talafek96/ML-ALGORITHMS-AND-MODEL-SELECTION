from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial import distance
# from statistics import mode
from collections import Counter
import numpy as np


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, k: Optional[int]=None):
        self.train_x = X
        self.train_y = y
        if k is not None:
            self.n_neighbors = k
        return self

    def predict(self, target_x):
        # Note: You can use self.n_neighbors here
        dist_mat = distance.cdist(target_x, self.train_x, 'euclidean')
        ind = np.argpartition(dist_mat, self.n_neighbors, axis=1)
        labeling_mat = np.array(
            [self.train_y[ind[i][:self.n_neighbors]] for i in range(len(ind))])
        predictions = np.array(
            [type(self).__most_common(row) for row in labeling_mat])
        return predictions

    @staticmethod
    def __most_common(arr):
        return Counter(arr).most_common(1)[0][0]
