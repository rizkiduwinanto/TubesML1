from math import inf
from scipy.spatial.distance import cdist
import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iters=inf, metric='euclidean'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.metric = metric
        self.iters = 0
        self.objects = None
        self.centroids = None
        self.labels = None
        self.is_convergent = False

    def _calculate_centroids(self):
        sums = np.zeros((self.n_clusters, self.objects.shape[1]))
        counts = np.zeros((self.n_clusters, 1))
        for i, obj in enumerate(self.objects):
            sums[self.labels[i]] += obj
            counts[self.labels[i]] += 1
        return sums / counts

    def fit(self, objects, init_centroids=None):
        self.objects = objects
        self.labels = np.full(len(self.objects), -1)
        if init_centroids is None:
            self.centroids = self.objects[np.random.choice(len(self.objects), self.n_clusters, False), :]
        else:
            self.centroids = init_centroids

        while self.iters < self.max_iters and not self.is_convergent:
            prev_labels = self.labels
            self.labels = cdist(self.centroids, self.objects, self.metric).argmin(0)
            self.is_convergent = np.array_equal(prev_labels, self.labels)
            self.centroids = self._calculate_centroids()
            self.iters += 1

    def fit_predict(self, objects, init_centroids=None):
        self.fit(objects, init_centroids)
        return self.labels
