from scipy.spatial import distance
import numpy as np
import math


class DistanceMatrix:
    def __init__(self, objects, metric):
        self.matrix = distance.pdist(objects, metric)
        self.n = len(objects)

    def _s2c(self, i, j):
        assert i != j, 'no diagonal elements in condensed matrix'
        if i < j:
            i, j = j, i
        return int(self.n * j - j * (j + 1) / 2 + i - 1 - j)

    def _elem_in_i_rows(self, i):
        return i * (self.n - 1 - i) + (i * (i + 1)) / 2

    def _c2s(self, k):
        i = int(math.ceil((1 / 2.) * (- (-8 * k + 4 * self.n ** 2 - 4 * self.n - 7) ** 0.5 + 2 * self.n - 1) - 1))
        j = int(self.n - self._elem_in_i_rows(i + 1) + k)
        return i, j

    def __getitem__(self, key):
        return self.matrix[self._s2c(*key)]

    def __setitem__(self, key, value):
        self.matrix[self._s2c(*key)] = value

    def get_min_row_col(self):
        return self._c2s(np.unravel_index(np.argmin(self.matrix), self.matrix.shape)[0])

    def delete_line(self, line):
        self.matrix = np.delete(self.matrix, [self._s2c(line, k) for k in range(self.n) if k != line])
        self.n -= 1

    def __str__(self):
        return str(distance.squareform(self.matrix))


class Agglomerative:
    def __init__(self, n_clusters, linkage, metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.objects = None
        self.labels = None
        self.clusters = None
        self.distance_matrix = None

    def linkage_function(self, i, j, k):
        if self.linkage == 'single':
            return min(self.distance_matrix[(i, k)], self.distance_matrix[(j, k)])
        elif self.linkage == 'complete':
            return max(self.distance_matrix[(i, k)], self.distance_matrix[(j, k)])
        elif self.linkage == 'average':
            return np.mean(distance.cdist(
                self.objects[self.clusters[i] + self.clusters[j]],
                self.objects[self.clusters[k]], self.metric))
        elif self.linkage == 'averagegroup':
            return getattr(distance, self.metric)(
                np.mean(self.objects[self.clusters[i] + self.clusters[j]], 0),
                np.mean(self.objects[self.clusters[k]], 0))
        else:
            return None

    def fit(self, objects):
        self.objects = objects
        self.clusters = list([x] for x in range(len(self.objects)))
        self.distance_matrix = DistanceMatrix(self.objects, self.metric)

        while self.distance_matrix.n > self.n_clusters:
            i, j = self.distance_matrix.get_min_row_col()
            for k in range(self.distance_matrix.n):
                if k != j and k != i:
                    self.distance_matrix[(i, k)] = self.linkage_function(i, j, k)
            self.distance_matrix.delete_line(j)
            self.clusters[i].extend(self.clusters[j])
            del self.clusters[j]

        self.labels = np.full(len(self.objects), -1)
        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                self.labels[index] = i

    def fit_predict(self, objects):
        self.fit(objects)
        return self.labels
