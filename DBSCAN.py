# coding: utf-8
import pandas as pd
import numpy as np
import math

UNVISITED = False
OUTLIER = None

class DBSCAN:
    def __init__(self, min_points, epsilon):
        self.min_points = min_points
        self.epsilon = epsilon

    def fit_predict(self, data):
        idx_cluster = 1
        self.data = data
        number_of_points = self.data.shape[1]
        self.classifications = [UNVISITED] * number_of_points
        for idx_point in range(0, number_of_points):
            if self.classifications[idx_point] == UNVISITED:
                if self.expand_cluster(idx_point, idx_cluster):
                    idx_cluster += 1
        return self.classifications
    
    def expand_cluster(self, point, cluster):
        seeds = self.region_query(point)
        if len(seeds) < self.min_points:
            self.classifications[point] = OUTLIER
            return False
        else:
            self.classifications[point] = cluster
            for idx_seed in seeds:
                self.classifications[idx_seed] = cluster
                
            while len(seeds) > 0:
                current_point = seeds[0]
                results = self.region_query(current_point)
                if len(results) >= self.min_points:
                    for idx in range(0, len(results)):
                        result_point = results[idx]
                        if self.classifications[result_point] == UNVISITED or self.classifications[result_point] == OUTLIER:
                            if self.classifications[result_point] == UNVISITED:
                                seeds.append(result_point)
                            self.classifications[result_point] = cluster
                seeds = seeds[1:]
            
            return True
    
    def region_query(self, point):
        number_of_points = self.data.shape[1]
        seeds = []
        for idx in range(0, number_of_points):
            if self.neighborhood(self.data[:,point], self.data[:,idx]):
                seeds.append(idx)
        return seeds
        
    def neighborhood(self, point1, point2):
        return self.distance(point1, point2) < self.epsilon
        
    def distance(self, point1, point2):
        return math.sqrt(np.power(point1 - point2, 2).sum())

