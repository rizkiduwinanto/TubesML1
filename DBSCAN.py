import pandas as pd
import numpy as np

class DBSCAN:
    def __init__(self, min_points, epsilon):
        self.min_points = min_points
        self.epsilon = epsilon

    def fit_predict(self, data):
        return data

model = DBSCAN(2, 2)