import numpy as np
from typing import List
from classifier import Classifier


class DecisionStump(Classifier):
    def __init__(self, s: int, b: float, d: int):
        self.clf_name = 'Decision_stump'
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass

    def predict(self, features: List[List[float]]) -> List[int]:
        X = np.array(features)
        N, D = X.shape
        result = np.array([(-1)*self.s] * N)
        result[X[:, int(self.d)] > self.b] = self.s
        return np.array(result).tolist()
