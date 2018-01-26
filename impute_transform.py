import numpy as np
import pandas as pd
from fancyimpute import MICE, MatrixFactorization

class ImputeTransform(object):
    def __init__(self, strategy=MICE()):
        self.solver = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if np.sum(np.isnan(X.values)) > 0:
            return self.solver.complete(X)
        else:
            return X

    def get_params(self, deep=True):
        params = {'strategy': self.solver}
        return params
