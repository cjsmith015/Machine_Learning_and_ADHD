import numpy as np
import pandas as pd
from fancyimpute import MICE, MatrixFactorization

class Model(object):
    def __init__(self, classifier, impute=True, impute_mode=MICE()):
        """
        INPUT:
        - classifier = Model classifier object
        - impute = Bool, runs imputation
        """
        self.clf = classifier
        self.solver = impute_mode
        self.impute = impute

    def fit(self, X, y):
        """
        INPUT:
        - X: dataframe representing feature matrix for training data
        - y: series representing labels for training data
        """
        if self.impute = True:
            if np.sum(np.isnan(X)) > 0:
                X_fit = _impute_data(X)
        else:
            X_fit = X

        self.clf.fit(X, y)

    def _impute_data(X):
        solver =
            X_impute =
