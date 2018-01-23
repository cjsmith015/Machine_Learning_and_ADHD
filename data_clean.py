from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import clone
from fancyimpute import MatrixFactorization
import pandas as pd
import numpy as np

def impute_data(X):
    """Impute the data using Matrix Factorization

    Parameters
    ----------
    X: np.array
       Matrix of predictors

    Returns
    -------
    impute_data_filled: np.array
       X, with missing values filled
    """
    #impute_data = X
    #data_index = X.index
    #data_cols = df.columns

    solver = MatrixFactorization(verbose=False)
    impute_data = solver.complete(X)
    #impute_df = pd.DataFrame(impute_data_filled, index=data_index, columns=data_cols)
    return impute_data

def cv(X, y, base_estimator, n_folds, metric, random_state=56):
    """Estimate the in and out-of-sample error of a model using cross validation.
    Code by Matthew Drury (madrury)

    Parameters
    ----------
    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.

    n_folds: int
      The number of folds in the cross validation.

    random_seed: int
      A seed for the random number generator, for repeatability.

    Returns
    -------

    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, random_state=random_state)
    train_cv_errors, test_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]

        # Impute the data if missing numbers
        if np.sum(np.isnan(X_cv_train)) > 0:
            X_cv_train_final = impute_data(X_cv_train)
        else:
            X_cv_train_final = X_cv_train.copy()

        if np.sum(np.isnan(X_cv_test)) > 0:
            X_cv_test_final = impute_data(X_cv_test)
        else:
            X_cv_test_final = X_cv_test.copy()

        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_final, y_cv_train)

        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_final)
        y_hat_test = estimator.predict(X_cv_test_final)

        # Calculate the error metrics
        train_cv_errors[idx] = metric(y_cv_train, y_hat_train)
        test_cv_errors[idx] = metric(y_cv_test, y_hat_test)

    return train_cv_errors, test_cv_errors

def print_metric(train_metric, test_metric):
    print("Training CV metric: {:2.2f}".format(train_metric.mean()))
    print("Test CV metric: {:2.2f}".format(test_metric.mean()))
