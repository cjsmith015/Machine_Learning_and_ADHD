from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from data_clean import *
import pandas as pd
import numpy as np

def run_classifiers(data, list_of_classifiers = [LogisticRegression(), RandomForestClassifier(),
                    GradientBoostingClassifier()],
                    list_of_metrics = [mean_squared_error, accuracy_score],
                    n_folds=10):
    X_train_DX = data.drop(columns=['DX','DXSUB'])
    X_train_DXSUB = data.drop(columns=['DX','DXSUB'])

    y_train_DX = data['DX']
    y_train_DXSUB = data['DXSUB']

    list_of_data = [(X_train_DX, y_train_DX), (X_train_DXSUB, y_train_DXSUB)]

    classifier_metric_results = []
    for classifier in list_of_classifiers:
        predictor_result = []
        for X, y in list_of_data:
            metric_results = []
            for metric in list_of_metrics:
                train_metric, test_metric = cv(X.values, y.values,
                                                    classifier, n_folds=10,
                                                    metric=metric)
                metric_results.append((train_metric, test_metric))
            predictor_result.append(metric_results)
        classifier_metric_results.append(predictor_result)

    return classifier_metric_results
