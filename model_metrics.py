from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from data_clean import *
import pandas as pd
import numpy as np

def run_classifiers(X_train, y_all,
                    list_of_classifiers = [LogisticRegression(), RandomForestClassifier(),
                    GradientBoostingClassifier()],
                    name_of_classifiers = ['LogReg', 'RandomForest', 'GradBoost'],
                    list_of_metrics = [mean_squared_error, accuracy_score],
                    name_of_metrics = ['mse', 'acc'],
                    n_folds=10):
    y_DX = y_all['DX']
    y_DXSUB = y_all['DXSUB']

    list_of_data = [(X_train, y_DX), (X_train, y_DXSUB)]
    name_of_data = ['DX', 'DXSUB']

    metrics_df = pd.DataFrame(data=None,
                              columns=['DX_acc_train', 'DX_acc_test', 'DX_mse_train', 'DX_mse_test',
                                       'DXSUB_acc_train', 'DXSUB_acc_test', 'DXSUB_mse_train', 'DXSUB_mse_test'],
                              index=['LogReg', 'RandomForest', 'GradBoost'])

    for (X, y), data_name in zip(list_of_data, name_of_data):
        for metric, metric_name in zip(list_of_metrics, name_of_metrics):
            for clf, clf_name in zip(list_of_classifiers, name_of_classifiers):
                train_metric, test_metric = cv(X.values, y.values,
                                               clf, n_folds=10,
                                               metric=metric)

                col_train_name = data_name + '_' + metric_name + '_train'
                col_test_name = data_name + '_' + metric_name + '_test'

                metrics_df[col_train_name].loc[clf_name] = np.mean(train_metric)
                metrics_df[col_test_name].loc[clf_name] = np.mean(test_metric)

    dx_df = metrics_df[['DX_acc_train', 'DX_acc_test', 'DX_mse_train', 'DX_mse_test']]
    dxsub_df = metrics_df[['DXSUB_acc_train', 'DXSUB_acc_test', 'DXSUB_mse_train', 'DXSUB_mse_test']]
    return dx_df, dxsub_df
