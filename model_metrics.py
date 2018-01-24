from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
from data_clean import *
import pandas as pd
import numpy as np

def run_classifiers(X_train, y_all,
                    list_of_classifiers = [LogisticRegression(), RandomForestClassifier(),
                    GradientBoostingClassifier()],
                    name_of_classifiers = ['LogReg', 'RandomForest', 'GradBoost'],
                    list_of_metrics = [accuracy_score, log_loss],
                    name_of_metrics = ['acc', 'logloss'],
                    n_folds=10,
                    class_label_dict = {3:1, 1:0}):
    y_DX = y_all['DX']
    y_DXSUB = y_all['DXSUB']

    list_of_data = [(X_train, y_DX), (X_train, y_DXSUB)]
    name_of_data = ['DX', 'DXSUB']

    column_list = []
    for data in name_of_data:
        for metric in name_of_metrics:
            column_list.append(data + '_' + metric + '_train')
            column_list.append(data + '_' + metric + '_test')

    metrics_df = pd.DataFrame(data=None,
                              columns=column_list,
                              index=name_of_classifiers)

    for (X, y), data_name in zip(list_of_data, name_of_data):
        for clf, clf_name in zip(list_of_classifiers, name_of_classifiers):
            train_cv_metric, test_cv_metric = cv(X.values, y.values,
                                           clf, n_folds=n_folds,
                                           metrics=list_of_metrics)
            train_metric = np.mean(train_cv_metric, axis=0)
            test_metric = np.mean(test_cv_metric, axis=0)
            for idx, metric in enumerate(name_of_metrics):
                col_train = data_name + '_' + metric + '_train'
                col_test = data_name + '_' + metric + '_test'
                metrics_df[col_train].loc[clf_name] = train_metric[idx]
                metrics_df[col_test].loc[clf_name] = test_metric[idx]

    return metrics_df
