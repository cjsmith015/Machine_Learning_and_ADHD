import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def get_metrics(X, y, clf_list, clf_names,
                    scoring, metric_df_cols,
                    n_folds, return_train_score=True,
                    multiclass=False):
    """Runs cross validation to obtain error metrics for several classifiers.
    Outputs a formatted dataframe.

    INPUTS
    ------
    - X: dataframe representing feature matrix for training data
    - y: series representing target for training data
    - clf_list: list of objects of prepared classifiers
    - clf_names: strings corresponding to clf_list
    - scoring: list of strings for sklearn scoring
    - metric_df_cols: list of strings for desired columns of output DataFrame
    - n_folds: int, number of folds for k-fold cross validation
    - return_train_score: bool, returns training score on cv
    - multiclass: bool, whether target has multiple classes

    OUTPUTS
    -------
    """
    clf_metrics = _run_clfs(clf_list, clf_names,
                            X, y, scoring,
                            n_folds, return_train_score, multiclass)

    clf_metrics_mean = _mean_metrics(clf_metrics)

    metric_df = _create_df(clf_names, metric_df_cols)

    return _fill_df(metric_df, clf_metrics_mean, clf_names, metric_df_cols)

def multiclass_roc_auc_score(truth, pred, average=None):
    """Returns multiclass roc auc score"""
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return list(roc_auc_score(truth, pred, average=average))

def _run_clfs(clf_list, clf_names,
                X, y, scoring,
                n_folds, return_train_score, multiclass):
    """Runs cross validation on classifiers"""
    clf_dict = {}
    for clf, name in zip(clf_list, clf_names):
        scores = cross_validate(clf, X, y,
                                scoring=scoring, cv=n_folds,
                                return_train_score=True)
        clf_dict[name] = scores
    return clf_dict

def _mean_metrics(clf_dict):
    """Averages metrics across folds of cross validation"""
    for clf, dictionary in clf_dict.items():
        for metric, score in clf_dict[clf].items():
            clf_dict[clf][metric] = np.mean(score)
    return clf_dict

def _create_df(clf_names, cols):
    """Returns empty dataframe with index as classifier names and columns as
    metrics of interest"""
    return pd.DataFrame(data=None, index=clf_names, columns=cols)

def _fill_df(df, clf_dict, clf_names, cols):
    """Fills dataframe with metric scores for each classifier"""
    for clf in clf_names:
        for col in cols:
            df[col].loc[clf] = clf_dict[clf][col]
    return df

#
# def run_classifiers(X_train, y_all,
#                     list_of_classifiers = [LogisticRegression(), RandomForestClassifier(),
#                     GradientBoostingClassifier()],
#                     name_of_classifiers = ['LogReg', 'RandomForest', 'GradBoost'],
#                     list_of_metrics = [accuracy_score, log_loss],
#                     name_of_metrics = ['acc', 'logloss'],
#                     n_folds=10):
#     y_DX = y_all['DX']
#     y_DXSUB = y_all['DXSUB']
#
#     list_of_data = [(X_train, y_DX), (X_train, y_DXSUB)]
#     name_of_data = ['DX', 'DXSUB']
#
#     column_list = []
#     for data in name_of_data:
#         for metric in name_of_metrics:
#             column_list.append(data + '_' + metric + '_train')
#             column_list.append(data + '_' + metric + '_test')
#
#     metrics_df = pd.DataFrame(data=None,
#                               columns=column_list,
#                               index=name_of_classifiers)
#
#     for (X, y), data_name in zip(list_of_data, name_of_data):
#         for clf, clf_name in zip(list_of_classifiers, name_of_classifiers):
#             train_cv_metric, test_cv_metric = cv(X.values, y.values,
#                                            clf, n_folds=n_folds,
#                                            metrics=list_of_metrics)
#             train_metric = np.mean(train_cv_metric, axis=0)
#             test_metric = np.mean(test_cv_metric, axis=0)
#             for idx, metric in enumerate(name_of_metrics):
#                 col_train = data_name + '_' + metric + '_train'
#                 col_test = data_name + '_' + metric + '_test'
#                 metrics_df[col_train].loc[clf_name] = train_metric[idx]
#                 metrics_df[col_test].loc[clf_name] = test_metric[idx]
#
#     return metrics_df
