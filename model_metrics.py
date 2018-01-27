import pandas as pd
import numpy as np
import pickle, sys

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from fancyimpute import MatrixFactorization

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

if __name__ == '__main__':
    pickle_name = sys.argv[1]

    train_data = pd.read_csv('data/train_data.csv')
    train_data_small = train_data.sample(n=200)
    X = train_data_small.drop(columns=['DX','DXSUB'])
    y = train_data_small['DX'].map({3:1, 1:0})

    # make pipeline
    log_reg_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                        LogisticRegression(random_state=56))

    rf_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                           RandomForestClassifier(n_jobs=-1, random_state=56))

    gb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                           GradientBoostingClassifier(random_state=56))

    xgb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                            XGBClassifier(max_depth=3, learning_rate=0.1,
                            random_state=56, n_jobs=-1))

    # create lists
    scoring_list = ['accuracy', 'roc_auc', 'neg_log_loss']

    classifier_list = [log_reg_clf, rf_clf, gb_clf, xgb_clf]
    classifier_name = ['LogReg', 'RandomForest', 'GradientBoosting', 'XGB']

    metrics_of_interest = ['fit_time', 'score_time', 'test_accuracy',
                       'test_neg_log_loss', 'test_roc_auc']

    metric_dx_df = model_metrics.get_metrics(X, y,
                                classifier_list, classifier_name,
                                scoring_list, metrics_of_interest,
                                n_folds=2)

    with open(pickle_name, 'wb') as f:
        pickle.dump(metric_dx_df, f)
