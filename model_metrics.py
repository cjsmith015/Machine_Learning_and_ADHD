import pandas as pd
import numpy as np
import pickle, sys

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer

from xgboost import XGBClassifier

from fancyimpute import MatrixFactorization, SimpleFill

from impute_transform import ImputeTransform

def get_metrics(X, y, clf_dict,
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
    clf_metrics = _run_clfs(clf_dict,
                            X, y, scoring,
                            n_folds, return_train_score, multiclass)
    #
    # clf_metrics_mean = _mean_metrics(clf_metrics)
    #
    # metric_df = _create_df(clf_names, metric_df_cols)
    #
    # return _fill_df(metric_df, clf_metrics_mean, clf_names, metric_df_cols)

    return clf_metrics

def _run_clfs(clf_dict,
                X, y, scoring,
                n_folds, return_train_score, multiclass):
    """Runs cross validation on classifiers"""
    for name in clf_dict.keys():
        clf = clf_dict[name]['clf']
        scores = cross_validate(clf, X, y,
                                scoring=scoring, cv=n_folds,
                                return_train_score=True)
        clf_dict[name]['metrics'] = scores
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

def multiclass_roc_auc_score(truth, pred, average=None):
    """Returns multiclass roc auc score"""
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    # with open('multiclass.txt', 'a') as f:
    #     data = list(roc_auc_score(truth, pred, average=None))
    #     data.append(truth.shape[0])
    #     f.write(str(data))
    #     f.write('\n')

    return roc_auc_score(truth, pred, average='macro')

def prep_x_y(df, target, feature):
    if target == 'DX':
        y = df[target].map({3:1, 1:0})
    else:
        y = df[target]

    if feature == 'tmcq':
        cols = ['Y1_P_TMCQ_ACTIVCONT', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL',
          'Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_HIP',
           'Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_SAD',
           'Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_ASSERT',
           'Y1_P_TMCQ_ATTFOCUS', 'Y1_P_TMCQ_LIP', 'Y1_P_TMCQ_PERCEPT',
           'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_SURGENCY',
           'Y1_P_TMCQ_EFFCONT', 'Y1_P_TMCQ_NEGAFFECT']
    elif feature == 'neuro':
        cols = ['STOP_SSRTAVE_Y1', 'DPRIME1_Y1', 'DPRIME2_Y1', 'SSBK_NUMCOMPLETE_Y1',
            'SSFD_NUMCOMPLETE_Y1', 'V_Y1', 'Y1_CLWRD_COND1', 'Y1_CLWRD_COND2',
            'Y1_DIGITS_BKWD_RS', 'Y1_DIGITS_FRWD_RS', 'Y1_TRAILS_COND2',
            'Y1_TRAILS_COND3', 'CW_RES', 'TR_RES', 'Y1_TAP_SD_TOT_CLOCK']
    elif feature == 'all':
        cols = ['Y1_P_TMCQ_ACTIVCONT', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL',
          'Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_HIP',
           'Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_SAD',
           'Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_ASSERT',
           'Y1_P_TMCQ_ATTFOCUS', 'Y1_P_TMCQ_LIP', 'Y1_P_TMCQ_PERCEPT',
           'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_SURGENCY',
           'Y1_P_TMCQ_EFFCONT', 'Y1_P_TMCQ_NEGAFFECT',
           'STOP_SSRTAVE_Y1', 'DPRIME1_Y1', 'DPRIME2_Y1', 'SSBK_NUMCOMPLETE_Y1',
           'SSFD_NUMCOMPLETE_Y1', 'V_Y1', 'Y1_CLWRD_COND1', 'Y1_CLWRD_COND2',
           'Y1_DIGITS_BKWD_RS', 'Y1_DIGITS_FRWD_RS', 'Y1_TRAILS_COND2',
           'Y1_TRAILS_COND3', 'CW_RES', 'TR_RES', 'Y1_TAP_SD_TOT_CLOCK']
    X = df[cols]

    X_no_null = X[X.isnull().sum(axis=1) != X.shape[1]]
    y_no_null = y[X.isnull().sum(axis=1) != X.shape[1]]

    return X_no_null, y_no_null

def prep_clfs():
    log_reg_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                        LogisticRegression(random_state=56))

    rf_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                           RandomForestClassifier(n_jobs=-1, random_state=56))

    gb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                           GradientBoostingClassifier(random_state=56))

    xgb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                            XGBClassifier(max_depth=3, learning_rate=0.1,
                            random_state=56, n_jobs=-1))
    classifier_dict = {'LogReg':
                            {'clf': log_reg_clf},
                       'RandomForest':
                            {'clf': rf_clf},
                       'GradientBoosting':
                            {'clf': gb_clf},
                       'XGB':
                            {'clf': xgb_clf}}
    return classifier_dict

def prep_scoring(target):
    scoring_dict = {'accuracy': 'accuracy',
                    'neg_log_loss': 'neg_log_loss'}
    if target == 'DXSUB':
        multiclass_roc = make_scorer(multiclass_roc_auc_score,
                                 greater_is_better=True)
        scoring_dict['roc_auc'] = multiclass_roc
    else:
        scoring_dict['roc_auc'] = 'roc_auc'
    return scoring_dict

if __name__ == '__main__':
    # run with "python model_metrics.py filename target feature pickle/csv"
    # target can be DX/DXSUB
    # feature can be all, neuro, or tmcq
    filename = sys.argv[1]
    target = sys.argv[2]
    feature = sys.argv[3]
    output = sys.argv[4]

    train_data = pd.read_csv('data/train_data.csv')
    small_data = train_data.sample(n=100)

    # Prep stuff
    #X, y = prep_x_y(train_data, dataset)
    X, y = prep_x_y(small_data, target, feature)
    classifier_dict = prep_clfs()
    scoring = prep_scoring(target)

    # Standard across datasets
    metrics_of_interest = ['fit_time', 'score_time', 'test_accuracy',
                           'test_neg_log_loss', 'test_roc_auc',
                           'train_accuracy', 'train_neg_log_loss',
                           'train_roc_auc']
    # Get metrics
    metric_dict = get_metrics(X, y,
                            classifier_dict,
                            scoring, metrics_of_interest,
                            n_folds=2)

    # save dict as pickle
    with open(filename, 'wb') as f:
        pickle.dump(metric_dict, f)

    # metric_df.to_csv(csv_name)
