from model_metrics import *
import pandas as pd

if __name__ == '__main__':
    cv = False
    # Set cross-validation, or train/test

    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/holdout_data.csv')
    #small_data = train_data.sample(n=200)

    dataset_dict = {'DX_All':
                        {'target': 'DX',
                         'feature': 'all'},
                    'DXSUB_All':
                        {'target': 'DXSUB',
                         'feature': 'all'},
                    'DX_TMCQ':
                        {'target': 'DX',
                         'feature': 'tmcq'},
                    'DX_Neuro':
                        {'target': 'DX',
                         'feature': 'neuro'}}

    # Standard across datasets
    metrics_of_interest = {'ROCAUC': 'test_roc_auc',
                           'LogLoss': 'test_neg_log_loss'}

    final_final_dict = {}

    for dataset in dataset_dict.keys():
        target = dataset_dict[dataset]['target']
        feature = dataset_dict[dataset]['feature']

        if cv == True:
            csv_name = 'final_csvs/' + dataset + '.csv'
        else:
            csv_name = 'final_csvs/' + dataset + '_holdout.csv'

        # Prep stuff
        #X, y = prep_x_y(train_data, dataset)
        X, y = prep_x_y(train_data, target, feature)
        if not cv:
            X_test, y_test = prep_x_y(test_data, target, feature)
        else:
            X_test = None
            y_test = None

        classifier_dict = prep_clfs(feature)
        scoring = prep_scoring(target)

        # Get metrics
        if cv == True:
            metric_dict = get_metrics(X, y,
                                    classifier_dict,
                                    scoring, metrics_of_interest,
                                    X_test, y_test,
                                    cv,
                                    n_folds=10)
        else:
            metric_dict = get_metrics(X, y,
                                    classifier_dict,
                                    scoring, metrics_of_interest,
                                    X_test, y_test,
                                    cv)

        # make final dict
        final_dict = {}
        for clf_name in metric_dict.keys():
            for metric in metrics_of_interest.keys():
                key_name = clf_name + '_' + metric
                metric_col = metrics_of_interest[metric]
                final_dict[key_name] = metric_dict[clf_name]['metrics'][metric_col]

        # make df
        if cv == True:
            final_df = pd.DataFrame.from_dict(final_dict, orient='columns')
            final_df.to_csv(csv_name)
        else:
            final_final_dict[dataset] = final_dict

    if cv == False:
        final_df = pd.DataFrame.from_dict(final_final_dict, orient='columns')
        final_df.to_csv('final_csvs/holdout_metrics.csv')
