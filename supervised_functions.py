import pandas as pd
import numpy as np
import random
random.seed(56)

def create_metric_graph(model_dict, metric_dict, axs):
    for ax, metric in zip(axs, metric_dict.keys()):
        idx = range(1, len(metric_dict[metric]['col_names'])+1)
        col_labels = metric_dict[metric]['col_names']
        df = metric_dict[metric]['dataframe']
        for model in model_dict.keys():
            model_idx = model_dict[model]['name']
            marker = model_dict[model]['marker']
            color = model_dict[model]['color']
            line = model_dict[model]['linestyle']
            values = df.loc[model_idx,col_labels]
            ax.scatter(idx, values, label=model, s=75, marker=marker, color=color, zorder=2)
            ax.plot(idx, values, label='_nolegend_', linestyle=line, linewidth=1.5, color=color, zorder=1, alpha=0.5)
            ax.legend(framealpha=True, borderpad=1.0, facecolor="white")
        ax.set_xticks(idx)
        ax.set_xticklabels(col_labels)
        ax.set_xlim(0.5, len(col_labels)+1)
        ax.set_ylabel(metric_dict[metric]['ylabel'])
        ax.set_title(metric_dict[metric]['Title'])

def create_missing_data(df):
    # Drop NaNs
    complete_df = df.dropna()

    # Drop DX and DXSUB
    complete_df = complete_df.drop(columns=['DX', 'DXSUB'])

    # Randomly insert NaNs
    nan_inserted_data = complete_df.copy()
    ix = [(row, col) for row in range(complete_df.shape[0]) for col in range(complete_df.shape[1])]
    for row, col in random.sample(ix, int(round(.1*len(ix)))):
        nan_inserted_data.iat[row, col] = np.nan

    missing_mask = nan_inserted_data.isna().any(axis=1)

    return nan_inserted_data, missing_mask, complete_df

def create_mse_df(df, missing_mask, complete_df, solver_list, solver_names):
    solver_df_list = []
    for solver in solver_list:
        solver_df_list.append(test_imputation(df, solver))

    # Create blank dataframe
    mse_df = pd.DataFrame(index=solver_names, columns=df.columns)

    # Add MSE scores to DataFrame
    for solver_df, solver_name in zip(solver_df_list, solver_names):
        mse = calculate_mse(solver_df, complete_df, missing_mask)
        mse_df.loc[solver_name] = mse

    return mse_df

def min_mse(df, solver_names):
    df_bool = df.copy()
    for col in df.columns:
        df_bool[col] = (df_bool[col] == np.min(df)[col])
    print_mins(df_bool, solver_names)

def print_mins(df, names):
    total_n = df.shape[1]
    min_values = list(df.sum(axis=1).values)
    for name, min_freq in zip(names, min_values):
        print(('{} Frequency of Minimum MSE:\t{} of {} features').format(name, min_freq, total_n).expandtabs(50))



def calculate_mse(df, complete_df, mask):
    print(mask)
    mse = ((df[mask] - complete_df[mask]) ** 2).mean()
    return mse

def test_imputation(df, solver):
    """Impute the data using imputation method"""
    impute_data = df.values
    data_index = df.index
    data_cols = df.columns

    impute_data_filled = solver.complete(impute_data)
    impute_df = pd.DataFrame(impute_data_filled, index=data_index, columns=data_cols)
    return impute_df
