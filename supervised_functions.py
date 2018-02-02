import pandas as pd
import numpy as np
import random
random.seed(56)

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
