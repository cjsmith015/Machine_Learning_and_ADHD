def clean_data(df):
    """Impute the data using Multiple Imputation"""
    impute_data = df.drop(columns=['Unnamed: 0']).values
    data_index = df.drop(columns=['Unnamed: 0']).index
    data_cols = df.drop(columns=['Unnamed: 0']).columns

    solver = MICE()
    impute_data_filled = solver.complete(impute_data)
    impute_df = pd.DataFrame(impute_data_filled, index=data_index, columns=data_cols)
    return impute_df
