"""Splits the data into holdout and training sets.
Data not available on this github repo because of confidentiality."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def write_csvs(df):
    """Write train and holdout data to csvs"""
    train_data, holdout_data = train_test_split(df, test_size=0.20, random_state=56)
    train_data.to_csv('data/train_data.csv')
    holdout_data.to_csv('data/holdout_data.csv')

if __name__ == '__main__':
    full_data = pd.read_csv('data/Christie_diagnosis_20180118.csv')
    write_csvs(full_data)
