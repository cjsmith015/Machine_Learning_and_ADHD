"""Splits the data into holdout and training sets.
Data not available on this github repo because of confidentiality."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_data(df):
    """Clean the data"""
    new_df = df.drop(columns='Unnamed: 0')
    new_df['SSBK_NUMCOMPLETE_Y1']

if __name__ == '__main__':
    full_data = pd.read_csv('data/Christie_diagnosis_20180118.csv')
    train_data, holdout_data = train_test_split(full_data, test_size=0.20, random_state=56)
    train_data.reset_index(drop=True).to_csv('data/train_data.csv')
    holdout_data.reset_index(drop=True).to_csv('data/holdout_data.csv')
