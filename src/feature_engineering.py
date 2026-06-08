import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def scale_data(train_df, test_df):

    scaler = MinMaxScaler()

    numeric_cols = train_df.select_dtypes(include=np.number).columns

    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, test_df, scaler


def split_data(df):

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    return train, test