import pandas as pd
import numpy as np


def preprocess_data(input_path, output_path):

    print("Loading Dataset...")

    df = pd.read_csv(input_path)

    print("Original Shape:", df.shape)

    # ===================================
    # Date Conversion
    # ===================================

    df['date'] = pd.to_datetime(df['date'])

    # ===================================
    # Missing Values
    # ===================================

    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    df.interpolate(inplace=True)

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    # ===================================
    # Time Features
    # ===================================

    df['hour'] = df['date'].dt.hour

    df['day'] = df['date'].dt.day

    df['month'] = df['date'].dt.month

    df['weekday'] = df['date'].dt.weekday

    df['is_weekend'] = (
        df['weekday'] >= 5
    ).astype(int)

    # ===================================
    # Rolling Features
    # ===================================

    df['rolling_1h'] = (
        df['Appliances']
        .rolling(window=6)
        .mean()
    )

    df['rolling_3h'] = (
        df['Appliances']
        .rolling(window=18)
        .mean()
    )

    # ===================================
    # Lag Features
    # ===================================

    df['lag_1'] = df['Appliances'].shift(1)

    df['lag_3'] = df['Appliances'].shift(3)

    df['lag_6'] = df['Appliances'].shift(6)

    # ===================================
    # Interaction Features
    # ===================================

    if 'T1' in df.columns and 'RH_1' in df.columns:

        df['temp_humidity'] = (
            df['T1'] * df['RH_1']
        )

    # ===================================
    # Remove NaNs from lag/rolling
    # ===================================

    df.dropna(inplace=True)

    # ===================================
    # Outlier Treatment (IQR)
    # ===================================

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)

        upper = q3 + (1.5 * iqr)

        df[col] = np.where(
            df[col] > upper,
            upper,
            df[col]
        )

        df[col] = np.where(
            df[col] < lower,
            lower,
            df[col]
        )

    print("\nFinal Shape:", df.shape)

    # ===================================
    # Save Processed Dataset
    # ===================================

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nProcessed dataset saved to:\n{output_path}"
    )

    return df


if __name__ == "__main__":

    preprocess_data(
        input_path="data/raw/energy_data_set.csv",
        output_path="data/processed/processed_energy_data.csv"
    )