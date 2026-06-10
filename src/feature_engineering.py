import pandas as pd
import numpy as np
import os


class FeatureEngineering:

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading dataset for feature engineering...")

        df = pd.read_csv(self.input_path)

        print("Shape:", df.shape)

        return df

    # =========================
    # TIME FEATURES (SAFE FIX)
    # =========================

    def time_features(self, df):

        print("\nCreating time-based features...")

        # SAFE CHECK (IMPORTANT FIX)
        if 'date' not in df.columns:
            print("No 'date' column found → skipping time features.")
            return df

        df['date'] = pd.to_datetime(df['date'])

        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)

        return df

    # =========================
    # LAG FEATURES
    # =========================

    def lag_features(self, df):

        print("\nCreating lag features...")

        df['lag_1'] = df['Appliances'].shift(1)
        df['lag_3'] = df['Appliances'].shift(3)
        df['lag_6'] = df['Appliances'].shift(6)

        return df

    # =========================
    # ROLLING FEATURES
    # =========================

    def rolling_features(self, df):

        print("\nCreating rolling features...")

        df['rolling_1h'] = df['Appliances'].rolling(window=6).mean()
        df['rolling_3h'] = df['Appliances'].rolling(window=18).mean()

        return df

    # =========================
    # INTERACTION FEATURES
    # =========================

    def interaction_features(self, df):

        print("\nCreating interaction features...")

        if 'T1' in df.columns and 'RH_1' in df.columns:
            df['temp_humidity'] = df['T1'] * df['RH_1']

        return df

    # =========================
    # CLEAN DATA
    # =========================

    def clean_data(self, df):

        print("\nCleaning dataset...")

        df.dropna(inplace=True)

        return df

    # =========================
    # SAVE DATA
    # =========================

    def save_data(self, df):

        df.to_csv(self.output_path, index=False)

        print(f"\nSaved feature engineered data to: {self.output_path}")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        df = self.load_data()

        df = self.time_features(df)
        df = self.lag_features(df)
        df = self.rolling_features(df)
        df = self.interaction_features(df)

        df = self.clean_data(df)

        self.save_data(df)

        print("\nFeature Engineering Completed Successfully!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    fe = FeatureEngineering(
        input_path="data/processed/energy_data_processed.csv",
        output_path="data/processed/feature_engineered_data.csv"
    )

    fe.run()