import pandas as pd
import numpy as np
import os


class SequenceGenerator:

    def __init__(self, input_path, output_dir, time_steps=24):

        self.input_path = input_path
        self.output_dir = output_dir
        self.time_steps = time_steps

        os.makedirs(output_dir, exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading dataset for sequence generation...")

        df = pd.read_csv(self.input_path)

        print("Shape:", df.shape)

        return df

    # =========================
    # CREATE SEQUENCES
    # =========================

    def create_sequences(self, data, target_col):

        print("\nCreating sequences...")

        X, y = [], []

        for i in range(len(data) - self.time_steps):

            X.append(data[i:i + self.time_steps])
            y.append(data[i + self.time_steps, target_col])

        return np.array(X), np.array(y)

    # =========================
    # PREPARE DATA
    # =========================

    def prepare(self, df):

        print("\nPreparing data for LSTM...")

        target_col = df.columns.get_loc("Appliances")

        values = df.values

        X, y = self.create_sequences(values, target_col)

        print("Sequence shape:", X.shape)
        print("Target shape:", y.shape)

        return X, y

    # =========================
    # TRAIN TEST SPLIT (TIME SERIES SAFE)
    # =========================

    def split_data(self, X, y, train_ratio=0.8):

        print("\nSplitting sequences...")

        split_idx = int(len(X) * train_ratio)

        X_train = X[:split_idx]
        X_test = X[split_idx:]

        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

    # =========================
    # SAVE ARRAYS
    # =========================

    def save_data(self, X_train, X_test, y_train, y_test):

        print("\nSaving sequence datasets...")

        np.save(os.path.join(self.output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.output_dir, "y_test.npy"), y_test)

        print("Sequences saved successfully.")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        df = self.load_data()

        X, y = self.prepare(df)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        self.save_data(X_train, X_test, y_train, y_test)

        print("\nSequence Generation Completed!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    sg = SequenceGenerator(
        input_path="data/processed/selected_features_data.csv",
        output_dir="data/processed/sequences",
        time_steps=24
    )

    sg.run()