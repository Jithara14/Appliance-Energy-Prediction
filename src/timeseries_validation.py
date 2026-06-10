import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesValidation:

    def __init__(self, data_path, n_splits=5):

        self.data_path = data_path
        self.n_splits = n_splits

        os.makedirs("outputs/metrics", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading dataset for time-series validation...")

        df = pd.read_csv(self.data_path)

        X = df.drop(columns=["Appliances"]).values
        y = df["Appliances"].values

        return X, y

    # =========================
    # SPLIT FUNCTION (WALK FORWARD)
    # =========================

    def create_splits(self, X, y):

        print("\nCreating time-series splits...")

        fold_size = len(X) // (self.n_splits + 1)

        splits = []

        for i in range(1, self.n_splits + 1):

            train_end = fold_size * i
            test_end = fold_size * (i + 1)

            X_train = X[:train_end]
            y_train = y[:train_end]

            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            splits.append((X_train, X_test, y_train, y_test))

        return splits

    # =========================
    # EVALUATE FOLD
    # =========================

    def evaluate_fold(self, X_train, X_test, y_train, y_test):

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        return mae, rmse

    # =========================
    # RUN VALIDATION
    # =========================

    def run(self):

        X, y = self.load_data()

        splits = self.create_splits(X, y)

        results = []

        print("\nRunning Time-Series Cross Validation...")

        for i, (X_train, X_test, y_train, y_test) in enumerate(splits):

            mae, rmse = self.evaluate_fold(X_train, X_test, y_train, y_test)

            results.append([i + 1, mae, rmse])

            print(f"Fold {i+1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        results_df = pd.DataFrame(
            results,
            columns=["Fold", "MAE", "RMSE"]
        )

        results_df.to_csv("outputs/metrics/timeseries_validation.csv", index=False)

        print("\nValidation Results Saved!")

        return results_df


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    validator = TimeSeriesValidation(
        data_path="data/processed/selected_features_data.csv",
        n_splits=5
    )

    validator.run()