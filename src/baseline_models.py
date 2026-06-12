import pandas as pd
import joblib
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class BaselineModels:

    def __init__(self, data_path):

        self.data_path = data_path

        os.makedirs(
            "outputs/models",
            exist_ok=True
        )

        os.makedirs(
            "outputs/metrics",
            exist_ok=True
        )

    def load_data(self):

        df = pd.read_csv(self.data_path)

        X = df.drop(
            columns=["Appliances"]
        )

        y = df["Appliances"]

        return X, y

    def train_models(self):

        X, y = self.load_data()

        tscv = TimeSeriesSplit(
            n_splits=5
        )

        models = {

            "LinearRegression":
                LinearRegression(),

            "Ridge":
                Ridge(alpha=1.0),

            "DecisionTree":
                DecisionTreeRegressor(
                    max_depth=10,
                    random_state=42
                ),

            "RandomForest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),

            "GradientBoosting":
                GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                ),

            "ExtraTrees":
                ExtraTreesRegressor(
                    n_estimators=100,
                    random_state=42
                )
        }

        results = []

        best_model = None
        best_score = -999

        for name, model in models.items():

            print(f"\nTraining {name}...")

            for train_idx, test_idx in tscv.split(X):

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            rmse = mean_squared_error(
                y_test,
                y_pred
            ) ** 0.5

            r2 = r2_score(
                y_test,
                y_pred
            )

            results.append([
                name,
                mae,
                rmse,
                r2
            ])

            if r2 > best_score:

                best_score = r2
                best_model = model

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "MAE",
                "RMSE",
                "R2"
            ]
        )

        print(results_df)

        results_df.to_csv(
            "outputs/metrics/baseline_results.csv",
            index=False
        )

        joblib.dump(
            best_model,
            "outputs/models/best_baseline.pkl"
        )

        print(
            f"\nBest baseline model saved. R2={best_score:.4f}"
        )

    def run(self):

        self.train_models()


if __name__ == "__main__":

    model = BaselineModels(
        "data/processed/energy_data_processed.csv"
    )

    model.run()