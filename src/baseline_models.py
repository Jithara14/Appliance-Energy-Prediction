import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


class BaselineModels:

    def __init__(self, input_path):

        self.input_path = input_path

        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/metrics", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading dataset for baseline models...")

        df = pd.read_csv(self.input_path)

        print("Shape:", df.shape)

        return df

    # =========================
    # SPLIT DATA
    # =========================

    def split_data(self, df):

        print("\nSplitting data...")

        target = "Appliances"

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    # =========================
    # TRAIN MODEL
    # =========================

    def train_models(self, X_train, y_train):

        print("\nTraining baseline models...")

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        }

        trained_models = {}

        for name, model in models.items():

            model.fit(X_train, y_train)

            trained_models[name] = model

            print(f"{name} trained.")

        return trained_models

    # =========================
    # EVALUATE MODELS
    # =========================

    def evaluate(self, models, X_test, y_test):

        print("\nEvaluating models...")

        results = []

        for name, model in models.items():

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append([name, mae, rmse, r2])

        results_df = pd.DataFrame(
            results,
            columns=["Model", "MAE", "RMSE", "R2"]
        )

        results_df.to_csv("outputs/metrics/baseline_results.csv", index=False)

        print(results_df)

        return results_df

    # =========================
    # SAVE BEST MODEL
    # =========================

    def save_best_model(self, models, X_test, y_test):

        print("\nSaving best model...")

        best_model = None
        best_score = -np.inf

        for name, model in models.items():

            score = model.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_model = model

        joblib.dump(best_model, "outputs/models/best_model.pkl")

        print(f"Best model saved with R2: {best_score:.4f}")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        df = self.load_data()

        X_train, X_test, y_train, y_test = self.split_data(df)

        models = self.train_models(X_train, y_train)

        self.evaluate(models, X_test, y_test)

        self.save_best_model(models, X_test, y_test)

        print("\nBaseline Modeling Completed!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    bm = BaselineModels(
        input_path="data/processed/selected_features_data.csv"
    )

    bm.run()