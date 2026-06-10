import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluation:

    def __init__(self, model_path, data_path):

        self.model_path = model_path
        self.data_path = data_path

        os.makedirs("outputs/metrics", exist_ok=True)
        os.makedirs("outputs/plots", exist_ok=True)

    # =========================
    # LOAD MODEL (FIXED)
    # =========================

    def load_model(self):

        print("\nLoading trained model...")

        # 🔥 FIX: compile=False avoids keras deserialization error
        model = load_model(self.model_path, compile=False)

        # Re-compile manually for safe evaluation
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        return model

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading test data...")

        X_test = np.load(self.data_path + "/X_test.npy")
        y_test = np.load(self.data_path + "/y_test.npy")

        print("X_test shape:", X_test.shape)

        return X_test, y_test

    # =========================
    # PREDICT
    # =========================

    def predict(self, model, X_test):

        print("\nGenerating predictions...")

        y_pred = model.predict(X_test)

        return y_pred

    # =========================
    # EVALUATION METRICS
    # =========================

    def evaluate(self, y_test, y_pred):

        print("\nEvaluating model...")

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results = pd.DataFrame([{
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        }])

        print("\nResults:")
        print(results)

        return results

    # =========================
    # SAVE RESULTS
    # =========================

    def save_results(self, results):

        output_path = "outputs/metrics/evaluation_results.csv"

        results.to_csv(output_path, index=False)

        print(f"\nSaved results to: {output_path}")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        model = self.load_model()

        X_test, y_test = self.load_data()

        y_pred = self.predict(model, X_test)

        results = self.evaluate(y_test, y_pred)

        self.save_results(results)

        print("\nEvaluation Completed Successfully!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    evaluator = ModelEvaluation(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )

    evaluator.run()