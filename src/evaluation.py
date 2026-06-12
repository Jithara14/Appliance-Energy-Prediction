import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)


class ModelEvaluation:

    def __init__(self, model_path, data_path):

        self.model_path = model_path
        self.data_path = data_path

        os.makedirs("outputs/metrics", exist_ok=True)
        os.makedirs("outputs/plots", exist_ok=True)

    # =========================
    # LOAD MODEL
    # =========================
    def load_model(self):

        print("\nLoading trained model...")

        model = load_model(self.model_path, compile=False)

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
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Explained Variance
        evs = explained_variance_score(y_test, y_pred)

        results = pd.DataFrame([{
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "R2_Score": r2,
            "Explained_Variance": evs
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
    # RESIDUAL ANALYSIS
    # =========================
    def residual_analysis(self, y_test, y_pred):

        residuals = y_test - y_pred

        plt.figure()
        plt.hist(residuals, bins=50)
        plt.title("Residual Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")

        plt.savefig("outputs/plots/residual_distribution.png")
        plt.close()

        print("Saved residual plot.")

    # =========================
    # ACTUAL VS PREDICTED
    # =========================
    def plot_predictions(self, y_test, y_pred):

        plt.figure()
        plt.plot(y_test[:200], label="Actual")
        plt.plot(y_pred[:200], label="Predicted")
        plt.title("Actual vs Predicted Energy Consumption")
        plt.legend()

        plt.savefig("outputs/plots/actual_vs_predicted.png")
        plt.close()

        print("Saved prediction plot.")

    # =========================
    # ERROR OVER TIME
    # =========================
    def error_over_time(self, y_test, y_pred):

        errors = np.abs(y_test - y_pred)

        plt.figure()
        plt.plot(errors[:500])
        plt.title("Absolute Error Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Error")

        plt.savefig("outputs/plots/error_over_time.png")
        plt.close()

        print("Saved error trend plot.")

    # =========================
    # FULL PIPELINE
    # =========================
    def run(self):

        model = self.load_model()

        X_test, y_test = self.load_data()

        y_pred = self.predict(model, X_test)

        results = self.evaluate(y_test, y_pred)

        self.save_results(results)

        self.residual_analysis(y_test, y_pred)
        self.plot_predictions(y_test, y_pred)
        self.error_over_time(y_test, y_pred)

        print("\nEvaluation Completed Successfully!")


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":

    evaluator = ModelEvaluation(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )

    evaluator.run()