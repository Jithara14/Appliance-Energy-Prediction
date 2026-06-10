import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model as keras_load_model


class PlotGenerator:

    def __init__(self, model_path, data_path):

        self.model_path = model_path
        self.data_path = data_path

        os.makedirs("outputs/plots", exist_ok=True)

    # =========================
    # LOAD DATA + MODEL (FIXED)
    # =========================

    def load_data(self):

        print("\nLoading data for plotting...")

        X_test = np.load(self.data_path + "/X_test.npy")
        y_test = np.load(self.data_path + "/y_test.npy")

        print("Loading trained model...")

        # 🔥 FIX: compile=False avoids keras error
        model = keras_load_model(self.model_path, compile=False)

        # Safe compile (optional but stable)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        y_pred = model.predict(X_test)

        return y_test, y_pred

    # =========================
    # ACTUAL VS PREDICTED
    # =========================

    def plot_actual_vs_predicted(self, y_test, y_pred):

        print("\nPlotting Actual vs Predicted...")

        plt.figure(figsize=(10, 5))

        plt.plot(y_test[:200], label="Actual")
        plt.plot(y_pred[:200], label="Predicted")

        plt.title("Actual vs Predicted Energy Consumption")
        plt.xlabel("Samples")
        plt.ylabel("Energy")
        plt.legend()

        plt.savefig("outputs/plots/actual_vs_predicted.png")
        plt.close()

    # =========================
    # ERROR DISTRIBUTION
    # =========================

    def plot_error_distribution(self, y_test, y_pred):

        print("\nPlotting Error Distribution...")

        errors = y_test - y_pred.flatten()

        plt.figure(figsize=(8, 5))

        plt.hist(errors, bins=50)

        plt.title("Prediction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")

        plt.savefig("outputs/plots/error_distribution.png")
        plt.close()

    # =========================
    # RUN PIPELINE
    # =========================

    def run(self):

        y_test, y_pred = self.load_data()

        self.plot_actual_vs_predicted(y_test, y_pred)
        self.plot_error_distribution(y_test, y_pred)

        print("\nPlots generated successfully!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    plotter = PlotGenerator(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )

    plotter.run()