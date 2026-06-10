import numpy as np
import os
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class HyperparameterTuning:

    def __init__(self, data_path):

        self.data_path = data_path

        os.makedirs("outputs/models", exist_ok=True)
        os.makedirs("outputs/metrics", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading sequence data...")

        X_train = np.load(self.data_path + "/X_train.npy")
        X_test = np.load(self.data_path + "/X_test.npy")
        y_train = np.load(self.data_path + "/y_train.npy")
        y_test = np.load(self.data_path + "/y_test.npy")

        return X_train, X_test, y_train, y_test

    # =========================
    # BUILD MODEL
    # =========================

    def build_model(self, units=64, dropout=0.2, input_shape=None):

        model = Sequential()

        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))

        model.add(LSTM(units // 2))
        model.add(Dropout(dropout))

        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        return model

    # =========================
    # TRAIN + EVALUATE
    # =========================

    def train_eval(self, model, X_train, y_train, X_test, y_test, epochs, batch_size):

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop]
        )

        val_loss = min(history.history["val_loss"])

        return val_loss

    # =========================
    # GRID SEARCH (SIMPLE)
    # =========================

    def run_search(self):

        X_train, X_test, y_train, y_test = self.load_data()

        input_shape = (X_train.shape[1], X_train.shape[2])

        param_grid = {
            "units": [32, 64],
            "dropout": [0.2, 0.3],
            "batch_size": [16, 32],
            "epochs": [10]
        }

        best_score = float("inf")
        best_params = None

        print("\nStarting Hyperparameter Tuning...")

        for units in param_grid["units"]:
            for dropout in param_grid["dropout"]:
                for batch_size in param_grid["batch_size"]:
                    for epochs in param_grid["epochs"]:

                        print(f"\nTesting: units={units}, dropout={dropout}, batch={batch_size}")

                        model = self.build_model(
                            units=units,
                            dropout=dropout,
                            input_shape=input_shape
                        )

                        score = self.train_eval(
                            model,
                            X_train, y_train,
                            X_test, y_test,
                            epochs,
                            batch_size
                        )

                        print("Validation Loss:", score)

                        if score < best_score:
                            best_score = score
                            best_params = {
                                "units": units,
                                "dropout": dropout,
                                "batch_size": batch_size,
                                "epochs": epochs
                            }

        # Save best parameters
        with open("outputs/metrics/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print("\nBest Parameters Found:", best_params)

        return best_params


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    tuner = HyperparameterTuning(
        data_path="data/processed/sequences"
    )

    tuner.run_search()