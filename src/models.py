import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


class LSTMModel:

    def __init__(self, data_path, model_path):

        self.data_path = data_path
        self.model_path = model_path

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading sequence data...")

        X_train = np.load(self.data_path + "/X_train.npy")
        X_test = np.load(self.data_path + "/X_test.npy")
        y_train = np.load(self.data_path + "/y_train.npy")
        y_test = np.load(self.data_path + "/y_test.npy")

        print("X_train shape:", X_train.shape)

        return X_train, X_test, y_train, y_test

    # =========================
    # BUILD MODEL
    # =========================

    def build_model(self, input_shape):

        print("\nBuilding LSTM model...")

        model = Sequential()

        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        model.summary()

        return model

    # =========================
    # TRAIN MODEL
    # =========================

    def train_model(self, model, X_train, y_train, X_test, y_test):

        print("\nTraining LSTM model...")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        return model, history

    # =========================
    # SAVE MODEL
    # =========================

    def save_model(self, model):

        print("\nSaving trained model...")

        model.save(self.model_path)

        print(f"Model saved at: {self.model_path}")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        X_train, X_test, y_train, y_test = self.load_data()

        input_shape = (X_train.shape[1], X_train.shape[2])

        model = self.build_model(input_shape)

        model, history = self.train_model(model, X_train, y_train, X_test, y_test)

        self.save_model(model)

        print("\nLSTM Training Completed!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    lstm = LSTMModel(
        data_path="data/processed/sequences",
        model_path="outputs/models/lstm_model.h5"
    )

    lstm.run()