import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


class GRUTrainer:

    def __init__(self):

        pass

    def load_data(self):

        X_train = np.load(
            "data/processed/sequences/X_train.npy"
        )

        y_train = np.load(
            "data/processed/sequences/y_train.npy"
        )

        return X_train, y_train

    def build_model(self, input_shape):

        model = Sequential()

        model.add(
            GRU(
                64,
                return_sequences=False,
                input_shape=input_shape
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(32, activation="relu")
        )

        model.add(
            Dense(1)
        )

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

        return model

    def run(self):

        X_train, y_train = self.load_data()

        model = self.build_model(
            (X_train.shape[1],
             X_train.shape[2])
        )

        early_stop = EarlyStopping(
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        model.save(
            "outputs/models/gru_model.h5"
        )

        print(
            "GRU model saved."
        )