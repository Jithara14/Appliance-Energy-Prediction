import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class ANNTrainer:

    def load_data(self):

        X_train = np.load(
            "data/processed/sequences/X_train.npy"
        )

        y_train = np.load(
            "data/processed/sequences/y_train.npy"
        )

        X_train = X_train.reshape(
            X_train.shape[0],
            -1
        )

        return X_train, y_train

    def build_model(self, input_dim):

        model = Sequential()

        model.add(
            Dense(
                128,
                activation="relu",
                input_dim=input_dim
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(
                64,
                activation="relu"
            )
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
            X_train.shape[1]
        )

        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2
        )

        model.save(
            "outputs/models/ann_model.h5"
        )

        print(
            "ANN model saved."
        )