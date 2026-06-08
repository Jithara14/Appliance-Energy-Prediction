import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_shape, units=64, dropout=0.2, learning_rate=0.001):

    model = Sequential()

    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))

    model.add(LSTM(units // 2))
    model.add(Dropout(dropout))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model


def tune_hyperparameters(X_train, y_train, input_shape):

    param_grid = {
        "units": [64, 128],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32]
    }

    grid = list(ParameterGrid(param_grid))

    best_model = None
    best_score = float("inf")
    best_params = None

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    for params in grid:

        print(f"\nTraining with params: {params}")

        model = build_model(
            input_shape=input_shape,
            units=params["units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"]
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=0
        )

        val_loss = min(history.history['val_loss'])

        print(f"Validation Loss: {val_loss}")

        if val_loss < best_score:
            best_score = val_loss
            best_model = model
            best_params = params

    print("\nBEST PARAMETERS FOUND:")
    print(best_params)
    print("Best Validation Loss:", best_score)

    return best_model, best_params