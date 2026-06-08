import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# Convert to LSTM sequences
# ==============================
def create_sequences(X):

    X = np.array(X)

    # LSTM expects 3D input: (samples, timesteps, features)
    return X.reshape((X.shape[0], 1, X.shape[1]))


# ==============================
# Train model (baseline training)
# ==============================
def train_model(model, X_train, y_train, batch_size=32, epochs=30):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return history


# ==============================
# Prediction
# ==============================
def predict(model, X_test):

    return model.predict(X_test)