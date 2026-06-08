import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred):

    plt.figure(figsize=(12,5))

    plt.plot(y_true.values[:300], label="Actual")
    plt.plot(y_pred[:300], label="Predicted")

    plt.title("Actual vs Predicted Energy Consumption")
    plt.legend()
    plt.show()


def plot_training(history):

    plt.figure(figsize=(10,5))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')

    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()