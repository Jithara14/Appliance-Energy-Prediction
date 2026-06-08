import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data
from src.feature_engineering import split_data, scale_data
from src.model import build_lstm
from src.train import create_sequences, train_model, predict
from src.evaluate import evaluate_model
from src.utils import plot_predictions, plot_training


# =========================
# STEP 1: Preprocessing
# =========================
df = preprocess_data(
    "data/raw/energy_data_set.csv",
    "data/processed/processed_energy_data.csv"
)

# =========================
# STEP 2: Load processed
# =========================
df = pd.read_csv("data/processed/processed_energy_data.csv")
df['date'] = pd.to_datetime(df['date'])

# =========================
# STEP 3: Split
# =========================
train_df, test_df = split_data(df)

X_train = train_df.drop(['Appliances', 'date'], axis=1)
y_train = train_df['Appliances']

X_test = test_df.drop(['Appliances', 'date'], axis=1)
y_test = test_df['Appliances']

# =========================
# STEP 4: Scaling
# =========================
train_df_scaled, test_df_scaled, scaler = scale_data(train_df, test_df)

X_train = train_df_scaled.drop(['Appliances', 'date'], axis=1)
X_test = test_df_scaled.drop(['Appliances', 'date'], axis=1)

y_train = train_df_scaled['Appliances']
y_test = test_df_scaled['Appliances']

# =========================
# STEP 5: Reshape for LSTM
# =========================
X_train = create_sequences(X_train)
X_test = create_sequences(X_test)

# =========================
# STEP 6: Model
# =========================
model = build_lstm((X_train.shape[1], X_train.shape[2]))

# =========================
# STEP 7: Train
# =========================
history = train_model(model, X_train, y_train)

# =========================
# STEP 8: Predict
# =========================
predictions = predict(model, X_test)

# =========================
# STEP 9: Evaluate
# =========================
evaluate_model(y_test, predictions)

# =========================
# STEP 10: Plot
# =========================
plot_predictions(y_test, predictions)
plot_training(history)

# =========================
# STEP 11: Save Model
# =========================
model.save("models/trained_model.h5")

print("\nMODEL SAVED SUCCESSFULLY ✔")