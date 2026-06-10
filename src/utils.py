import os
import json
import joblib
import numpy as np
import pandas as pd


# =========================
# FILE HANDLING
# =========================

def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# DATA LOADER
# =========================

def load_csv(path):

    print(f"\nLoading data from: {path}")

    return pd.read_csv(path)


# =========================
# SAVE MODEL
# =========================

def save_model(model, path):

    ensure_dir(os.path.dirname(path))

    joblib.dump(model, path)

    print(f"\nModel saved at: {path}")


# =========================
# LOAD MODEL
# =========================

def load_model(path):

    print(f"\nLoading model from: {path}")

    return joblib.load(path)


# =========================
# SAVE JSON
# =========================

def save_json(data, path):

    ensure_dir(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nJSON saved at: {path}")


# =========================
# LOAD JSON
# =========================

def load_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    return data


# =========================
# TIME SERIES TRAIN-TEST SPLIT
# =========================

def time_series_split(df, target_col="Appliances", split_ratio=0.8):

    print("\nPerforming time-series split...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    split_index = int(len(df) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


# =========================
# ROOT LOGGER
# =========================

def log(message):

    print(f"[LOG]: {message}")