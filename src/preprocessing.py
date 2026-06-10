import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(input_path, output_path):

    print("\n==============================")
    print("1. LOADING DATASET")
    print("==============================")

    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    # ===================================
    # 2. DATE CONVERSION
    # ===================================

    print("\n2. Converting Date Column...")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ===================================
    # 3. MISSING VALUE REPORT
    # ===================================

    print("\n3. Missing Value Report:")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values
    })

    os.makedirs("outputs/metrics", exist_ok=True)

    missing_df.to_csv("outputs/metrics/missing_values.csv", index=False)

    print(missing_df)

    # ===================================
    # 4. MISSING VALUE HANDLING
    # ===================================

    print("\n4. Handling Missing Values...")

    df.interpolate(method='linear', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # ===================================
    # 5. OUTLIER REPORT
    # ===================================

    print("\n5. Outlier Report (IQR Method)")

    numeric_cols = df.select_dtypes(include=np.number).columns

    outlier_summary = []

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = ((df[col] < lower) | (df[col] > upper)).sum()

        outlier_summary.append([col, count])

    outlier_df = pd.DataFrame(outlier_summary, columns=["Feature", "Outlier_Count"])

    outlier_df.to_csv("outputs/metrics/outlier_report.csv", index=False)

    print(outlier_df)

    # ===================================
    # 6. OUTLIER TREATMENT (IQR CAPPING)
    # ===================================

    print("\n6. Treating Outliers...")

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    # ===================================
    # 7. DUPLICATE REMOVAL
    # ===================================

    print("\n7. Removing Duplicates...")

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    print(f"Duplicates Removed: {before - after}")

    # ===================================
    # 8. FEATURE SCALING SUPPORT
    # ===================================

    print("\n8. Scaling Features...")

    scaler = MinMaxScaler()

    target_column = "Appliances"

    X = df.drop(columns=["date", target_column])
    y = df[target_column]

    X_scaled = scaler.fit_transform(X)

    # ===================================
    # 9. SAVE PROCESSED DATASET
    # ===================================

    print("\n9. Saving Processed Dataset...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df[target_column] = y.values

    processed_df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print("Final Shape:", processed_df.shape)

    return processed_df


if __name__ == "__main__":

    preprocess_data(
        input_path="data/raw/energy_data_set.csv",
        output_path="data/processed/energy_data_processed.csv"
    )