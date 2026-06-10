import pandas as pd
import numpy as np
import os


class FeatureSelection:

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs("outputs/metrics", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    def load_data(self):

        print("\nLoading feature engineered dataset...")

        df = pd.read_csv(self.input_path)

        print("Shape:", df.shape)

        return df

    # =========================
    # CORRELATION FILTERING
    # =========================

    def remove_high_correlation(self, df, threshold=0.90):

        print("\nRemoving highly correlated features...")

        corr_matrix = df.corr(numeric_only=True)

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]

        print(f"Features removed due to correlation: {len(to_drop)}")

        df.drop(columns=to_drop, inplace=True)

        return df, to_drop

    # =========================
    # FEATURE IMPORTANCE (SIMPLE METHOD)
    # =========================

    def feature_importance(self, df):

        print("\nCalculating feature importance (correlation with target)...")

        target = "Appliances"

        if target not in df.columns:
            print("Target column not found!")
            return

        correlations = df.corr(numeric_only=True)[target].sort_values(ascending=False)

        importance_df = pd.DataFrame({
            "Feature": correlations.index,
            "Correlation": correlations.values
        })

        importance_df.to_csv("outputs/metrics/feature_importance.csv", index=False)

        print(importance_df.head(10))

        return importance_df

    # =========================
    # CLEAN FINAL DATASET
    # =========================

    def clean_target_safe(self, df):

        print("\nFinal cleanup before saving...")

        # Ensure target exists
        if "Appliances" not in df.columns:
            raise ValueError("Target column 'Appliances' missing!")

        return df

    # =========================
    # SAVE DATA
    # =========================

    def save_data(self, df):

        df.to_csv(self.output_path, index=False)

        print(f"\nSaved selected dataset to: {self.output_path}")

    # =========================
    # FULL PIPELINE
    # =========================

    def run(self):

        df = self.load_data()

        df = self.clean_target_safe(df)

        df, removed = self.remove_high_correlation(df)

        self.feature_importance(df)

        self.save_data(df)

        print("\nFeature Selection Completed!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    fs = FeatureSelection(
        input_path="data/processed/feature_engineered_data.csv",
        output_path="data/processed/selected_features_data.csv"
    )

    fs.run()