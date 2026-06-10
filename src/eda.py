import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EDA:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

        os.makedirs("outputs/plots", exist_ok=True)

    def load_data(self):

        print("\nLoading data for EDA...")

        self.df = pd.read_csv(self.file_path)

        print("Shape:", self.df.shape)
        print(self.df.head())

    # =========================
    # BASIC INFO
    # =========================

    def basic_info(self):

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

    # =========================
    # MISSING VALUES HEATMAP
    # =========================

    def missing_values(self):

        print("\nPlotting Missing Values...")

        plt.figure(figsize=(10, 5))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")

        plt.title("Missing Values Heatmap")

        plt.savefig("outputs/plots/missing_values_heatmap.png")
        plt.close()

    # =========================
    # CORRELATION HEATMAP
    # =========================

    def correlation_matrix(self):

        print("\nCorrelation Matrix...")

        plt.figure(figsize=(12, 8))

        corr = self.df.corr(numeric_only=True)

        sns.heatmap(corr, cmap="coolwarm", annot=False)

        plt.title("Feature Correlation Heatmap")

        plt.savefig("outputs/plots/correlation_heatmap.png")
        plt.close()

    # =========================
    # ENERGY DISTRIBUTION
    # =========================

    def energy_distribution(self):

        print("\nEnergy Distribution...")

        plt.figure(figsize=(8, 5))

        sns.histplot(self.df["Appliances"], bins=50, kde=True)

        plt.title("Appliances Energy Consumption Distribution")

        plt.savefig("outputs/plots/energy_distribution.png")
        plt.close()

    # =========================
    # TIME BASED ANALYSIS
    # =========================

    def time_analysis(self):

        if "date" in self.df.columns:

            self.df["date"] = pd.to_datetime(self.df["date"])

            self.df["hour"] = self.df["date"].dt.hour

            hourly = self.df.groupby("hour")["Appliances"].mean()

            plt.figure(figsize=(10, 5))

            hourly.plot(kind="line")

            plt.title("Average Energy Consumption by Hour")

            plt.xlabel("Hour")

            plt.ylabel("Energy")

            plt.savefig("outputs/plots/hourly_energy.png")

            plt.close()

    # =========================
    # RUN ALL EDA
    # =========================

    def run_eda(self):

        self.load_data()

        self.basic_info()

        self.missing_values()

        self.correlation_matrix()

        self.energy_distribution()

        self.time_analysis()

        print("\nEDA Completed Successfully!")


# =========================
# RUN FILE
# =========================

if __name__ == "__main__":

    eda = EDA("data/processed/energy_data_processed.csv")
    eda.run_eda()