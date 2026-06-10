from src.preprocessing import preprocess_data
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.feature_selection import FeatureSelection
from src.sequence_generator import SequenceGenerator
from src.baseline_models import BaselineModels
from src.models import LSTMModel
from src.evaluation import ModelEvaluation
from src.plots import PlotGenerator


def run_pipeline():

    print("\n==============================")
    print("APPLIANCE ENERGY PIPELINE STARTED")
    print("==============================")

    # ===================================
    # 1. PREPROCESSING
    # ===================================

    print("\nSTEP 1: Preprocessing")

    preprocess_data(
        input_path="data/raw/energy_data_set.csv",
        output_path="data/processed/energy_data_processed.csv"
    )

    # ===================================
    # 2. EDA (OPTIONAL - COMMENT IF NOT NEEDED)
    # ===================================

    print("\nSTEP 2: EDA")

    eda = EDA("data/processed/energy_data_processed.csv")
    eda.run_eda()

    # ===================================
    # 3. FEATURE ENGINEERING
    # ===================================

    print("\nSTEP 3: Feature Engineering")

    fe = FeatureEngineering(
        input_path="data/processed/energy_data_processed.csv",
        output_path="data/processed/feature_engineered_data.csv"
    )
    fe.run()

    # ===================================
    # 4. FEATURE SELECTION
    # ===================================

    print("\nSTEP 4: Feature Selection")

    fs = FeatureSelection(
        input_path="data/processed/feature_engineered_data.csv",
        output_path="data/processed/selected_features_data.csv"
    )
    fs.run()

    # ===================================
    # 5. SEQUENCE GENERATION
    # ===================================

    print("\nSTEP 5: Sequence Generation")

    sg = SequenceGenerator(
        input_path="data/processed/selected_features_data.csv",
        output_dir="data/processed/sequences",
        time_steps=24
    )
    sg.run()

    # ===================================
    # 6. BASELINE MODELS
    # ===================================

    print("\nSTEP 6: Baseline Models")

    bm = BaselineModels(
        input_path="data/processed/selected_features_data.csv"
    )
    bm.run()

    # ===================================
    # 7. LSTM MODEL TRAINING
    # ===================================

    print("\nSTEP 7: LSTM Training")

    lstm = LSTMModel(
        data_path="data/processed/sequences",
        model_path="outputs/models/lstm_model.h5"
    )
    lstm.run()

    # ===================================
    # 8. MODEL EVALUATION
    # ===================================

    print("\nSTEP 8: Evaluation")

    evaluator = ModelEvaluation(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )
    evaluator.run()

    # ===================================
    # 9. PLOTS
    # ===================================

    print("\nSTEP 9: Plots")

    plotter = PlotGenerator(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )
    plotter.run()

    print("\n==============================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================")


# =========================
# RUN PROJECT
# =========================

if __name__ == "__main__":

    run_pipeline()