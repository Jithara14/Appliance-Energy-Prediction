from src.preprocessing import preprocess_data
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.feature_selection import FeatureSelection
from src.sequence_generator import SequenceGenerator

from src.baseline_models import BaselineModels

from src.ann_model import ANNTrainer
from src.gru_model import GRUTrainer
from src.models import LSTMModel

from src.evaluation import ModelEvaluation
from src.plots import PlotGenerator


def run_pipeline():

    print("\n" + "=" * 60)
    print("APPLIANCE ENERGY PREDICTION PIPELINE STARTED")
    print("=" * 60)

    # ==================================================
    # STEP 1 - PREPROCESSING
    # ==================================================

    print("\nSTEP 1: PREPROCESSING")

    preprocess_data(
        input_path="data/raw/energy_data_set.csv",
        output_path="data/processed/energy_data_processed.csv"
    )

    # ==================================================
    # STEP 2 - EDA
    # ==================================================

    print("\nSTEP 2: EDA")

    eda = EDA(
        "data/processed/energy_data_processed.csv"
    )

    eda.run_eda()

    # ==================================================
    # STEP 3 - FEATURE ENGINEERING
    # ==================================================

    print("\nSTEP 3: FEATURE ENGINEERING")

    fe = FeatureEngineering(
        input_path="data/processed/energy_data_processed.csv",
        output_path="data/processed/feature_engineered_data.csv"
    )

    fe.run()

    # ==================================================
    # STEP 4 - FEATURE SELECTION
    # ==================================================

    print("\nSTEP 4: FEATURE SELECTION")

    fs = FeatureSelection(
        input_path="data/processed/feature_engineered_data.csv",
        output_path="data/processed/selected_features_data.csv"
    )

    fs.run()

    # ==================================================
    # STEP 5 - SEQUENCE GENERATION
    # ==================================================

    print("\nSTEP 5: SEQUENCE GENERATION")

    sg = SequenceGenerator(
        input_path="data/processed/selected_features_data.csv",
        output_dir="data/processed/sequences",
        time_steps=24
    )

    sg.run()

    # ==================================================
    # STEP 6 - BASELINE MODELS
    # ==================================================

    print("\nSTEP 6: BASELINE MODEL TRAINING")

    bm = BaselineModels(
        data_path="data/processed/selected_features_data.csv"
    )

    bm.run()

    # ==================================================
    # STEP 7 - ANN MODEL
    # ==================================================

    print("\nSTEP 7: ANN TRAINING")

    ann = ANNTrainer()

    ann.run()

    # ==================================================
    # STEP 8 - GRU MODEL
    # ==================================================

    print("\nSTEP 8: GRU TRAINING")

    gru = GRUTrainer()

    gru.run()

    # ==================================================
    # STEP 9 - LSTM MODEL
    # ==================================================

    print("\nSTEP 9: LSTM TRAINING")

    lstm = LSTMModel(
        data_path="data/processed/sequences",
        model_path="outputs/models/lstm_model.h5"
    )

    lstm.run()

    # ==================================================
    # STEP 10 - MODEL EVALUATION
    # ==================================================

    print("\nSTEP 10: MODEL EVALUATION")

    evaluator = ModelEvaluation(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )

    evaluator.run()

    # ==================================================
    # STEP 11 - VISUALIZATION
    # ==================================================

    print("\nSTEP 11: PLOT GENERATION")

    plotter = PlotGenerator(
        model_path="outputs/models/lstm_model.h5",
        data_path="data/processed/sequences"
    )

    plotter.run()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()