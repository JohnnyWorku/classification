import numpy as np
import pandas as pd
from data.feature_engineering import (
    get_aac_features,
    get_dipeptide_features,
    get_svd_embedding_features,
    get_tripeptide_frequency,
)
from models.ensembling_models import HardVotingModel, SoftVotingModel, StackingModel
from models.lightgbm import LightGBMModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from utils.train import run_pipeline_for_features
from utils.logger import logger


def load_split(filepath):
    """Loads CSV splits and removes rows with missing targets or sequences."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["seq", "main_ec"]).reset_index(drop=True)
    return df["seq"].values, df["main_ec"].values


def main():
    logger.info("Loading pre-split datasets...")
    X_train_raw, y_train = load_split("SwissProt-EC-train_updated.csv")
    X_dev_raw, y_dev = load_split("SwissProt-EC-dev_updated.csv")
    X_test_raw, y_test = load_split("SwissProt-EC-test_updated.csv")

    logger.info(
        "Data splits loaded -> Train: %d, Dev: %d, Test: %d",
        len(X_train_raw),
        len(X_dev_raw),
        len(X_test_raw),
    )

    logger.info("Extracting features using feature engineering functions...")
    # Extract feature representations (returns tuples of (X_train, X_dev, X_test))
    # aac_features = get_aac_features(X_train_raw, y_train, X_dev_raw, X_test_raw)
    dipep_features = get_dipeptide_features(X_train_raw, y_train, X_dev_raw, X_test_raw)
    # tripep_features = get_tripeptide_frequency(X_train_raw, y_train, X_dev_raw, X_test_raw)
    # embed_features = get_svd_embedding_features(X_train_raw, ytrain, X_dev_raw, X_test_raw)

    feature_extractors = [
        # ("AAC", aac_features),
        ("Dipeptide", dipep_features),
        # ("Tripeptide", tripep_features),
        # ("SVD Embedding", embed_features),
    ]

    # Initialize Base Model Wrappers
    rf_wrapper = RandomForestModel(n_estimators=100, max_depth=25, random_state=42)
    lgb_wrapper = LightGBMModel(n_estimators=100, learning_rate=0.05, random_state=42)
    svm_wrapper = SVMModel(C=1.0, random_state=42)

    base_estimators = [
        ("rf", rf_wrapper.rf_model),
        ("lgb", lgb_wrapper.lightgbm_model),
        ("svm", svm_wrapper.svm_model),
    ]

    # Loop over feature representations
    for feature_name, (X_train, X_dev, X_test) in feature_extractors:
        # Re-initialize fresh ensemble instances for each feature representation
        ensembles = {
            "Hard Voting": HardVotingModel(base_estimators),
            "Soft Voting": SoftVotingModel(base_estimators),
            "Stacking": StackingModel(base_estimators),
        }

        run_pipeline_for_features(
            feature_name, X_train, X_dev, X_test, y_train, y_dev, y_test, ensembles
        )


if __name__ == "__main__":
    main()
