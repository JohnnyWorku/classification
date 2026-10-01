from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.feature_engineering import (
    get_aac_features,
    get_combined_features,
    get_dipeptide_features,
    get_svd_embedding_features,
    get_tripeptide_frequency,
)
from models.lightgbm import LightGBMModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from utils.logger import logger
from utils.train import run_pipeline_for_features

PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

# Start with ["AAC"] as a quick smoke test, then add the others one at a time.
FEATURES_TO_RUN = ["AAC"]
# available_features = ["AAC", "Dipeptide", "Tripeptide", "Combined", "SVD Embedding"]

# While developing on limited hardware, set e.g. 0.1 to train on a stratified 10% of train.
# Dev and test are never subsampled. Set to None for the final full run.
TRAIN_SUBSAMPLE_FRAC = None

USE_CACHE = True
RANDOM_STATE = 42

FEATURE_BUILDERS = {
    "AAC": lambda tr, y, dv, te: get_aac_features(tr, y, dv, te, return_names=True),
    "Dipeptide": lambda tr, y, dv, te: get_dipeptide_features(tr, y, dv, te, return_names=True),
    "Tripeptide": lambda tr, y, dv, te: get_tripeptide_frequency(tr, y, dv, te, return_names=True),
    "Combined": lambda tr, y, dv, te: get_combined_features(tr, y, dv, te, return_names=True),
    "SVD Embedding": lambda tr, y, dv, te: get_svd_embedding_features(tr, dv, te, return_names=True),
}

# Factories build a fresh, untrained model each time, so feature sets never share fitted state.
BASE_MODEL_FACTORIES = {
    "rf": lambda: RandomForestModel(n_estimators=100, max_depth=25, random_state=RANDOM_STATE),
    "lgb": lambda: LightGBMModel(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE),
    "svm": lambda: SVMModel(C=0.1, random_state=RANDOM_STATE),
}


def load_split(filepath):
    """Loads a cleaned CSV split and removes rows with missing targets or sequences."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["seq", "main_ec"]).reset_index(drop=True)
    return df["seq"].values, df["main_ec"].values


def get_features(name, X_train_raw, y_train, X_dev_raw, X_test_raw, tag):
    """Builds (or loads from cache) the feature matrices and feature names for one representation."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name.lower().replace(' ', '_')}_{tag}.npz"

    if USE_CACHE and cache_file.exists():
        logger.info("Loading cached features: %s", cache_file)
        data = np.load(cache_file, allow_pickle=False)
        return (data["train"], data["dev"], data["test"]), list(data["names"])

    logger.info("Extracting '%s' features...", name)
    (X_train, X_dev, X_test), names = FEATURE_BUILDERS[name](X_train_raw, y_train, X_dev_raw, X_test_raw)
    np.savez_compressed(cache_file, train=X_train, dev=X_dev, test=X_test, names=np.array(names))
    return (X_train, X_dev, X_test), names


def main():
    logger.info("Loading pre-split datasets...")
    X_train_raw, y_train = load_split(PROCESSED_DIR / "SwissProt-EC-train_updated.csv")
    X_dev_raw, y_dev = load_split(PROCESSED_DIR / "SwissProt-EC-dev_updated.csv")
    X_test_raw, y_test = load_split(PROCESSED_DIR / "SwissProt-EC-test_updated.csv")

    if TRAIN_SUBSAMPLE_FRAC is not None:
        X_train_raw, _, y_train, _ = train_test_split(
            X_train_raw, y_train, train_size=TRAIN_SUBSAMPLE_FRAC,
            stratify=y_train, random_state=RANDOM_STATE,
        )
        tag = f"sub{TRAIN_SUBSAMPLE_FRAC}"
        logger.warning("Training on a %.0f%% stratified subsample of train.", 100 * TRAIN_SUBSAMPLE_FRAC)
    else:
        tag = "full"

    logger.info("Data splits loaded -> Train: %d, Dev: %d, Test: %d",
                len(X_train_raw), len(X_dev_raw), len(X_test_raw))

    summary = []
    for feature_name in FEATURES_TO_RUN:
        (X_train, X_dev, X_test), feature_names = get_features(
            feature_name, X_train_raw, y_train, X_dev_raw, X_test_raw, tag
        )
        logger.info("%s feature matrix: train %s, dev %s, test %s",
                    feature_name, X_train.shape, X_dev.shape, X_test.shape)

        outcome = run_pipeline_for_features(
            feature_name, X_train, X_dev, X_test, y_train, y_dev, y_test,
            base_model_factories=BASE_MODEL_FACTORIES,
            feature_names=feature_names,
            output_dir=RESULTS_DIR,
        )

    best = outcome["best_model"]
    best_row = outcome["test_results"].set_index("Model").loc[best]
    summary.append({
        "Features": feature_name,
        "Best model (chosen on dev)": best,
        "Test Macro F1": best_row["Macro F1"],
        "Test MCC": best_row["MCC"],
        "Test AUPRC": best_row["AUPRC (macro)"],
        "Test Accuracy": best_row["Accuracy"],
    })

    summary_df = pd.DataFrame(summary)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / f"{feature_name}_summary_{tag}.csv", index=False)
    logger.info("================ Summary across feature sets ================\n%s",
                summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
