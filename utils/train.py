import numpy as np
import pandas as pd
from utils.evaluation_metrics import (
    accuracy_metric,
    macro_f1_metric,
    mattews_corrcoef_metric,
    precision_metric,
    recall_metric,
)
from utils.logger import logger


def evaluate_ensembles(ensembles, X_train, y_train, X_dev, y_dev):
    """Trains each ensemble on X_train and evaluates performance on X_dev."""
    results = []

    for name, model in ensembles.items():
        logger.info("--- Training %s ---", name)
        model.fit(X_train, y_train)

        # Predict on Dev Set
        y_dev_pred = model.predict(X_dev)

        # Compute Metrics
        acc = accuracy_metric(y_dev, y_dev_pred)
        prec = precision_metric(y_dev, y_dev_pred)
        rec = recall_metric(y_dev, y_dev_pred)
        f1_macro = macro_f1_metric(y_dev, y_dev_pred)
        mcc = mattews_corrcoef_metric(y_dev, y_dev_pred)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision (Macro)": prec,
                "Recall (Macro)": rec,
                "Macro F1": f1_macro,
                "MCC": mcc,
            }
        )

    return results


def run_pipeline_for_features(
    feature_name, X_train, X_dev, X_test, y_train, y_dev, y_test, ensembles
):
    """Runs training, validation, and testing for a given feature matrix representation."""
    logger.info("================ Running Pipeline for: %s ================", feature_name)

    results = evaluate_ensembles(ensembles, X_train, y_train, X_dev, y_dev)
    results_df = pd.DataFrame(results)

    # Log validation results
    logger.info("================ Validation Results (%s - Dev Set) ================", feature_name)
    logger.info("\n%s", results_df.to_string(index=False))

    # Evaluate the best model on the test set
    best_model_name = results_df.sort_values(by="Macro F1", ascending=False).iloc[0]["Model"]
    best_ensemble = ensembles[best_model_name]

    logger.info("Evaluating Best Model (%s) on Holdout Test Set...", best_model_name)

    y_test_pred = best_ensemble.predict(X_test)

    test_f1 = macro_f1_metric(y_test, y_test_pred)
    test_mcc = mattews_corrcoef_metric(y_test, y_test_pred)

    logger.info("Test Set Macro F1: %.4f", test_f1)
    logger.info("Test Set MCC: %.4f", test_mcc)

    return results_df