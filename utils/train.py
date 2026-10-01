import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from models.ensembling_models import (
    HardVotingModel,
    SoftVotingModel,
    StackingModel,
    collect_probas,
)
from utils.evaluation_metrics import (
    confusion_matrix_normalized,
    easiest_and_hardest_classes,
    ensemble_gain,
    evaluate_all,
    get_feature_importance,
    most_confused_pairs,
    per_class_metrics_table,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from utils.logger import logger

ENSEMBLE_NAMES = ["Hard Voting", "Soft Voting", "Stacking"]


def fit_base_models(factories, X_train, y_train, X_dev, y_dev):
    """Fits every base model exactly once on the training set."""
    fitted = {}
    for name, make_model in factories.items():
        model = make_model()
        start = time.perf_counter()

        # Models that support early stopping (LightGBM) get the dev set as validation data
        if "X_val" in inspect.signature(model.fit).parameters:
            model.fit(X_train, y_train, X_val=X_dev, y_val=y_dev)
        else:
            model.fit(X_train, y_train)

        logger.info("Trained '%s' in %.1fs", name, time.perf_counter() - start)
        fitted[name] = model
    return fitted


def _predictions_for_split(fitted, X, probas, hard, soft, stack_pred_proba):
    """Builds {model_name: (y_pred, y_proba)} for base models and the three ensembles."""
    preds = {name: (model.predict(X), probas[name]) for name, model in fitted.items()}
    preds["Hard Voting"] = (hard.predict_from_probas(probas), None)  # votes only, no probabilities
    preds["Soft Voting"] = (
        soft.predict_from_probas(probas),
        np.mean([probas[n] for n in fitted], axis=0),
    )
    preds["Stacking"] = stack_pred_proba
    return preds


def run_pipeline_for_features(
    feature_name,
    X_train,
    X_dev,
    X_test,
    y_train,
    y_dev,
    y_test,
    base_model_factories,
    feature_names=None,
    output_dir="results",
):
    """Fit base models once -> build ensembles from cached probabilities -> select on dev -> analyse on test."""
    logger.info("================ Running Pipeline for: %s ================", feature_name)
    out_dir = Path(output_dir) / feature_name.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base models: trained once, probabilities computed once per split
    fitted = fit_base_models(base_model_factories, X_train, y_train, X_dev, y_dev)
    probas_dev = collect_probas(fitted, X_dev)
    probas_test = collect_probas(fitted, X_test)

    hard, soft, stack = HardVotingModel(fitted), SoftVotingModel(fitted), StackingModel(fitted)
    classes = stack.classes_
    base_names = list(fitted)

    # 2. Dev-set predictions.
    # Stacking is trained on dev, so its dev score comes from cross-validation inside dev.
    Z_dev = np.hstack([probas_dev[n] for n in stack.names])
    stack_dev_proba = cross_val_predict(stack.meta, Z_dev, y_dev, cv=3, method="predict_proba")
    stack_dev = (classes[stack_dev_proba.argmax(axis=1)], stack_dev_proba)
    dev_preds = _predictions_for_split(fitted, X_dev, probas_dev, hard, soft, stack_dev)

    dev_results = evaluate_all(y_dev, dev_preds, classes)
    logger.info("================ Validation Results (%s - Dev Set) ================", feature_name)
    logger.info("\n%s", dev_results.round(4).to_string(index=False))
    dev_results.to_csv(out_dir / "dev_results.csv", index=False)

    # 3. Model selection uses the dev set only
    best_name = dev_results.iloc[0]["Model"]
    logger.info("Best model on dev (Macro F1): %s", best_name)

    # 4. Fit the stacking meta-learner on dev, then predict on test
    stack.fit_from_probas(probas_dev, y_dev)
    Z_test = np.hstack([probas_test[n] for n in stack.names])
    stack_test = (stack.predict_from_probas(probas_test), stack.meta.predict_proba(Z_test))
    test_preds = _predictions_for_split(fitted, X_test, probas_test, hard, soft, stack_test)

    test_results = evaluate_all(y_test, test_preds, classes)
    logger.info("================ Test Results (%s) ================", feature_name)
    logger.info("\n%s", test_results.round(4).to_string(index=False))
    test_results.to_csv(out_dir / "test_results.csv", index=False)

    # 5. Final analysis on the model chosen via dev
    analysis = run_final_analysis(
        best_name, test_preds[best_name], y_test, classes, fitted, test_results, base_names,
        feature_names, out_dir,
    )

    return {
        "dev_results": dev_results,
        "test_results": test_results,
        "best_model": best_name,
        "analysis": analysis,
    }


def run_final_analysis(best_name, best_pred, y_test, classes, fitted, test_results, base_names,
                       feature_names, out_dir):
    """Confusion matrix, per-class table, ROC, feature importance and the error-analysis numbers."""
    y_pred, y_proba = best_pred
    logger.info("---------- Final analysis: %s (test set) ----------", best_name)

    # 1. Confusion matrix, normalized by true class
    cm_df = confusion_matrix_normalized(y_test, y_pred, classes)
    cm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")
    plot_confusion_matrix(y_test, y_pred, classes, title=f"{best_name}: normalized confusion matrix",
                          save_path=out_dir / "confusion_matrix.png")
    logger.info("Confusion matrix (rows = actual):\n%s", cm_df.round(3).to_string())

    # 2. Per-class precision / recall / F1
    per_class = per_class_metrics_table(y_test, y_pred, classes)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    logger.info("Per-class metrics:\n%s", per_class.round(4).to_string(index=False))

    # 3. One-vs-rest ROC curves (hard voting has no probabilities, so it is skipped)
    if y_proba is not None:
        aucs = plot_roc_curves(y_test, y_proba, classes, title=f"{best_name}: one-vs-rest ROC",
                               save_path=out_dir / "roc_curves.png")
        logger.info("Per-class ROC-AUC:\n%s", aucs.round(4).to_string())
    else:
        logger.info("Best model is hard voting (no probabilities), skipping ROC curves.")

    # 4. Feature importance from every tree-based model that was trained
    importances = {}
    for name, model in fitted.items():
        estimator = getattr(model, "model", model)
        if not hasattr(estimator, "feature_importances_"):
            continue
        if hasattr(estimator, "importance_type"):
            estimator.importance_type = "gain"  # more informative than split counts for LightGBM
        imp = get_feature_importance(model, feature_names=feature_names, top_n=20)
        imp.to_csv(out_dir / f"feature_importance_{name}.csv", index=False)
        plot_feature_importance(imp, title=f"Top features ({name})",
                                save_path=out_dir / f"feature_importance_{name}.png")
        importances[name] = imp
        logger.info("Top 10 features (%s):\n%s", name, imp.head(10).round(4).to_string(index=False))

    # 5. Numbers for the written error analysis
    easiest, hardest = easiest_and_hardest_classes(y_test, y_pred, classes)
    confusions = most_confused_pairs(y_test, y_pred, classes)
    gain = ensemble_gain(test_results, base_names, ENSEMBLE_NAMES)
    logger.info("Easiest classes (by F1):\n%s", easiest.round(4).to_string(index=False))
    logger.info("Hardest classes (by F1):\n%s", hardest.round(4).to_string(index=False))
    logger.info("Most confused pairs:\n%s", confusions.round(4).to_string(index=False))
    logger.info(
        "Ensemble vs best single model (test Macro F1): %s %.4f vs %s %.4f (gain %+.4f)",
        gain["best_ensemble"][0], gain["best_ensemble"][1],
        gain["best_base"][0], gain["best_base"][1], gain["gain"],
    )

    return {
        "confusion_matrix": cm_df,
        "per_class": per_class,
        "easiest": easiest,
        "hardest": hardest,
        "confusions": confusions,
        "ensemble_gain": gain,
        "feature_importance": importances,
    }
