from pathlib import Path

import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Basic metrics (same names as before)
# ---------------------------------------------------------------------------


def accuracy_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision_metric(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def recall_metric(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def f1(y_true, y_pred, pos_label="EC:1"):
    """F1 for one specific class (one-vs-rest). The old version failed on multi-class data."""
    return f1_score(y_true, y_pred, labels=[pos_label], average=None, zero_division=0)[0]


def macro_f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def mattews_corrcoef_metric(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


# ---------------------------------------------------------------------------
# Probability-based metrics (need y_proba with columns in `classes` order)
# ---------------------------------------------------------------------------


def _binarize(y_true, classes):
    Y = label_binarize(y_true, classes=classes)
    if len(classes) == 2:  # label_binarize returns a single column for 2 classes
        Y = np.hstack([1 - Y, Y])
    return Y


def auprc_metric(y_true, y_proba, classes):
    """Macro-averaged area under the precision-recall curve (one-vs-rest)."""
    if y_proba is None:
        return np.nan
    Y = _binarize(y_true, classes)
    scores = [
        average_precision_score(Y[:, i], y_proba[:, i])
        for i in range(len(classes))
        if Y[:, i].sum() > 0  # skip classes absent from y_true
    ]
    return float(np.mean(scores)) if scores else np.nan


def roc_auc_metric(y_true, y_proba, classes):
    """Macro-averaged one-vs-rest ROC-AUC."""
    if y_proba is None:
        return np.nan
    Y = _binarize(y_true, classes)
    scores = [
        roc_auc_score(Y[:, i], y_proba[:, i])
        for i in range(len(classes))
        if 0 < Y[:, i].sum() < len(Y)  # need both positives and negatives
    ]
    return float(np.mean(scores)) if scores else np.nan


# ---------------------------------------------------------------------------
# Summary table across models (feeds the "did ensembles improve?" question)
# ---------------------------------------------------------------------------


def evaluate_all(y_true, predictions, classes):
    """predictions: {model_name: (y_pred, y_proba_or_None)}. Returns one row per model."""
    rows = []
    for name, (y_pred, y_proba) in predictions.items():
        rows.append(
            {
                "Model": name,
                "Macro F1": macro_f1_metric(y_true, y_pred),  # primary metric
                "MCC": mattews_corrcoef_metric(y_true, y_pred),
                "AUPRC (macro)": auprc_metric(y_true, y_proba, classes),
                "ROC-AUC (macro)": roc_auc_metric(y_true, y_proba, classes),
                "Accuracy": accuracy_metric(y_true, y_pred),  # reference only if imbalanced
                "Precision (Macro)": precision_metric(y_true, y_pred),
                "Recall (Macro)": recall_metric(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-class metrics and confusion matrix
# ---------------------------------------------------------------------------


def per_class_metrics_table(y_true, y_pred, classes):
    """Class / Precision / Recall / F1 (+ support) table."""
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), zero_division=0
    )
    return pd.DataFrame(
        {"Class": classes, "Precision": p, "Recall": r, "F1": f, "Support": support}
    )


def confusion_matrix_normalized(y_true, y_pred, classes):
    """Rows = actual class, columns = predicted class, each row sums to 1 (normalized by true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes), normalize="true")
    return pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                        columns=[f"Pred {c}" for c in classes])


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion matrix (normalized by true class)",
                          save_path=None, show=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(classes), normalize="true")
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 3, 1.0 * len(classes) + 2.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# ROC curves (one-vs-rest)
# ---------------------------------------------------------------------------


def plot_roc_curves(y_true, y_proba, classes, title="One-vs-rest ROC curves",
                    save_path=None, show=False):
    Y = _binarize(y_true, classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = {}
    for i, cls in enumerate(classes):
        if not (0 < Y[:, i].sum() < len(Y)):
            continue
        fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
        aucs[cls] = roc_auc_score(Y[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {aucs[cls]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}\nmacro AUC = {np.mean(list(aucs.values())):.3f}")
    ax.legend(loc="lower right", fontsize=8)
    _finish(fig, save_path, show)
    return pd.Series(aucs, name="ROC-AUC")


# ---------------------------------------------------------------------------
# Feature importance (Random Forest or LightGBM)
# ---------------------------------------------------------------------------


def get_feature_importance(model, feature_names=None, top_n=20):
    """model: fitted RandomForestModel / LightGBMModel wrapper, or a raw sklearn/LightGBM estimator."""
    estimator = getattr(model, "model", model)
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    total = importances.sum()
    if total > 0:
        importances = importances / total  # relative importance, sums to 1
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(importance_df, title="Top feature importances", save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(importance_df) + 1.5))
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
    ax.set_xlabel("Relative importance")
    ax.set_title(title)
    return _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------------


def most_confused_pairs(y_true, y_pred, classes, top_n=5):
    """Most frequent (actual -> predicted) mistakes, as a count and as a share of the actual class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    row_totals = cm.sum(axis=1)
    rows = []
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": int(cm[i, j]),
                    "Share of actual class": cm[i, j] / row_totals[i],
                })
    df = pd.DataFrame(rows, columns=["Actual", "Predicted", "Count", "Share of actual class"])
    return df.sort_values("Count", ascending=False).head(top_n).reset_index(drop=True)


def easiest_and_hardest_classes(y_true, y_pred, classes, n=2):
    """Classes ranked by F1: returns (easiest, hardest) DataFrames."""
    table = per_class_metrics_table(y_true, y_pred, classes).sort_values("F1", ascending=False)
    return table.head(n).reset_index(drop=True), table.tail(n).iloc[::-1].reset_index(drop=True)


def ensemble_gain(results_df, base_names, ensemble_names, metric="Macro F1"):
    """Best ensemble vs best single model on one metric (answers 'did ensembles improve?')."""
    scores = results_df.set_index("Model")[metric]
    best_base = scores[base_names].idxmax()
    best_ens = scores[ensemble_names].idxmax()
    return {
        "best_base": (best_base, float(scores[best_base])),
        "best_ensemble": (best_ens, float(scores[best_ens])),
        "gain": float(scores[best_ens] - scores[best_base]),
    }


# ---------------------------------------------------------------------------


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return save_path
