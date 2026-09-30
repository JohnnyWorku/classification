import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
except ImportError:
    FrozenEstimator = None


class SVMModel:
    def __init__(self, C=0.1, max_iter=2000, random_state=42):
        # Scaling speeds up convergence a lot: k-mer frequencies are tiny and on different scales.
        self.model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=C,
                class_weight="balanced",
                dual="auto",
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        self.calibrated_ = None  # set only if calibrate() is called

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        """Optional: sigmoid calibration on held-out data (e.g. dev).
        Fits only the small calibration layer; the SVM is NOT retrained."""
        if FrozenEstimator is None:
            raise RuntimeError("calibrate() needs scikit-learn >= 1.6; use the default softmax scores instead.")
        self.calibrated_ = CalibratedClassifierCV(FrozenEstimator(self.model), method="sigmoid")
        self.calibrated_.fit(X_cal, y_cal)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.calibrated_ is not None:
            return self.calibrated_.predict_proba(X)

        # Default: softmax of the decision scores. Not calibrated, but free to compute
        # and fine as input to soft voting and stacking.
        scores = self.model.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    @property
    def classes_(self):
        return self.model.classes_
