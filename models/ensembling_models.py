import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def collect_probas(fitted_models, X):
    """Runs predict_proba once per base model. Share the result across all ensembles."""
    return {name: model.predict_proba(X) for name, model in fitted_models.items()}


class _BaseEnsemble:
    """Common setup: holds already-fitted base models with a consistent class order."""

    def __init__(self, fitted_models):
        self.models = dict(fitted_models)
        self.names = list(self.models)

        # Every model must have seen the same labels in the same order,
        # otherwise probability columns would not line up.
        first = next(iter(self.models.values())).classes_
        for name, model in self.models.items():
            if not np.array_equal(model.classes_, first):
                raise ValueError(f"Model '{name}' has a different class order than the others.")
        self.classes_ = first

    def predict(self, X):
        return self.predict_from_probas(collect_probas(self.models, X))


class HardVotingModel(_BaseEnsemble):
    """Each base model votes for its top class; the most votes wins."""

    def fit(self, X=None, y=None):
        return self  # nothing to learn, base models are already trained

    def predict_from_probas(self, probas):
        n_classes = len(self.classes_)
        votes = np.zeros_like(next(iter(probas.values())))
        for name in self.names:
            top = probas[name].argmax(axis=1)
            votes[np.arange(len(top)), top] += 1

        # Break ties with the average probability (mean <= 1, scaled by 0.5,
        # so it can never override a real vote difference).
        mean_proba = np.mean([probas[n] for n in self.names], axis=0)
        return self.classes_[(votes + 0.5 * mean_proba).argmax(axis=1)]


class SoftVotingModel(_BaseEnsemble):
    """Averages the predicted probabilities and takes the top class."""

    def __init__(self, fitted_models, weights=None):
        super().__init__(fitted_models)
        self.weights = weights  # optional list, same order as the models

    def fit(self, X=None, y=None):
        return self

    def predict_from_probas(self, probas):
        mean_proba = np.average([probas[n] for n in self.names], axis=0, weights=self.weights)
        return self.classes_[mean_proba.argmax(axis=1)]


class StackingModel(_BaseEnsemble):
    """Logistic-regression meta-learner trained on base-model probabilities.

    IMPORTANT: fit it on data the base models were NOT trained on (the dev set).
    Fitting on the training set would show the meta-learner overconfident
    probabilities and it would learn to over-trust the flexible models (RF).
    """

    def __init__(self, fitted_models, random_state=42):
        super().__init__(fitted_models)
        self.meta = LogisticRegression(
            class_weight="balanced", max_iter=300, random_state=random_state
        )

    def _meta_features(self, probas):
        return np.hstack([probas[n] for n in self.names])

    def fit_from_probas(self, probas, y):
        self.meta.fit(self._meta_features(probas), y)
        return self

    def fit(self, X, y):
        return self.fit_from_probas(collect_probas(self.models, X), y)

    def cross_val_predict_from_probas(self, probas, y, cv=3):
        """Honest dev-set estimate of stacking performance, without retraining base models."""
        return cross_val_predict(self.meta, self._meta_features(probas), y, cv=cv)

    def predict_from_probas(self, probas):
        return self.meta.predict(self._meta_features(probas))
