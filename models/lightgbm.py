import lightgbm as lgb
from lightgbm import LGBMClassifier


class LightGBMModel:
    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        early_stopping_rounds=20,
        random_state=42,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.model = LGBMClassifier(
            n_estimators=n_estimators,      # upper limit, early stopping usually ends sooner
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            subsample=0.8,
            subsample_freq=1,               # row subsampling: faster trees
            colsample_bytree=0.8,           # feature subsampling: faster trees
            max_bin=63,                     # fewer histogram bins: faster, low accuracy cost
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """Trains on (X, y). If a validation set is given, stops early when it stops improving."""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_

    @property
    def best_iteration_(self):
        return self.model.best_iteration_
