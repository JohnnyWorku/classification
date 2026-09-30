from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(
        self,
        n_estimators=100,
        max_depth=25,
        min_samples_split=5,
        max_samples=0.5,
        random_state=42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_samples=max_samples,             # each tree sees 50% of the rows: faster, more diverse trees
            max_features="sqrt",                 # explicit (this is the default for classifiers)
            class_weight="balanced_subsample",   # rebalances inside each tree's bootstrap sample
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_
