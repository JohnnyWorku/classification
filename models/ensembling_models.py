from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


class HardVotingModel:
    def __init__(self, base_estimators):
        self.hard_voting_model = VotingClassifier (
            estimators=base_estimators,
            voting="hard",
        )
        
    def fit(self, X, y):
        self.hard_voting_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        prediction = self.hard_voting_model.predict(X)
        
        return prediction
    
    
class SoftVotingModel:
    def __init__(self, base_estimators):
        self.soft_voting_model = VotingClassifier (
            estimators=base_estimators,
            voting="soft",
        )
        
    def fit(self, X, y):
        self.soft_voting_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        prediction = self.soft_voting_model.predict(X)
        
        return prediction
    
    
class StackingModel:
    def __init__(self, base_estimators):
        meta_classifier = LogisticRegression(class_weight="balanced", random_state=42)
        
        self.stacking_model = StackingClassifier (
            estimators=base_estimators,
            final_estimator=meta_classifier,
            cv=5,
            stack_method="predict_proba",
        )
        
    def fit(self, X, y):
        self.stacking_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        prediction = self.stacking_model.predict(X)
        
        return prediction
