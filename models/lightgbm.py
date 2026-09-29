from lightgbm import LGBMClassifier


class LightGBMModel:
    def __init__(self, n_estimators=100, learning_rate=0.05, random_state=42):
        self.lightgbm_model = LGBMClassifier (
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        
        
    def fit(self, X, y):
        self.lightgbm_model.fit(X, y)
        
        print("LightGBM model trained successfully!")
        
        return self
    
    
    def predict(self, X):
        prediction = self.lightgbm_model.predict(X)
        
        return prediction
