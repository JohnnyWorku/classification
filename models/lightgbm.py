from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA


class LightGBMModel:
    def __init__(self, n_estimators=100, learning_rate=0.05, random_state=42):
        self.lightgbm_model = LGBMClassifier (
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            class_weight="balanced",
            n_jobs=4,
            random_state=random_state,
            verbose=-1
        )
        
        
    def fit(self, X, y):
        pca = PCA(0.95)
        X_pca = pca.fit(X)
        
        self.lightgbm_model.fit(X_pca, y)
        
        print("LightGBM model trained successfully!")
        
        return self
    
    
    def predict(self, X):
        prediction = self.lightgbm_model.predict(X)
        
        return prediction
