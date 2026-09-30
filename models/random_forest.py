from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=25, random_state=42):
        self.rf_model = RandomForestClassifier (
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=5,
                    class_weight="balanced",
                    n_jobs=4,
                    random_state=random_state,
                )
    
    
    def fit(self, X, y):
        pca = PCA(0.95)
        X_pca = pca.fit(X)
        
        self.rf_model.fit(X_pca, y)
        
        print("Random forest model trained successfully")
        
        return self
    
    
    def predict(self, X):
        prediction = self.rf_model.predict(X)
        
        return prediction
        