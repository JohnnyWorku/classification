from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

class SVMModel:
    def __init__(self, C=1.0, random_state=42):
        self.base_svm = LinearSVC (
            C=C,
            dual="auto",
            max_iter=2000,
            random_state=random_state,
        )
        
        self.svm_model = CalibratedClassifierCV (
            estimator=self.base_svm,
            ensemble=False,
            n_jobs=2,
        )
        
        
    def fit(self, X, y):
        pca = PCA(0.95)
        X_pca = pca.fit(X)
        
        self.svm_model.fit(X_pca, y)
        
        print("SVM model trained successfully!")
        
        return self
    
    
    def predict(self, X):
        prediction = self.svm_model.predict(X)
        
        return prediction
