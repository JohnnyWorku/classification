from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


class SVMModel:
    def __init__(self, kernel="rbf", C=1.0, random_state=42):
        self.base_svm = LinearSVC (
            C=C,
            dual="auto",
            max_iter=2000,
            random_state=random_state,
        )
        
        self.svm_mode = CalibratedClassifierCV (
            estimator=self.base_svm,
            ensemble=False,
            n_jobs=-1,
        )
        
        
    def fit(self, X, y):
        self.svm_model.fit(X, y)
        
        print("SVM model trained successfully!")
        
        return self
    
    
    def predict(self, X):
        prediction = self.svm_model.predict(X)
        
        return prediction
