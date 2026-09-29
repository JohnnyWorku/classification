from sklearn.svm import SVC


class SVMModel:
    def __init__(self, kernel="rbf", C=1.0, random_state=42):
        self.svm_model = SVC (
            kernel=kernel,
            C=C,
            probability=True,   # probability=True is REQUIRED for Soft Voting and Stacking Probabilities
            class_weight="balanced",
            random_state=random_state,
        )
        
        
    def fit(self, X, y):
        self.svm_model.fit(X, y)
        
        print("SVM model trained successfully!")
        
        return self
    
    
    def predict(self, X):
        prediction = self.svm_model.predict(X)
        
        return prediction
