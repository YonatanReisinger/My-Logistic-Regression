class LogisticRegression:

    def __init__(self, threshold = 0.5):
        self._weights = None
        self.fit_completed = False
        self.positive_label = 1
        self.negative_label = 0
        self.threshold = threshold


    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.fit_completed:
            pass
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def predict_proba(self, X):
        if self.fit_completed:
            pass
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def score(self, X, y):
        if self.fit_completed:
            pass
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

class LogisticRegressionMulticlass(LogisticRegression):
    pass

if __name__ == '__main__':
    pass