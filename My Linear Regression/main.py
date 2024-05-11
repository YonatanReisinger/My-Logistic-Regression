import numpy as np
import pandas as pd
class LogisticRegression:

    def __init__(self, fit_intercept=True, threshold=0.5):
        self._weights = None
        self.fit_completed = False
        self.positive_label = 1
        self.negative_label = 0
        self.fit_intercept = True
        self.threshold = threshold


    def fit(self, X, y):
        if len(set(y)) != 2:
            raise RuntimeError("This Logistic Regression model can only do binary classification")

        self.positive_label = list(set(y))[0]
        self.negative_label = list(set(y))[1]



    def add_intercept(self, feature_matrix):
        # Add 1 to every feature vector
        if isinstance(feature_matrix, np.ndarray):
            intercept_column = np.ones((feature_matrix.shape[0], 1))
            feature_matrix = np.concatenate((feature_matrix, intercept_column), axis=1)
            return feature_matrix  # Returning the modified feature_matrix
        elif isinstance(feature_matrix, pd.DataFrame):
            feature_matrix["constant"] = 1
            return feature_matrix  # Returning the modified DataFrame

    def predict_label(self, feature_vector):
        sigmoid_res = self.sigmoid(feature_vector)
        if sigmoid_res >= self.threshold:
            return self.positive_label
        else:
            return self.negative_label

    def predict(self, X):
        if self.fit_completed:
            feature_matrix = np.array(X)
            if self.fit_intercept:
                feature_matrix = self.add_intercept(feature_matrix)

            predictions = np.apply_along_axis(self.predict_label, axis=1, arr=feature_matrix).tolist()
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def predict_labels_probability(self, feature_vector) -> tuple:
        probability_for_positive_label = self.sigmoid(feature_vector)
        probability_for_negative_label = 1 - probability_for_positive_label
        return probability_for_positive_label, probability_for_negative_label

    def sigmoid(self, feature_vector):
        try:
            feature_vector = np.array(feature_vector)
            z = np.inner(feature_vector, self._weights)
            return 1 / (1 + np.exp(-z))
        except:
            raise RuntimeError("All feature vectors must have numerical values only")


    def predict_proba(self, X):
        if self.fit_completed:
            feature_matrix = np.array(X)
            if self.fit_intercept:
                feature_matrix = self.add_intercept(feature_matrix)

            probabilities = np.apply_along_axis(self.predict_labels_probability, axis=1, arr=feature_matrix).tolist()
            return probabilities
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def score(self, X, y):
        if self.fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            return len(y) / num_of_correct_classifications

        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

class LogisticRegressionMulticlass(LogisticRegression):
    pass

if __name__ == '__main__':
    pass
