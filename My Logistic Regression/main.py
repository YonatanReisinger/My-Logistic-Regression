import numpy as np
import pandas as pd

def cross_entropy_generator(X, y):
    def cross_entropy(input_w: np.array):
        result = 0
        input_w_transpose = input_w.transopse()
        num_of_observations = X.shape[0]
        for feature_vector, true_label in zip(X, y):
            a = np.dot(-1 * true_label * input_w_transpose, feature_vector)
            result += np.log(1 + np.e ** a)
        result /= num_of_observations
        return result

    def cross_entropy_gradient(input_w: np.array):
        pass

    return cross_entropy, cross_entropy_gradient

class LogisticRegression:

    def __init__(self, fit_intercept=True, threshold_for_pos_classification=0.5):
        self._weights = None
        self.fit_completed = False
        self.positive_label = 1
        self.negative_label = 0
        # Save the data as numpy matrix in order to make calculations
        self.X_train_np = None
        self.y_train_np = None
        # Save the data as DataFrame in order to keep the columns names if given
        self.X_train_df = None
        self.y_train_df = None
        self.num_of_observations = 0
        self.fit_intercept = fit_intercept
        self.threshold_for_pos_classification = threshold_for_pos_classification
        self.gradient_descent_learning_rate = 0.001
        self.loss_func = None
        self.loss_func_gradient = None

    def set_fit_params(self, X, y):
        self.positive_label = list(set(y))[0]
        self.negative_label = list(set(y))[1]
        self.num_of_observations = X.shape[0]
        # Save the data as numpy matrix in order to make calculations
        self.X_train_np = np.array(X)
        self.y_train_np = np.array(y)
        # Save the data as DataFrame in order to keep the columns names if given
        self.X_train_df = pd.DataFrame(X)
        self.y_train_df = pd.Series(y)
        self.loss_func, self.loss_func_gradient = cross_entropy_generator(X, y)

    def fit(self, X, y):
        self.set_fit_params(X, y)
        if self.fit_intercept:
            self.X_train_np = self.add_intercept(self.X_train_np)
        self._weights = np.zeros(len(self.X_train_np[0])) # init all the weights to 0

        self.gradient_descent()

        self.fit_completed = True

    def gradient_descent(self, zero_threshold = 0.0001):
        # A vector v is the 0 vector if and only if ||v|| = 0
        while np.linalg.norm(self.loss_func_gradient(self._weights)) > zero_threshold:
            self._weights = self._weights - self.gradient_descent_learning_rate * self.loss_func(self._weights)

    def cross_entropy(self, X, y):
        result = 0
        weights_transpose = self._weights.transopse()
        for feature_vector, true_label in zip(X, y):
            a = np.dot(-1 * true_label * weights_transpose, feature_vector)
            result += np.log(1 + np.e**a)
        result /= self.num_of_observations

        return result

        #a = -np.diag(y) @ X @ self._weights
        #return np.log(1 + np.e**a).sum() / len(a)

    def cross_entropy_gradient(self, X, y):
        pass
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
        if sigmoid_res >= self.threshold_for_pos_classification:
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
            return self.logistic_function(z)
        except:
            raise RuntimeError("All feature vectors must have numerical values only")

    def logistic_function(self, x):
        return 1 / (1 + np.exp(-x))

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

    def summary(self, X_test = None, y_test = None):
        self.X_train = pd.DataFrame(self.X_train)
        self.y_train = pd.Series(self.y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)

        summary_str = "==============================================================================\n"
        summary_str += f"Dep. Variable:                {self.y_train.name}\n"
        summary_str += "Loss Function:                MSE\n"

        if X_test is not None and y_test is not None:
            r_squared = self.score(X_test, y_test)
            adj_r_squared = self.adjusted_score(X_test, y_test)
        else:
            r_squared = self.score(self.X_train, self.y_train)
            adj_r_squared = self.adjusted_score(self.X_train, self.y_train)

        summary_str += f"R-squared:                    {r_squared:.6f}\n"
        summary_str += f"Adj. R-squared:               {adj_r_squared:.6f}\n"
        summary_str += f"No. Observations for train:   {len(self.X_train)}\n"

        if X_test is not None and y_test is not None:
            summary_str += f"No. Observations for test:    {len(X_test)}\n"

        summary_str += "==============================================================================\n"
        summary_str += "Weights Found After Training: \n"
        summary_str += "{:<15} {:>10}\n".format(" ", "coef")

        X_column_names = self.X_train.columns.tolist()
        for i in range(len(self._weights)):
            summary_str += "{:<15} {:10.6f}\n".format(
                X_column_names[i] if i < len(X_column_names) else "const",
                self._weights[i]
            )

        summary_str += "==============================================================================\n"

        return summary_str

class LogisticRegressionMulticlass(LogisticRegression):
    pass

def your_function(row_X, number_y):
    # Your function logic here
    print("Row:", row_X, "Number:", number_y)

if __name__ == '__main__':
    #import statsmodels.api as sm
    df = pd.read_csv("students-loan-data.csv")
    #X = sm.add_constant(df[["balance", "income", "student"]])
    #y = df["valid"]

    #model = sm.Logit(y, X)
    #res = model.fit()

    #print(res.summary())


