import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(number):
    return 1 / (1 + np.exp(-number))


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
        # a = -np.diag(y) @ X @ self._weights
        # return np.log(1 + np.e**a).sum() / len(a)

    def cross_entropy_gradient(input_w: np.array):
        # Compute the dot product <w, xi> for all xi in X
        dot_products = np.dot(X, input_w)
        # Compute yi * <w, xi> for all i
        y_dot_products = y * dot_products
        # Compute sigmoid(yi * <w, xi>) for all i
        sigmoid_values = sigmoid(y_dot_products)
        # Computes the product yi * xi * sigmoid(yi * <w, xi>) for each i
        # results in a matrix where each row is a vector contributing to the gradient from each sample.
        gradient_contributions = y[:, np.newaxis] * X * sigmoid_values[:, np.newaxis]
        gradient = np.sum(gradient_contributions, axis=0) / X.shape[0]

        return gradient

    return cross_entropy, cross_entropy_gradient


class LogisticRegression:

    def __init__(self, fit_intercept=True, threshold_for_pos_classification=0.5):
        self._weights = None
        self.__fit_completed = False
        self.__positive_label = 1
        self.__negative_label = 0
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = None
        self.__y_train_np = None
        # Save the data as DataFrame in order to keep the columns names if given
        self.__X_train_df = None
        self.__y_train_df = None
        self.__num_of_observations = 0
        self.__fit_intercept = fit_intercept
        self.__threshold_for_pos_classification = threshold_for_pos_classification
        self.__gradient_descent_learning_rate = 500
        self.__loss_func = None
        self.__loss_func_gradient = None

    def __set_fit_params(self, X, y):
        if self.__fit_intercept:
            X = self.add_intercept(X)
        self.__num_of_observations = X.shape[0]
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = np.array(X)
        self.__y_train_np = np.array(y)
        # Save the data as DataFrame in order to keep the columns names if given
        self.__X_train_df = pd.DataFrame(X)
        self.__y_train_df = pd.Series(y)
        self.__loss_func, self.__loss_func_gradient = cross_entropy_generator(self.__X_train_np, self.__y_train_np)

    def fit(self, X, y):
        self.__set_fit_params(X, y)
        self._weights = np.zeros(self.__X_train_np.shape[1]) # init all the weights to 0

        #for l_rate in range(100,1000, 50):
            #self.__gradient_descent_learning_rate = l_rate
            #self.gradient_descent()
        self.gradient_descent()

        self.__fit_completed = True

    def gradient_descent(self, zero_threshold=0.0000001):
        num_of_itrs = 0
        while self.__is_loss_function_reached_local_min(zero_threshold) == False:
            self._weights = self._weights - self.__gradient_descent_learning_rate * self.__loss_func_gradient(self._weights)
            num_of_itrs += 1

        num_of_itrs = num_of_itrs

    def __is_loss_function_reached_local_min(self, zero_threshold=0.0000001):
        # A function reached local minimum if it's gradient is 0
        gradient_of_weights = self.__loss_func_gradient(self._weights)
        gradient_of_weights_norm = np.linalg.norm(gradient_of_weights)
        # A vector v is the 0 vector if and only if ||v|| = 0
        return gradient_of_weights_norm <= zero_threshold

    def predict(self, X):
        if self.__fit_completed:
            feature_matrix = np.array(X)
            if self.__fit_intercept:
                feature_matrix = self.add_intercept(feature_matrix)

            predictions = np.apply_along_axis(self.predict_label, axis=1, arr=feature_matrix).tolist()
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def predict_label(self, feature_vector):
        feature_vector = np.array(feature_vector)
        z = np.inner(feature_vector, self._weights)
        sigmoid_res = sigmoid(z)
        if sigmoid_res >= self.__threshold_for_pos_classification:
            return self.__positive_label
        else:
            return self.__negative_label

    def predict_proba(self, X):
        if self.__fit_completed:
            feature_matrix = np.array(X)
            if self.__fit_intercept:
                feature_matrix = self.add_intercept(feature_matrix)

            probabilities = np.apply_along_axis(self.__predict_feature_probabilities_for_each_label,
                                                axis=1, arr=feature_matrix).tolist()
            return probabilities
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def __predict_feature_probabilities_for_each_label(self, feature_vector) -> tuple:
        if isinstance(feature_vector, np.array) is False:
            feature_vector = np.array(feature_vector)
        z = np.inner(feature_vector, self._weights)
        probability_for_positive_label = sigmoid(z)
        probability_for_negative_label = 1 - probability_for_positive_label
        return probability_for_positive_label, probability_for_negative_label

    def score(self, X, y):
        if self.__fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            return num_of_correct_classifications / len(y)

        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    # ------------------- Getters -------------------

    def get_training_dataset(self, as_frame: bool = True):
        if self.__fit_completed:
            if as_frame is True:
                return self.__X_train_df
            else:
                return self.__X_train_np
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def get_training_true_labels(self, as_frame: bool = True):
        if self.__fit_completed:
            if as_frame is True:
                return self.__y_train_df
            else:
                return self.__y_train_np
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    # ------------------- Setters -------------------
    def set_threshold_for_pos_classification(self, threshold_for_pos_classification):
        self.__threshold_for_pos_classification = threshold_for_pos_classification

    # ------------------- General Functions -------------------
    def add_intercept(self, feature_matrix):
        # Add 1 to every feature vector
        if isinstance(feature_matrix, np.ndarray):
            intercept_column = np.ones((feature_matrix.shape[0], 1))
            feature_matrix = np.concatenate((feature_matrix, intercept_column), axis=1)
            return feature_matrix  # Returning the modified feature_matrix
        elif isinstance(feature_matrix, pd.DataFrame):
            feature_matrix["constant"] = 1
            return feature_matrix  # Returning the modified DataFrame

    def summary(self, X_test=None, y_test=None):
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)

        summary_str = "==============================================================================\n"
        summary_str += f"Dep. Variable:                {self.__y_train_df.name}\n"
        summary_str += "Loss Function:                Cross Entropy\n"

        if X_test is not None and y_test is not None:
            score = self.score(X_test, y_test)
        else:
            score = self.score(self.__X_train_df, self.__y_train_df)

        summary_str += f"Score:                    {score:.6f}\n"
        summary_str += f"No. Observations for train:   {self.__num_of_observations}\n"

        if X_test is not None and y_test is not None:
            summary_str += f"No. Observations for test:    {len(X_test)}\n"

        summary_str += "==============================================================================\n"
        summary_str += "Weights Found After Training: \n"
        summary_str += "{:<15} {:>10}\n".format(" ", "coef")

        X_column_names = self.__X_train_df.columns.tolist()
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
    import statsmodels.api as sm
    df = pd.read_csv("students-loan-data.csv")
    scaler = StandardScaler()
    # Fit and transform the data
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = normalized_df[["balance", "income", "student"]]
    y = df["valid"]

    model = LogisticRegression(threshold_for_pos_classification=0.1)
    model.
    #model.fit(X, y)
    #X2 = normalized_df[["balance", "income", "student"]]
    #print(model.score(X2, y))





