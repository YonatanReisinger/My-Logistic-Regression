import numpy as np
import pandas as pd
import LogisticRegression


class LogisticRegressionMulticlass:

    def __init__(self, num_of_classes, l_rate=0.1, fit_intercept=True, loss_func_name="Cross Entropy"):
        self.__num_of_classes = num_of_classes
        self.__learning_rate = l_rate
        self.__classifiers = []
        self.__fit_completed = False
        self.__fit_intercept = fit_intercept
        self.__X_train_np = None
        self.__y_train_np = None
        self.__X_train_df = None
        self.__y_train_df = None
        self.__num_samples = 0
        self.__loss_func_name = loss_func_name

    def fit(self, X, y):
        self.__set_fit_params(X, y)

        y_lists = [np.where(self.__y_train_np == i, 1, -1) for i in range(self.__num_of_classes)]
        self.__classifiers = [LogisticRegression.LogisticRegression(fit_intercept=self.__fit_intercept
                                                 , loss_func_name=self.__loss_func_name
                                                 , l_rate=self.__learning_rate)
                              for _ in range(self.__num_of_classes)]
        for i in range(self.__num_of_classes):
            self.__classifiers[i].fit(X, y_lists[i])

        self.__fit_completed = True

    def __set_fit_params(self, X, y):
        transformed_y = self.__transform_to_class_number(y)
        self.__X_train_np = np.array(X)
        self.__y_train_np = np.array(transformed_y)
        self.__X_train_df = pd.DataFrame(X)
        self.__y_train_df = pd.Series(y)
        self.__num_samples = X.shape[0]

    def __transform_to_class_number(self, y):
        unique_values = sorted(set(y))
        # Create a mapping from value to class number (0 to n-1)
        value_to_class_number = {value: class_number for class_number, value in enumerate(unique_values)}
        transformed_y = [value_to_class_number[value] for value in y]

        return transformed_y

    def predict(self, X):
        if self.__fit_completed:
            feature_matrix = np.array(X)
            predictions = []
            num_samples = X.shape[0]

            if self.__fit_intercept:
                feature_matrix = np.array(LogisticRegression.LogisticRegression.add_intercept(X))

            for i in range(num_samples):
                probas_for_each_class = [self.__classifiers[j].predict_single_vector_pos_label_prob(feature_matrix[i])
                                         for j in range(self.__num_of_classes)]
                # Predict the vector as the label that got the highest probability
                predictions.append(probas_for_each_class.index(max(probas_for_each_class)))

            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def score(self, X, y):
        if self.__fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            return num_of_correct_classifications / len(y)

        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def summary(self, X_test=None, y_test=None):
        summary_str = "==============================================================================\n"
        summary_str += f"Dep. Variable:                {self.__y_train_df.name}\n"
        summary_str += f"Number Of Classes:            {self.__num_of_classes}\n"
        summary_str += f"Loss Function:                {self.__loss_func_name}\n"

        if X_test is not None and y_test is not None:
            score = self.score(X_test, y_test)
        else:
            score = self.score(self.__X_train_np, self.__y_train_np)

        summary_str += f"Score:                    {score:.6f}\n"
        summary_str += f"No. Observations for train:   {self.__num_samples}\n"

        if X_test is not None and y_test is not None:
            summary_str += f"No. Observations for test:    {len(X_test)}\n"

        summary_str += "==============================================================================\n"

        return summary_str
