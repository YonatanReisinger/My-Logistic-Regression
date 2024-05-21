import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re # Used to remove punctuations from the words
from collections import Counter  # Used to count the number of time a word appears in text
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import datasets


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

    def cross_entropy_gradient(input_w: np.array):
        a = -y * sigmoid(-y*(X@input_w))
        return np.dot(a, X)

    return cross_entropy, cross_entropy_gradient


class LogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.1, loss_func_name="Cross Entropy",
                 threshold_for_pos_classification=0.5, max_iter=10_000):
        self._weights = None
        self.__fit_completed = False
        # The model expect that the true labels will be 1 and -1
        self.__positive_label = 1
        self.__negative_label = -1
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = None
        self.__y_train_np = None
        # Save the data as DataFrame in order to keep the columns names if given
        self.__X_train_df = None
        self.__y_train_df = None
        self.__num_samples = 0
        self.__fit_intercept = fit_intercept
        self.__threshold_for_pos_classification = threshold_for_pos_classification
        self.__gradient_descent_learning_rate = l_rate
        self.__max_iter = max_iter
        self.__loss_func_name = loss_func_name
        self.__loss_func = None
        self.__loss_func_gradient = None

    def __set_fit_params(self, X, y):
        unique_labels = np.unique(y)

        if len(unique_labels) != 2:
            raise RuntimeError("Model is binary. can train just on classifications with just 2 classes")

        if self.__fit_intercept:
            X = self.add_intercept(X)

        self.__original_labels = unique_labels
        # The model expect that the true labels will be 1 and -1
        y = np.where(y == unique_labels[0], self.__negative_label, self.__positive_label)
        self.__num_samples = X.shape[0]
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = np.array(X)
        self.__y_train_np = np.array(y)
        # Save the data as DataFrame in order to keep the columns names if given
        self.__X_train_df = pd.DataFrame(X)
        self.__y_train_df = pd.Series(y)
        if self.__loss_func_name == "Cross Entropy":
            self.__loss_func, self.__loss_func_gradient = cross_entropy_generator(self.__X_train_np, self.__y_train_np)

    def fit(self, X, y):
        self.__set_fit_params(X, y)
        self._weights = np.zeros(self.__X_train_np.shape[1])  # init all the weights to 0
        self.__gradient_descent()
        self.__fit_completed = True

    def __gradient_descent(self, zero_threshold=0.00001):
        num_of_itrs = 0
        while self.__is_loss_function_reached_local_min(zero_threshold) == False and num_of_itrs < self.__max_iter:
            gradient = self.__loss_func_gradient(self._weights)
            self._weights -= self.__gradient_descent_learning_rate * gradient
            num_of_itrs += 1

    def __is_loss_function_reached_local_min(self, zero_threshold=0.00001):
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
        pos_label_prob = self.predict_single_vector_pos_label_prob(feature_vector)
        if pos_label_prob >= self.__threshold_for_pos_classification:
            return self.__original_labels[1]
        else:
            return self.__original_labels[0]

    def predict_single_vector_pos_label_prob(self, feature_vector):
        if self.__fit_completed:
            if type(feature_vector) is not np.array:
                feature_vector = np.array(feature_vector)
            feature_vector = np.array(feature_vector)
            z = np.inner(feature_vector, self._weights)
            return sigmoid(z)
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

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
        probability_for_positive_label = self.predict_single_vector_pos_label_prob(feature_vector)
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

    @staticmethod
    def add_intercept(feature_matrix):
        # Add 1 to every feature vector
        if isinstance(feature_matrix, np.ndarray):
            intercept_column = np.ones((feature_matrix.shape[0], 1))
            feature_matrix_with_constant = np.concatenate((feature_matrix, intercept_column), axis=1)
            return np.array(feature_matrix_with_constant)  # Returning the modified feature_matrix
        elif isinstance(feature_matrix, pd.DataFrame):
            feature_matrix_with_constant = feature_matrix.copy()
            feature_matrix_with_constant["constant"] = 1
            return pd.DataFrame(feature_matrix_with_constant) # Returning the modified DataFrame

    def plot_ROC(self, X, y):
        thresholds = np.linspace(0, 1, 100)
        true_positive_rate_lst = []
        false_positive_rate_lst = []
        j_statistic_list = []
        original_threshold = self.__threshold_for_pos_classification
        pos_label = self.__original_labels[1]
        neg_label = self.__original_labels[0]

        for threshold in thresholds:
            self.set_threshold_for_pos_classification(threshold)
            predictions = np.array(self.predict(X))
            true_positive_counter = np.sum((predictions == pos_label) & (y == pos_label))
            false_positive_counter = np.sum((predictions == pos_label) & (y == neg_label))
            false_negative_counter = np.sum((predictions == neg_label) & (y == pos_label))
            true_negative_counter = np.sum((predictions == neg_label) & (y == neg_label))

            true_positive_rate = true_positive_counter / (true_positive_counter + false_negative_counter)
            false_positive_rate = false_positive_counter / (false_positive_counter + true_negative_counter)

            true_positive_rate_lst.append(true_positive_rate)
            false_positive_rate_lst.append(false_positive_rate)
            j_statistic_list.append(true_positive_rate - false_positive_rate)

        # Return the original_threshold to the model
        self.set_threshold_for_pos_classification(original_threshold)
        # Draw ROC/AUC curve
        plt.figure()
        plt.plot(false_positive_rate_lst, true_positive_rate_lst, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        self.__show_best_thershold_on_ROC(thresholds, j_statistic_list,
                                        true_positive_rate_lst, false_positive_rate_lst)

        plt.show()

    def __show_best_thershold_on_ROC(self, thresholds, j_statistic_list, tpr_lst, fpr_lst):
        # Annotate the best threshold point with coordinates
        best_threshold_index = np.argmax(j_statistic_list)
        best_threshold = thresholds[best_threshold_index]
        best_tpr = tpr_lst[best_threshold_index]
        best_fpr = fpr_lst[best_threshold_index]
        # Annotate the best threshold point with coordinates
        plt.annotate(f'Threshold = {best_threshold:.2f}\n(TPR = {best_tpr:.2f}, FPR = {best_fpr:.2f})',
                     xy=(best_fpr, best_tpr), xycoords='data',
                     xytext=(-40, 20), textcoords='offset points',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                     fontsize=8,
                     horizontalalignment='right', verticalalignment='bottom')

    def summary(self, X_test=None, y_test=None):
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)

        summary_str = "==============================================================================\n"
        summary_str += f"Dep. Variable:                {self.__y_train_df.name}\n"
        summary_str += f"Loss Function:                {self.__loss_func_name}\n"

        if X_test is not None and y_test is not None:
            score = self.score(X_test, y_test)
        else:
            score = self.score(self.__X_train_df, self.__y_train_df)

        summary_str += f"Score:                    {score:.6f}\n"
        summary_str += f"No. Observations for train:   {self.__num_samples}\n"

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
        self.__classifiers = [LogisticRegression(fit_intercept=self.__fit_intercept
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
                feature_matrix = np.array(LogisticRegression.add_intercept(X))

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

def convert_texts_in_csv_to_bag_of_words_vectors(file_path: str, text_coulmn_name: str) -> np.array:
    df = pd.read_csv(file_path)
    # Assuming the text data is in a column named 'text'
    texts = df[text_coulmn_name].tolist()
    tokenized_texts = [preprocess_text(text) for text in texts]
    all_words = [word for text in tokenized_texts for word in text]
    all_unique_words = list(set(all_words))

    # Create a mapping from word to index
    word_to_index = {word: i for i, word in enumerate(all_unique_words)}
    # Create Bag of Words representation for all texts in the file
    bow_vectors = np.zeros((len(tokenized_texts), len(all_unique_words)), dtype=int)

    for i, text in enumerate(tokenized_texts):
        word_counts = Counter(text)
        # Each index, the BOW vector will have the amount of times that words appeared in it's text.
        for word, count in word_counts.items():
            if word in word_to_index:
                bow_vectors[i, word_to_index[word]] = count

    return bow_vectors

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    words = text.split()
    return words

def main1():
    dimension = 100
    true_labels = np.array(pd.read_csv("spam_ham_dataset.csv")["label_num"])
    # true_labels = np.where(true_labels == 0, -1, 1)
    bag_of_words_vectors = convert_texts_in_csv_to_bag_of_words_vectors(
        "spam_ham_dataset.csv", text_coulmn_name="text")
    # Reduce the vectors to lower dimension
    pca = PCA(n_components=dimension)
    principal_components = pca.fit_transform(bag_of_words_vectors)
    bag_of_words_vectors_n_dimension = pd.DataFrame(data=principal_components)

    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(bag_of_words_vectors_n_dimension, true_labels, test_size=0.2, shuffle=True))

    model = LogisticRegression(l_rate=0.0001, threshold_for_pos_classification=0.3)
    model.fit(feature_matrix_train, true_labels_train)
    Question2(model, feature_matrix_test, true_labels_test)
    Question3(model, feature_matrix_test, true_labels_test)


def Question2(model, feature_matrix_test, true_labels_test):
    print(model.summary(feature_matrix_test, true_labels_test))


def Question3(model, feature_matrix_test, true_labels_test):
    model.plot_ROC(feature_matrix_test, true_labels_test)
    print("The best Threshold that can be used is the one that gets as much True positives "
          "and as less False positive"
          ", thus the threshold point on the most left upper side of the ROC curve")


def main2():
    feature_matrix, true_labels = datasets.load_iris(return_X_y=True, as_frame=True)
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))

    multi_model = LogisticRegressionMulticlass(len(set(true_labels)), l_rate=0.001)
    multi_model.fit(feature_matrix_train, true_labels_train)
    print(multi_model.summary(feature_matrix_test, true_labels_test))


if __name__ == '__main__':
    main1()
    #main2()

