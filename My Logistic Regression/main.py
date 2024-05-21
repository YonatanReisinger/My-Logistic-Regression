import numpy as np
import pandas as pd
import LogisticRegression
import LogisticRegressionMulticlass
import re # Used to remove punctuations from the words
from collections import Counter  # Used to count the number of time a word appears in text
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import datasets


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
    bag_of_words_vectors = convert_texts_in_csv_to_bag_of_words_vectors(
        "spam_ham_dataset.csv", text_coulmn_name="text")
    # Reduce the vectors to lower dimension
    pca = PCA(n_components=dimension)
    principal_components = pca.fit_transform(bag_of_words_vectors)
    bag_of_words_vectors_n_dimension = pd.DataFrame(data=principal_components)

    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(bag_of_words_vectors_n_dimension, true_labels, test_size=0.2, shuffle=True))

    model = LogisticRegression.LogisticRegression(l_rate=0.0001, threshold_for_pos_classification=0.3)
    model.fit(feature_matrix_train, true_labels_train)
    Question2(model, feature_matrix_test, true_labels_test)
    Question3(model, feature_matrix_test, true_labels_test)


def Question2(model, feature_matrix_test, true_labels_test):
    print("\n\n---------- This is Question 2 ----------\n\n")
    print(model.summary(feature_matrix_test, true_labels_test))


def Question3(model, feature_matrix_test, true_labels_test):
    print("\n\n---------- This is Question 3 ----------\n\n")
    model.plot_ROC(feature_matrix_test, true_labels_test)
    print("The best Threshold that can be used is the one that gets as much True positives "
          "and as less False positive"
          ", thus the threshold point on the most left upper side of the ROC curve")


def main2():
    print("\n\n---------- This is Question 4 ----------\n\n")
    feature_matrix, true_labels = datasets.load_iris(return_X_y=True, as_frame=True)
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))

    multi_model = LogisticRegressionMulticlass.LogisticRegressionMulticlass(len(set(true_labels)), l_rate=0.001)
    multi_model.fit(feature_matrix_train, true_labels_train)
    print(multi_model.summary(feature_matrix_test, true_labels_test))


if __name__ == '__main__':
    main1()
    main2()

