import numpy as np
from keras.utils import to_categorical
import tensorflow as tf


def problem_one(data: np.array):
    """
    How many labels are there in the training data set?
    """
    print(data.shape[1])


def problem_two():
    """
    skipped because it just completes the data set given, which has been done in the main method.
    """
    pass


def problem_3(data: np.array, true_labels: np.array):
    """
    Compute the accuracy of your model on this test set
    """
    predicted_labels = np.argmax(data, axis=1)
    num_correct_predictions = np.sum(predicted_labels == true_labels)
    accuracy = num_correct_predictions / true_labels.size * 100
    print(f'Accuracy: {accuracy: .2f}%')


def problem_4(data: np.array, true_labels: np.array):
    """
    Compute the log-likelihood of your model on this test set with the true labels in part 3
    """
    # human code
    # predicted_probs = np.amax(data, axis=1)
    # predicted_probs[3] = 0.1
    # ll = np.sum(np.log(predicted_probs))
    # print(ll)
    # chat gpt
    chat_predicted = data[np.arange(data.shape[0]), true_labels]
    chat_ll = np.sum(np.log(chat_predicted))
    print(chat_ll)


def problem_5(true_labels: np.array):
    """
    Convert the true label vector in part 3 into a one-hot encoding matrix with the correct number of labels.
    """
    num_labels = np.max(true_labels) + 1
    ohe_matrix = tf.one_hot(true_labels, depth=num_labels)
    # ohe_matrix = to_categorical(true_labels)
    print(ohe_matrix)


if __name__ == '__main__':
    data: np.array = np.array([[0.2, 0.2, 0.5, 0.1],
                               [0.4, 0.5, 0.05, 0.05],
                               [0.5, 0.1, 0.3, 0.1],
                               [0.1, 0.1, 0.1, 0.7],
                               [0.1, 0.2, 0.6, 0.1]])
    true_labels = np.array([2, 1, 0, 2, 2])
    # problem_one(data)
    # problem_3(data, true_labels)
    # problem_4(data, true_labels)
    problem_5(true_labels)
