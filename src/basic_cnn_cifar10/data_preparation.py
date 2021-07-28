import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow import keras


def load_dataset():
    # use Keras to import pre-shuffled MNIST database
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print("The MNIST database has a training set of %d examples." % len(X_train))
    print("The MNIST database has a test set of %d examples." % len(X_test))
    return (X_train, y_train), (X_test, y_test)


def visualise_dataset(features):
    fig = plt.figure(figsize=(20, 5))
    for i in range(36):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(features[i]))


def preprocess_features(train_features, test_features):
    # rescale to have values within 0 - 1 range [0,255] --> [0,1]
    X_train = train_features.astype('float32') / 255
    X_test = test_features.astype('float32') / 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, X_test


def preprocess_labels(train_labels, test_labels):
    # one-hot encode the labels
    num_classes = len(np.unique(train_labels))
    y_train = keras.utils.to_categorical(train_labels, num_classes)
    y_test = keras.utils.to_categorical(test_labels, num_classes)
    return y_train, y_test
