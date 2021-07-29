import cv2
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils


def data_preparation(img_rows, img_cols):
    num_classes = 10
    # Load cifar10 training and validation sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape)
    print(X_test.shape)

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]])
    X_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_test[:, :, :, :]])
    print(X_train.shape)
    print(X_test.shape)

    # Transform targets to keras compatible format
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test


def data_preparation_transfer_learning():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print(X_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, y_test
