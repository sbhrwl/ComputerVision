import cv2
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


def data_preparation_cifar_original():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3)
    print('Train set', (X_train.shape, y_train.shape))
    print('Validation set', (X_validation.shape, y_validation.shape))
    print('Test set', (X_test.shape, y_test.shape))

    # OHE labels
    y_train = np_utils.to_categorical(y_train)
    y_validation = np_utils.to_categorical(y_validation)
    y_test = np_utils.to_categorical(y_test)
    print('Train set', (X_train.shape, y_train.shape))
    print('Validation set', (X_validation.shape, y_validation.shape))
    print('Test set', (X_test.shape, y_test.shape))

    return X_train, y_train, X_validation, y_validation, X_test, y_test


# Resizes CIFAR dataset to meet the Input dimension expected by the model using cv2
def data_preparation_cifar_resize(img_rows, img_cols):
    num_classes = 10
    # Load cifar10 training and validation sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3)
    print('Train set', (X_train.shape, y_train.shape))
    print('Validation set', (X_validation.shape, y_validation.shape))
    print('Test set', (X_test.shape, y_test.shape))

    # As Current system configuration does not support needed RAM, so take a subset of dataset
    idx_train = np.arange(len(X_train))
    idx_validation = np.arange(len(X_validation))
    idx_test = np.arange(len(X_test))
    X_train = X_train[:int(.10*len(idx_train))]
    y_train = y_train[:int(.10*len(idx_train))]
    X_validation = X_validation[:int(.10*len(idx_validation))]
    y_validation = y_validation[:int(.10*len(idx_validation))]
    X_test = X_test[:int(.10*len(idx_test))]
    y_test = y_test[:int(.10*len(idx_test))]

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]])
    X_validation = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_validation[:, :, :, :]])
    X_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_test[:, :, :, :]])
    print(X_train.shape)
    print(X_test.shape)

    # Transform targets to keras compatible format
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    X_validation = X_validation.astype('float32')
    X_test = X_test.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    X_validation = X_validation / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def data_preparation_custom():
    # Give dataset path, size of image 224x224
    train_path = 'images/train'
    test_path = 'images/validation'

    # Data Augmentation
    train_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Data Augmentation
    test_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Make sure you provide the same target size as initialized for the image size
    train_set = train_data_generator.flow_from_directory('images/train',
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')

    test_set = test_data_generator.flow_from_directory('images/validation',
                                                       target_size=(224, 224),
                                                       batch_size=32,
                                                       class_mode='categorical')
    return train_set, test_set
