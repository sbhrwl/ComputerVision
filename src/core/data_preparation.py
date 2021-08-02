import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


def data_preparation_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print(X_train.shape)
    # plotting the first image or the image at index zero in the training dataset
    plt.imshow(X_train[0])

    # Reshaping our training and testing dataset using numpy's reshape function which we will feed to the model
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Doing type conversion or changing the datatype to float32 for the data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Doing standardization or normalization here dividing each pixel by 255 in the train and test data
    X_train /= 255
    X_test /= 255

    # Checking first 10 image labels
    # print(y_train[:10])

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    # simply we can say we are doing sort of one hot encoding
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    # having a look in the first 10 data points after one hot encoding
    print(Y_train[:10])

    return X_train, Y_train, X_test, Y_test


def data_preparation_cifar_original():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3)
    print('Train set', (X_train.shape, y_train.shape))
    print('Validation set', (X_validation.shape, y_validation.shape))
    print('Test set', (X_test.shape, y_test.shape))

    X_train = X_train / 255.0
    X_validation = X_validation / 255.0
    X_test = X_test / 255.0

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

    # Preprocess training data
    X_train = X_train / 255.0
    X_validation = X_validation / 255.0
    X_test = X_test / 255.0

    # Transform targets to keras compatible format
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    X_validation = X_validation.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def data_preparation_cifar100_eraser():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.3)
    print('Train set', (X_train.shape, y_train.shape))
    print('Validation set', (X_validation.shape, y_validation.shape))
    print('Test set', (X_test.shape, y_test.shape))

    num_classes = 100
    # Pre-process the data
    X_train = preprocess_input(X_train)
    X_validation = preprocess_input(X_validation)
    X_test = preprocess_input(X_test)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
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
