from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils


def data_preparation():
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
