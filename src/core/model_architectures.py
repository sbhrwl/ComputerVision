from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D


def model_architecture_1conv_1max_pool():
    # build the model object
    model = Sequential()

    # CONV_1: add CONV layer with
    # RELU activation and depth = 32 of kernels (Feature extractor), each kernel of size 3*3
    # padding as same means we the output image size of this layer will be same as Input layer
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # Output: 28 * 28 * 32

    # POOL_1: down sample the image to choose the best features
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 14 * 14 * 32

    # CONV_2: here we increase the depth to 64
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # Output: 14 * 14 * 64

    # POOL_2: more down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 7 * 7 * 64

    # flatten since too many dimensions, we only want a classification output
    # Flatten layer is introduced after we are done with the Feature extraction
    model.add(Flatten())

    # FC_1: fully connected to get all relevant data
    model.add(Dense(64, activation='relu'))

    # FC_2: output a softmax to squash the matrix into output probabilities for the 10 classes
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model


def model_architecture_2conv_1max_pool():
    # build the model object
    model = Sequential()

    # CONV_1: add CONV layer with
    # RELU activation and depth = 32 of kernels (Feature extractor), each kernel of size 3*3
    # padding as same means we the output image size of this layer will be same as Input layer
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # Output: 28 * 28 * 32

    # POOL_1: down sample the image to choose the best features
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 14 * 14 * 32

    # CONV_2: here we increase the depth to 64
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # Output: 14 * 14 * 64

    # POOL_2: more down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 7 * 7 * 64

    # flatten since too many dimensions, we only want a classification output
    # Flatten layer is introduced after we are done with the Feature extraction
    model.add(Flatten())

    # FC_1: fully connected to get all relevant data
    model.add(Dense(64, activation='relu'))

    # FC_2: output a softmax to squash the matrix into output probabilities for the 10 classes
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model


def model_architecture_3conv_1max_pool():
    # build the model object
    model = Sequential()

    # CONV_1: add CONV layer with
    # RELU activation and depth = 32 of kernels (Feature extractor), each kernel of size 3*3
    # padding as same means we the output image size of this layer will be same as Input layer
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    # Output: 28 * 28 * 32

    # POOL_1: down sample the image to choose the best features
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 14 * 14 * 32

    # CONV_2: here we increase the depth to 64
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # Output: 14 * 14 * 64

    # POOL_2: more down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output: 7 * 7 * 64

    # flatten since too many dimensions, we only want a classification output
    # Flatten layer is introduced after we are done with the Feature extraction
    model.add(Flatten())

    # FC_1: fully connected to get all relevant data
    model.add(Dense(64, activation='relu'))

    # FC_2: output a softmax to squash the matrix into output probabilities for the 10 classes
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model
