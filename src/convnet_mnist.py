from matplotlib import pyplot as plt
from keras import layers, models
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
# from keras.optimizers import adam


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


def build_model():
    model = models.Sequential()

    # Channel dimension and Receptive field dimensions Change in opposite direction
    # Channel dimension decreases, Receptive field dimension increases

    # channel dimensions = 26x26x10 and Receptive field = 3x3
    model.add(layers.Conv2D(10, (3, 3),
                            activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # Next step Receptive field = 5x5 because result of 2 3x3 is 5x5
    # channel dimensions = 24x24x16 and Receptive field = 5x5
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # When using 1x1 size of Receptive field remains same
    # Performing 2D convolution followed Max pooling operation
    # channel dimensions = 24x24x10 and Receptive field = 5x5 as using 1x1 kernel
    model.add(layers.Conv2D(10, (1, 1),
                            activation='relu'))  # 24
    # channel dimensions = 12x12x10 and Receptive field = 10x10
    model.add(layers.MaxPooling2D((2, 2)))  # 12

    # channel dimensions = 10x10x16 and Receptive field = 12x12
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # channel dimensions = 8x8x16 and Receptive field = 14x14
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # channel dimensions = 8x8x16 and Receptive field = 16x16
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 6
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # using 4x4 kernel to see the complete image
    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model


# Creating the "scheduler" function with two arguments i.e learning rate and epoch
def scheduler(epoch, lr):
    # Learning rate = Learning rate * 1/(1 + decay * epoch)
    # here decay is 0.319 and epoch is 10.
    return round(0.003 * 1 / (1 + 0.319 * epoch), 10)


def compile_model(model):
    # compile the model
    # optimizer = adam(lr=0.003, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def start_training(model, X_train, y_train, X_test, y_test):
    # Here we are training our model using the data and
    # using batch size of 128,number of epochs are 20 and
    # using verbose=1 for printing out all the results.
    # In the callbacks parameter we are using the Learning Rate Scheduler

    model.fit(X_train, y_train,
              batch_size=128, epochs=1, verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[LearningRateScheduler(scheduler, verbose=1)])


def load_model(model):
    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')


def evaluate_model(model, X_test, y_test):
    # evaluate test accuracy
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = 100 * score[1]

    # print test accuracy
    print('Test accuracy: %.4f%%' % accuracy)
    return accuracy


def model_preparation():
    train_features, train_labels, test_features, test_labels = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)
    # load_model(model)
    model_accuracy = evaluate_model(model, test_features, test_labels)
    return model_accuracy


if __name__ == '__main__':
    accuracy_of_model = model_preparation()
    print(accuracy_of_model)
