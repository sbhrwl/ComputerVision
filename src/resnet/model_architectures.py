from keras import Sequential
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, Dense, Dropout


def resnet_transfer_learning(y_train):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=y_train.shape[1])
    model = Sequential()
    # Add the Dense layers along with activation and batch normalization
    model.add(base_model)
    model.add(Flatten())

    # Add the Dense layers along with activation and batch normalization
    model.add(Dense(4000, activation='relu', input_dim=512))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(.3))  # Adding a dropout layer that will randomly drop 30% of the weights
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(10, activation='softmax'))  # This is the classification layer
    model.summary()
    return model
