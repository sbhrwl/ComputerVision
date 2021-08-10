from tensorflow.keras import Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D


def efficient_net_transfer_learning(classes):
    base_model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=classes)
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.7))

    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))

    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.3))

    model.add(Dense(classes, activation='softmax'))
    model.summary()
    return model


def efficient_net_convnet_transfer_learning(classes):
    base_model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=classes)
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    return model
