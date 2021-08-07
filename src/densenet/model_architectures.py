from tensorflow.keras import Sequential
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D


def model_architectures():
    print("DenseNet")
    # model = dense_net_transfer_learning()
    model = dense_net_convnet_transfer_learning()
    return model


def dense_net_transfer_learning():
    num_classes = 10  # y_train.shape[1]
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=num_classes)
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

    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def dense_net_convnet_transfer_learning():
    num_classes = 10  # y_train.shape[1]
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=num_classes)
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
