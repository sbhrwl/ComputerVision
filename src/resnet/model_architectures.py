from keras import Sequential
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, Dense, Dropout, UpSampling2D, GlobalAveragePooling2D, BatchNormalization


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


def resnet_transfer_learning_skip_connection(y_train):
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    num_classes = 100

    for layer in resnet_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.25))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model
