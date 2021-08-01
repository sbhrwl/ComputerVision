from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout


def model_architecture():
    model = build_model_vgg_16()
    # model = build_model_vgg_19()
    return model


def build_model_vgg_16():
    vgg_16 = Sequential()

    # first block
    vgg_16.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      input_shape=(32, 32, 3)))  # input_shape=(224, 224, 3)))
    vgg_16.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # second block
    vgg_16.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # third block
    vgg_16.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # forth block
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # fifth block
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # sixth block (classifier)
    vgg_16.add(Flatten())
    vgg_16.add(Dense(4096, activation='relu'))
    vgg_16.add(Dropout(0.5))
    vgg_16.add(Dense(4096, activation='relu'))
    vgg_16.add(Dropout(0.5))
    # vgg_16.add(Dense(1000, activation='softmax'))
    vgg_16.add(Dense(10, activation='softmax'))

    vgg_16.summary()
    return vgg_16


def build_model_vgg_19():
    vgg_19 = Sequential()

    # first block
    vgg_19.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      input_shape=(32, 32, 3)))  # input_shape=(224, 224, 3)))
    vgg_19.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(MaxPool2D((2, 2), strides=(2, 2)))

    # second block
    vgg_19.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(MaxPool2D((2, 2), strides=(2, 2)))

    # third block
    vgg_19.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(MaxPool2D((2, 2), strides=(2, 2)))

    # forth block
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(MaxPool2D((2, 2), strides=(2, 2)))

    # fifth block
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    vgg_19.add(MaxPool2D((2, 2), strides=(2, 2)))

    # seventh block (classifier)
    vgg_19.add(Flatten())
    vgg_19.add(Dense(4096, activation='relu'))
    vgg_19.add(Dropout(0.5))
    vgg_19.add(Dense(4096, activation='relu'))
    vgg_19.add(Dropout(0.5))
    # vgg_19.add(Dense(1000, activation='softmax'))
    vgg_19.add(Dense(10, activation='softmax'))

    vgg_19.summary()
    return vgg_19
