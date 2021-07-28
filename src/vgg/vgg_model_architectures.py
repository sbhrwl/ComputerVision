from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from glob import glob


def build_vgg_model_transfer_leaning():
    IMAGE_SIZE = [224, 224]
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    # useful for getting number of classes
    folders = glob('images/train/*')
    print(len(folders))

    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    return model


def build_model_vgg_16():
    vgg_16 = Sequential()

    # first block
    vgg_16.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      input_shape=(224, 224, 3)))
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
    vgg_16.add(Dense(1000, activation='softmax'))

    vgg_16.summary()
    return vgg_16


def build_model_vgg_19():
    vgg_19 = Sequential()

    # first block
    vgg_19.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                      input_shape=(224, 224, 3)))
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
    vgg_19.add(Dense(1000, activation='softmax'))

    vgg_19.summary()
    return vgg_19
