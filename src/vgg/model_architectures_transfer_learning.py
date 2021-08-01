from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from glob import glob


def model_architecture_tf():
    model = build_vgg_model_transfer_leaning_custom()
    # model = build_vgg_model_vgg16_transfer_learning_cifar()
    # model = build_vgg_model_vgg19_transfer_learning_cifar()
    return model


def build_vgg_model_transfer_leaning_custom():
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
    # prediction = Dense(10, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    return model


def build_vgg_model_vgg16_transfer_learning_cifar():
    IMAGE_SIZE = [32, 32]
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    # don't train existing weights
    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    prediction = Dense(10, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    return model


def build_vgg_model_vgg19_transfer_learning_cifar():
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=10)
    # don't train existing weights
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)  # Adds the base model (in this case vgg19 to model)
    model.add(Flatten())
    model.summary()
    model.add(Dense(1024, activation='relu', input_dim=512))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # This is the classification layer
    model.summary()
    return model
