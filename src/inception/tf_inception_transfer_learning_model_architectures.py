from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, UpSampling2D
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def inception_transfer_learning():
    # @title Default title text
    conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    conv_base.summary()

    model = models.Sequential()
    # UpSampling increase the row and column of the data.
    # Sometimes if we have less data so we can try to increase the data in this way.
    model.add(UpSampling2D((2, 2)))
    model.add(UpSampling2D((2, 2)))
    model.add(UpSampling2D((2, 2)))

    # conv_base is the inception network
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    input_shape = (None, 32, 32, 3)
    model.build(input_shape)
    model.summary()
    return model


def inception_transfer_learning_starting_from_mixed_7_layer():
    # local_weights_file = path_inception
    pre_trained_model = InceptionV3(input_shape=(128, 128, 3),
                                    include_top=False,
                                    weights='imagenet')
    # pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    pre_trained_model.summary()
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    model.summary()
    return model
