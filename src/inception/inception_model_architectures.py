import keras
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout, Input, concatenate, \
    UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import models
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


def build_model_inception():
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)
    input_layer = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    # x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')

    classifier_1 = AveragePooling2D((5, 5), strides=3)(x)
    classifier_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_1)
    classifier_1 = Flatten()(classifier_1)
    classifier_1 = Dense(1024, activation='relu')(classifier_1)
    classifier_1 = Dropout(0.7)(classifier_1)
    classifier_1 = Dense(10, activation='softmax', name='auxiliary_output_1')(classifier_1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')

    classifier_2 = AveragePooling2D((5, 5), strides=3)(x)
    classifier_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_2)
    classifier_2 = Flatten()(classifier_2)
    classifier_2 = Dense(1024, activation='relu')(classifier_2)
    classifier_2 = Dropout(0.7)(classifier_2)
    classifier_2 = Dense(10, activation='softmax', name='auxiliary_output_2')(classifier_2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu', name='linear')(x)
    x = Dense(1000, activation='softmax', name='output')(x)

    model = Model(input_layer, [x], name='googlenet')
    model.summary()

    # GoogleNet model after adding classifiers 1 and 2
    model_with_classifiers = Model(input_layer, [x, classifier_1, classifier_2], name='googlenet_complete_architecture')
    model_with_classifiers.summary()
    return model_with_classifiers


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
