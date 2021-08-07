from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, \
    Conv2D, Activation, Add, AveragePooling2D, Input, ZeroPadding2D, MaxPooling2D
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
from tensorflow.keras.models import Model


def model_architectures():
    # model = resnet_transfer_learning()
    # model = resnet_convnet_transfer_learning()
    # model = resnet_transfer_learning_skip_connection()
    model = resNet50_scratch()
    return model


def dense_net_transfer_learning():
    num_classes = 10  # y_train.shape[1]
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=num_classes)
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
    model.add(Dense(num_classes, activation='softmax'))  # This is the classification layer
    model.summary()
    return model


def dense_net_convnet_transfer_learning():
    num_classes = 10  # y_train.shape[1]
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=num_classes)
    base_model.trainable = False
    model = Sequential()
    # Add the Dense layers along with activation and batch normalization
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))  # Regularize with dropout
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
