import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from src.core.data_preparation import *
from src.vgg.model_architectures import *
from src.vgg.model_architectures_transfer_learning import *


def build_model():
    # model = model_architecture()
    model = model_architecture_tf()
    return model


def compile_model(model):
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])


def start_training_custom(model, train_set, test_set):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath='artifacts/model/vgg/custom_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_reducer]

    start = datetime.now()
    model.fit(
        train_set,
        validation_data=test_set,
        epochs=1,
        steps_per_epoch=5,
        validation_steps=32,
        verbose=1,
        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def start_training_cifar(model, X_train, y_train, X_test, y_test):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath='artifacts/model/vgg/cifar10_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_reducer]

    start = datetime.now()
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=1,
                        batch_size=1000,
                        callbacks=callbacks)
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


# Usage:
# Option 1: For Transfer Learning on custom dataset
#   i. Enable "build_vgg_model_transfer_leaning_custom" in model_architectures_transfer_learning
#   ii. Uncomment calls for data_preparation_custom and start_training_custom

# Option 2: For Transfer Learning on CIFAR10 dataset
#   i. Enable "build_vgg_model_transfer_leaning_cifar" in model_architectures_transfer_learning
#   ii. Add prediction = Dense(10, activation='softmax')(x)
#   ii. Uncomment calls for data_preparation_cifar10 and start_training_cifar

# Option 3: For Scratch Training on CIFAR10 dataset with image size of 32
#   i. Enable "build_model_vgg_16 or build_model_vgg_19" in model_architectures
#   ii. Uncomment calls for data_preparation_cifar10_32 and start_training_cifar
def model_preparation():
    # train_dataset, test_dataset = data_preparation_custom()
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()
    model = build_model()
    compile_model(model)
    # start_training_custom(model, train_dataset, test_dataset)
    start_training_cifar(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    model_preparation()
