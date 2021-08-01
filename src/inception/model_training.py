import math
from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.optimizers import SGD
from src.core.data_preparation import *
from src.inception.model_architectures import build_model_inception
from src.inception.model_architectures_transfer_learning import *


def build_model():
    # model = build_model_inception()
    model = inception_transfer_learning()
    return model


def decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def start_scratch_training(model, X_train, y_train, X_validation, y_validation):
    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1, 0.3, 0.3],
        optimizer='sgd',
        metrics=['accuracy'])

    lr_sc = LearningRateScheduler(decay, verbose=1)
    checkpoint = ModelCheckpoint(filepath='artifacts/model/inception/my_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_sc]

    start = datetime.now()
    history = model.fit(X_train,
                        [y_train, y_train, y_train],
                        validation_data=(X_validation, [y_validation, y_validation, y_validation]),
                        epochs=1,
                        batch_size=256,
                        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def start_training(model, X_train, y_train, X_validation, y_validation):
    # initial_learning_rate = 0.01
    # sgd = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=False)
    model.compile(
        loss='categorical_crossentropy',
        loss_weights=[1, 0.3, 0.3],
        optimizer='sgd',
        metrics=['accuracy'])

    lr_sc = LearningRateScheduler(decay, verbose=1)
    checkpoint = ModelCheckpoint(filepath='artifacts/model/inception/my_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_sc]

    start = datetime.now()
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        epochs=1,
                        batch_size=256,
                        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()
    model = build_model()
    # start_scratch_training(model, train_features, train_labels, validation_features, validation_labels)
    start_training(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    model_preparation()
