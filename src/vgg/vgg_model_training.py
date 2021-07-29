import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from src.vgg.data_preparation import data_preparation_custom, data_preparation_cifar10, data_preparation_cifar10_32
from src.vgg.vgg_model_architectures import build_vgg_model_transfer_leaning, build_model_vgg_16, build_model_vgg_19
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


def build_model():
    # model = build_vgg_model_transfer_leaning()
    # model = build_model_vgg_16()
    model = build_model_vgg_19()
    return model


def compile_model(model):
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])


def start_training_custom(model, train_set, test_set):
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # num_epochs = 1000
    # num_batch_size = 32

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
        callbacks=callbacks, verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def start_training_cifar(model, X_train, y_train, X_test, y_test):
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # num_epochs = 1000
    # num_batch_size = 32

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


def model_preparation():
    # train_dataset, test_dataset = data_preparation_custom()
    # train_features, train_labels, test_features, test_labels = data_preparation_cifar10(224, 224)
    train_features, train_labels, test_features, test_labels = data_preparation_cifar10_32()
    model = build_model()
    compile_model(model)
    # start_training_custom(model, train_dataset, test_dataset)
    start_training_cifar(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()
