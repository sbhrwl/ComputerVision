import math
from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.optimizers import SGD
from src.core.data_preparation import *
from src.inception.model_architectures import model_architectures


def build_model():
    model = model_architectures()
    return model


def decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def start_scratch_training(model, X_train, y_train, X_validation, y_validation):
    # Step 1: Compile model
    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1, 0.3, 0.3],
        optimizer='sgd',
        metrics=['accuracy'])

    # Step 2: Setup Checkpoints
    lr_sc = LearningRateScheduler(decay, verbose=1)
    checkpoint = ModelCheckpoint(filepath='artifacts/model/inception/my_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_sc]

    # Step 3: Setup Training parameters
    batch_size = 256
    epochs = 1  # 50

    start = datetime.now()
    history = model.fit(X_train,
                        [y_train, y_train, y_train],
                        validation_data=(X_validation, [y_validation, y_validation, y_validation]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Step 5: Save Model
    print("Model saved to disk via ModelCheckpoint callback")


def start_training(model, X_train, y_train, X_validation, y_validation):
    # Step 1: Compile model
    # initial_learning_rate = 0.01
    # sgd = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=False)
    model.compile(
        loss='categorical_crossentropy',
        loss_weights=[1, 0.3, 0.3],
        optimizer='sgd',
        metrics=['accuracy'])

    # Step 2: Setup Checkpoints
    lr_sc = LearningRateScheduler(decay, verbose=1)
    checkpoint = ModelCheckpoint(filepath='artifacts/model/inception/my_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_sc]

    # Step 3: Setup Training parameters
    batch_size = 256
    epochs = 1  # 50

    # Step 4: Start Training
    start = datetime.now()
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        epochs=1,
                        batch_size=256,
                        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Step 5: Save Model
    print("Model saved to disk via ModelCheckpoint callback")


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()
    model = build_model()
    # start_scratch_training(model, train_features, train_labels, validation_features, validation_labels)
    start_training(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    model_preparation()
