import math
from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from src.inception.data_preparation import data_preparation
from src.inception.model_architectures import build_model_inception
# from keras.optimizers import SGD


def build_model():
    model = build_model_inception()
    return model


def decay(epoch, steps=100):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def compile_model(model):
    # initial_learning_rate = 0.01
    # sgd = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=False)

    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1, 0.3, 0.3],
        optimizer='sgd',
        metrics=['accuracy'])


def start_training(model, X_train, y_train, X_test, y_test):
    lr_sc = LearningRateScheduler(decay, verbose=1)

    checkpoint = ModelCheckpoint(filepath='artifacts/model/inception/my_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_sc]

    start = datetime.now()

    history = model.fit(X_train,
                        [y_train, y_train, y_train],
                        validation_data=(X_test, [y_test, y_test, y_test]),
                        epochs=1,
                        batch_size=256,
                        callbacks=callbacks)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def model_preparation():
    # train_features, train_labels, test_features, test_labels = data_preparation(32, 32)
    train_features, train_labels, test_features, test_labels = data_preparation(224, 224)
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()
