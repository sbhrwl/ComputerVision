import math
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
# from keras.optimizers import SGD, Adam
from src.core.data_preparation import data_preparation_cifar_32
from src.resnet.model_architectures import resnet_transfer_learning
from src.core.plot_learning_curve import plot_training_history


def build_model(y_train):
    model = resnet_transfer_learning(y_train)
    return model


def decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def compile_model(model):
    # learn_rate = .001
    # sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
    # adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def start_training(model, X_train, y_train, X_validation, y_validation):
    # Setup Train and Validation data
    train_generator = ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    validation_generator = ImageDataGenerator(
        rotation_range=2,
        horizontal_flip=True,
        zoom_range=.1)

    train_generator.fit(X_train)
    validation_generator.fit(X_validation)

    # Setup Checkpoints
    checkpoint = ModelCheckpoint(filepath='artifacts/model/resnet/my_model.h5',
                                 verbose=1,
                                 save_best_only=True)
    lr_sc = LearningRateScheduler(decay, verbose=1)
    lrr = ReduceLROnPlateau(
        monitor='val_accuracy',  # Metric to be measured
        factor=.01,  # Factor by which learning rate will be reduced
        patience=3,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
        min_lr=1e-5)  # The minimum learning rate
    callbacks = [checkpoint, lr_sc, lrr]

    # Setup Training parameters
    batch_size = 100
    epochs = 1  # 50

    start = datetime.now()
    model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size),
              epochs=epochs,
              steps_per_epoch=X_train.shape[0] // batch_size,
              validation_data=validation_generator.flow(X_validation, y_validation, batch_size=batch_size),
              validation_steps=250,
              callbacks=callbacks,
              verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")
    plot_training_history(model)


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_32()
    model = build_model(train_labels)
    compile_model(model)
    start_training(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    model_preparation()
