import math
from datetime import datetime
import time

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
# from keras.optimizers import SGD, Adam
from src.core.data_preparation import *
from src.core.model_architectures import *
from src.core.utils import get_random_eraser, decay
from src.core.plot_learning_curve import plot_training_history
from src.core.plot_losses import PlotLosses


def start_training_custom_dataset(model, train_set, test_set):
    # Step 1: Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Step 2: Setup Training parameters
    batch_size = 128
    epochs = 1  # 50
    steps_per_epoch = 5
    validation_steps = 32

    # Step 3: Start Training
    history = model.fit(train_set,
                        validation_data=test_set,
                        epochs=epochs,
                        batch_size=batch_size,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=1)


def start_training(model, X_train, y_train, X_validation, y_validation):
    data_augmentation = "none"
    inception_scratch_training = "N"

    # Setup Train and Validation data
    if data_augmentation == "none":
        print("Performing no data augmentation")
    elif data_augmentation == "preprocess":
        print("Performing preprocess data augmentation")
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
    elif data_augmentation == "random_erasing":
        print("Performing random erasing")
        train_generator = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        validation_generator = ImageDataGenerator(
            preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        train_generator.fit(X_train)
        validation_generator.fit(X_validation)

    # Step 1: Compile model
    # learn_rate = .001
    # sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
    # adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, rmsgrad=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 2: Setup Checkpoints
    checkpoint = ModelCheckpoint(filepath='artifacts/model/densenet/my_model.h5',
                                 verbose=1,
                                 save_best_only=True)
    lr_sc = LearningRateScheduler(decay, verbose=1)
    lrr = ReduceLROnPlateau(
        monitor='val_accuracy',  # Metric to be measured
        factor=.01,  # Factor by which learning rate will be reduced
        patience=3,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
        min_lr=1e-5)  # The minimum learning rate
    plot_losses = PlotLosses()
    callbacks = [checkpoint, lr_sc, lrr, plot_losses]

    # Step 3: Setup Training parameters
    batch_size = 100
    epochs = 1  # 50

    # Step 4: Start Training
    start = datetime.now()  # time.time()
    if data_augmentation == "none":
        if inception_scratch_training == "Y":
            print("Starting Inception training from scratch")
            history = model.fit(X_train, [y_train, y_train, y_train],
                                validation_data=(X_validation, [y_validation, y_validation, y_validation]),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks)
        else:
            history = model.fit(X_train, y_train,
                                validation_data=(X_validation, y_validation),
                                epochs=1,
                                batch_size=256,
                                callbacks=callbacks)
    else:
        history = model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=validation_generator.flow(X_validation, y_validation,
                                                                      batch_size=batch_size),
                            # validation_data=(X_validation, y_validation),
                            epochs=epochs,
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_steps=250,
                            callbacks=callbacks,
                            verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # print('Training time: %s' % (t - time.time()))

    # Step 5: Save Model
    print("Model saved to disk via ModelCheckpoint callback")

    # Step 6: Plot Training history
    plot_training_history(model)


def model_preparation_custom_dataset():
    train_dataset, test_dataset = data_preparation_custom()
    model = build_model()
    start_training_custom_dataset(model, train_dataset, test_dataset)


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()
    model = build_model()
    start_training(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    model_preparation_custom_dataset()
    # model_preparation()
