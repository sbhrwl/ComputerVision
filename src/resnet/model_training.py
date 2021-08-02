import math
from datetime import datetime
import time
from src.core.utils import get_random_eraser
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
# from keras.optimizers import SGD, Adam
from src.core.data_preparation import *
from src.resnet.model_architectures import model_architectures
from src.core.plot_learning_curve import plot_training_history


def build_model():
    model = model_architectures()
    return model


def decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def start_training(model, X_train, y_train, X_validation, y_validation, erasure_encoding):
    # Setup Train and Validation data
    if erasure_encoding:
        print("Performing erasure encoding")
        train_generator = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        validation_generator = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        train_generator.fit(X_train)
        validation_generator.fit(X_validation)
    else:
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

    # Step 1: Compile model
    # learn_rate = .001
    # sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
    # adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, rmsgrad=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 2: Setup Checkpoints
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

    # Step 3: Setup Training parameters
    batch_size = 100
    epochs = 1  # 50

    # Step 4: Start Training
    start = datetime.now()  # time.time()
    model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size),
              validation_data=validation_generator.flow(X_validation, y_validation, batch_size=batch_size),
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


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()  # data_preparation_cifar_original, data_preparation_cifar100_eraser
    model = build_model()
    start_training(model, train_features, train_labels, validation_features, validation_labels, 'N')


if __name__ == '__main__':
    model_preparation()
