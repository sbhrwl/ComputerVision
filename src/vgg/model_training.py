import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from src.core.data_preparation import *
from src.vgg.model_architectures import *


def build_model():
    model = model_architecture()
    return model


def start_training(model, X_train, y_train, X_test, y_test, custom, train_set, test_set):
    # Step 1: Compile model
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Step 2: Setup Checkpoints
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath='artifacts/model/vgg/cifar10_model.h5',
                                 verbose=1, save_best_only=True)
    callbacks = [checkpoint, lr_reducer]

    # Step 3: Setup Training parameters
    batch_size = 1000
    epochs = 1  # 50
    steps_per_epoch = 5
    validation_steps = 32

    # Step 4: Start Training
    start = datetime.now()
    if custom == "Y":
        print("Start training on Custom dataset")
        history = model.fit(train_set,
                            validation_data=test_set,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            verbose=1,
                            callbacks=callbacks)
    else:
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks)
    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Step 5: Save Model
    print("Model saved to disk via ModelCheckpoint callback")


# Usage:
# Option 1: For Transfer Learning on custom dataset
# Option 2: For Transfer Learning on CIFAR10 dataset
def model_preparation():
    train_dataset, test_dataset = data_preparation_custom()
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
        data_preparation_cifar_original()
    model = build_model()
    start_training(model, train_features, train_labels, validation_features, validation_labels,
                   'Y', train_dataset, test_dataset)


if __name__ == '__main__':
    model_preparation()
