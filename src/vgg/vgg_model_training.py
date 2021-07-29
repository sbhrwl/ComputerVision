import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from src.vgg.data_preparation import data_preparation
from src.vgg.vgg_model_architectures import build_vgg_model_transfer_leaning, build_model_vgg_16, build_model_vgg_19
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


def build_model():
    model = build_vgg_model_transfer_leaning()
    return model


def compile_model(model):
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])


def start_training(model, train_set, test_set):
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # num_epochs = 1000
    # num_batch_size = 32

    checkpoint = ModelCheckpoint(filepath='artifacts/model/vgg/my_model.h5',
                                 verbose=1, save_best_only=True)

    callbacks = [checkpoint, lr_reducer]

    start = datetime.now()

    model.fit_generator(
        train_set,
        validation_data=test_set,
        epochs=1,
        steps_per_epoch=5,
        validation_steps=32,
        callbacks=callbacks, verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def model_preparation():
    train_dataset, test_dataset = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, train_dataset, test_dataset)


if __name__ == '__main__':
    model_preparation()
