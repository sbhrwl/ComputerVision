import math
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from src.core.plot_losses import PlotLosses


def decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.96
    epochs_drop = 8
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def get_callbacks():
    filepath='artifacts/model/densenet/my_model.h5'
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 verbose=1,
                                 save_best_only=True)
    lr_sc = LearningRateScheduler(decay, verbose=1)
    lrr = ReduceLROnPlateau(
        monitor='val_accuracy',  # Metric to be measured
        factor=.01,  # Factor by which learning rate will be reduced
        patience=3,  # No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
        min_lr=1e-5)  # The minimum learning rate
    plot_losses = PlotLosses()
    return [checkpoint, lr_sc, lrr, plot_losses]
