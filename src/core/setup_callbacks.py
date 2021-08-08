import math
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from src.core.utils import get_parameters
from src.core.plot_losses import PlotLosses


def decay(epoch):
    config = get_parameters()
    lr_sc_config = config["callbacks"]["LearningRateScheduler"]
    initial_learning_rate = lr_sc_config["initial_learning_rate"]
    drop = lr_sc_config["drop"]
    epochs_drop = lr_sc_config["epochs_drop"]
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return learning_rate


def get_callbacks():
    config = get_parameters()
    filepath = config["callbacks"]["checkpoint_file_path"]

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 verbose=1,
                                 save_best_only=True)

    lr_sc = LearningRateScheduler(decay, verbose=1)

    lrr_config = config["callbacks"]["ReduceLROnPlateau"]
    lrr = ReduceLROnPlateau(
        monitor=lrr_config["monitor"],
        factor=lrr_config["factor"],
        patience=lrr_config["patience"],
        min_lr=lrr_config["min_lr"])

    plot_losses = PlotLosses()
    return [checkpoint, lr_sc, lrr, plot_losses]
