import tensorflow
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


class PlotLosses(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.val_acc = []
        self.baseline = []
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):

        self.x.append(self.i)
        self.acc.append(logs.get('acc'))
        self.baseline.append(0.1428)
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        # clear_output(wait=True)
        plt.plot(self.x, self.acc, label="training accuracy")
        plt.legend()
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()
        plt.show()
        print(self.val_acc)
