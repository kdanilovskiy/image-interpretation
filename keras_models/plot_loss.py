import tensorflow as tf
from IPython.display import clear_output
import pylab as plt


class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, title=''):
        self.title = title

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        if self.i < self.params['epochs']:
            clear_output(wait=True)

            plt.plot(self.x, self.losses, label="training")
            plt.plot(self.x, self.val_losses, label="validation")
            plt.xlabel('No of epochs')
            plt.ylabel('Loss')
            plt.title(self.title)
            plt.legend()
            plt.show()
