# coding: utf-8
import yaml

import numpy as np
from keras.models import Model
from keras.layers import (Input, Dense,
                          TimeDistributed,
                          Dropout,
                          BatchNormalization,
                          GRU)
from keras.callbacks import Callback, ModelCheckpoint


class ArmAngleBasic(object):
    def __init__(self, inference=False):
        self.inference = inference

        if self.inference:
            with open('./config/train_default.yml', 'r') as config_file:


            self.model = Model()

            # # define two seperate inputs
            input_kinematic = Input(shape=(16,))
            input_emg = Input(shape=(16,))

            # # define structures
            rnn_cell = GRU()



    def train(self):
        pass

    def predict(self):
        pass


class DumpHistory(Callback):
    """
    Recode every epoch histtory
    """

    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'w') as log_file:
            log_file.write('epoch,loss,acc,val_loss,val_acc,e_dist\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath
        with open(filepath, 'a') as log_file:
            log = "{0},{1},{2},{3},{4},{5}\n".format(
                epoch,
                logs.get('loss'),
                logs.get('acc'),
                logs.get('val_loss'),
                logs.get('val_acc'),
                edist_ave)
            log_file.write(log)


if __name__ == "__main__":
    pass
