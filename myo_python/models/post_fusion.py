# coding: utf-8
# import numpy as np
from keras.models import Model
from keras.layers import (Input, Concatenate,
                          Dense,
                          TimeDistributed,
                          Dropout,
                          BatchNormalization,
                          GRU)
# from keras.optimizers import Adam
from keras.callbacks import Callback


def basic_model(model_config, inference=False):
    if inference:
        print("load exits model")
    else:
        rnn_neurons = model_config['rnn_neurons']
        hidden_1_neurons = model_config['hidden_1_neurons']
        hidden_2_neurons = model_config['hidden_2_neurons']

        # # define two separate inputs
        input_kinematic = Input(shape=(model_config['time_length'], 16))
        input_emg = Input(shape=(model_config['time_length'], 16))

        # # define structures
        kinematic_rnn_cell = GRU(rnn_neurons, dropout=0.2, return_sequences=True)(input_kinematic)
        emg_rnn_cell = GRU(rnn_neurons, dropout=0.2, return_sequences=True)(input_emg)

        merge_data = Concatenate()([kinematic_rnn_cell, emg_rnn_cell])

        hidden_1 = TimeDistributed(Dense(hidden_1_neurons, activation='relu'))(merge_data)
        hidden_1 = Dropout(0.2)(hidden_1)
        hidden_2 = TimeDistributed(Dense(hidden_2_neurons, activation='relu'))(hidden_1)
        hidden_2 = Dropout(0.2)(hidden_2)
        output = TimeDistributed(Dense(3, activation=None))(hidden_2)

        model = Model([input_kinematic, input_emg], output)
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        return model


class DumpHistory(Callback):
    """
    Recode every epoch history
    """

    def __init__(self, file_path):
        super(self.__class__, self).__init__()
        self.file_path = file_path
        with open(file_path, 'w') as log_file:
            log_file.write('epoch,loss,acc,val_loss,val_acc,e_dist\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.file_path, 'a') as log_file:
            log = "{0},{1},{2},{3},{4}\n".format(
                epoch,
                logs.get('loss'),
                logs.get('acc'),
                logs.get('val_loss'),
                logs.get('val_acc'),
            )
            log_file.write(log)


if __name__ == "__main__":
    pass
