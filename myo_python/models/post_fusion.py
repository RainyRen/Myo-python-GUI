# coding: utf-8
# import numpy as np
from keras.models import Model
from keras.layers import (Input, Concatenate,
                          Dense,
                          TimeDistributed,
                          Dropout,
                          BatchNormalization,
                          GRU,
                          CuDNNLSTM)
# from keras.optimizers import Adam
from keras.callbacks import Callback
import tensorflow as tf


def multi2multi(model_config, inference=False):
    """
    multiple input get multiple output, return are vector
    :param dict model_config:
    :param bool inference:
    :return:
    """
    if inference:
        print("load exits model")
    else:
        rnn_neurons = model_config['rnn_neurons']
        hidden_1_neurons = model_config['hidden_1_neurons']
        hidden_2_neurons = model_config['hidden_2_neurons']

        # # define two separate inputs
        input_kinematic = Input(shape=(model_config['time_length'], 15))
        input_emg = Input(shape=(model_config['time_length'], 16))

        # # define structures
        kinematic_rnn_cell = GRU(
            rnn_neurons,
            dropout=0.2,
            return_sequences=True,
            stateful=False)(input_kinematic)
        emg_rnn_cell = GRU(
            rnn_neurons,
            dropout=0.2,
            return_sequences=True,
            stateful=False)(input_emg)

        merge_data = Concatenate()([kinematic_rnn_cell, emg_rnn_cell])

        hidden_1 = TimeDistributed(Dense(hidden_1_neurons, activation='relu'))(merge_data)
        hidden_1 = Dropout(0.25)(hidden_1)
        hidden_2 = TimeDistributed(Dense(hidden_2_neurons, activation='relu'))(hidden_1)
        hidden_2 = Dropout(0.25)(hidden_2)
        output = TimeDistributed(Dense(3, activation=None))(hidden_2)

        model = Model([input_kinematic, input_emg], output)
        model.summary()
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])

        return model


def multi2one(model_config, inference=False):
    """

    :param dict model_config:
    :param bool inference:
    :return:
    """
    if inference:
        print("load exits model")
    else:
        rnn_neurons = model_config['rnn_neurons']
        hidden_1_neurons = model_config['hidden_1_neurons']
        hidden_2_neurons = model_config['hidden_2_neurons']

        # # define two separate inputs
        input_kinematic = Input(shape=(model_config['time_length'], 15))
        input_emg = Input(shape=(model_config['time_length'], 16))

        # # define structures
        kinematic_rnn_cell = CuDNNLSTM(
            rnn_neurons,
            # dropout=0.2,
            return_sequences=False,
            stateful=False)(input_kinematic)
        emg_rnn_cell = CuDNNLSTM(
            rnn_neurons,
            # dropout=0.2,
            return_sequences=False,
            stateful=False)(input_emg)

        merge_data = Concatenate()([kinematic_rnn_cell, emg_rnn_cell])

        hidden_1 = Dense(hidden_1_neurons, activation='relu')(merge_data)
        hidden_1 = Dropout(0.25)(hidden_1)
        hidden_2 = Dense(hidden_2_neurons, activation='relu')(hidden_1)
        hidden_2 = Dropout(0.25)(hidden_2)
        output = Dense(3, activation=None)(hidden_2)

        model = Model([input_kinematic, input_emg], output)
        model.summary()
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])

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


# # ========== tensorflow model ==========
class Multi2One(object):
    def __init__(self, model_config):
        """

        """
        self.model_config = model_config
        self.dropout_rate = 0.2

        # # ===== build graph =====
        self.kinematic_in = tf.placeholder(tf.float32, [None, self.model_config['time_length'], 15])
        self.emg_in = tf.placeholder(tf.float32, [None, self.model_config['time_length'], 16])

        self.kinematic_rnn_cell = tf.contrib.rnn.GRUCell(self.model_config['rnn_neurons'])
        self.kinematic_rnn_cell = tf.contrib.rnn.DropoutWrapper(
            self.kinematic_rnn_cell,
            input_keep_prob=1.0 - self.dropout_rate,
            output_keep_prob=1.0 - self.dropout_rate,
            state_keep_prob=1.0 - self.dropout_rate,
            variational_recurrent=False
        )

        self.emg_rnn_cell = tf.contrib.rnn.BasicGRUCellLSTMCell()


if __name__ == "__main__":
    pass
