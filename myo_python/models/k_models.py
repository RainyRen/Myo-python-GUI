# coding: utf-8
# import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Concatenate,
                          Dense, Flatten,
                          Conv2D, MaxPooling2D,
                          TimeDistributed,
                          Dropout,
                          LSTM, ConvLSTM2D,
                          GRU,
                          Activation)
from keras.optimizers import Adam
from keras.callbacks import Callback


# # =============================================== post fusion model ==================================================
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
    model input kinematic and emg in voltage, output next time step position
    :param dict model_config:
    :param bool inference:
    :return:
    """
    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 16))

    # # define structures
    if inference:
        kinematic_rnn_cell = LSTM(
            rnn_neurons,
            return_sequences=False,
            stateful=False)(input_kinematic)
        emg_rnn_cell = LSTM(
            rnn_neurons,
            return_sequences=False,
            stateful=False)(input_emg)

    else:
        kinematic_rnn_cell = LSTM(
            rnn_neurons,
            dropout=0.2,
            return_sequences=False,
            stateful=False)(input_kinematic)
        emg_rnn_cell = LSTM(
            rnn_neurons,
            dropout=0.2,
            return_sequences=False,
            stateful=False)(input_emg)

    merge_data = Concatenate()([kinematic_rnn_cell, emg_rnn_cell])

    hidden_1 = Dense(hidden_1_neurons, activation='relu')(merge_data)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(hidden_2_neurons, activation='relu')(hidden_1)
    hidden_2 = Dropout(0.3)(hidden_2)
    output = Dense(4, activation=None)(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

    return model


def multi2multi_stft(model_config, inference=False):
    """
    model input kinematic and emg in short-time Fourier transform, output the whole time step prediction
    :param dict model_config:
    :param bool inference:
    :return:
    """
    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 160))

    # # define structures
    if inference:
        kinematic_rnn_cell = LSTM(
            rnn_neurons,
            return_sequences=True,
            stateful=False)(input_kinematic)
        emg_rnn_cell = LSTM(
            rnn_neurons,
            return_sequences=True,
            stateful=False)(input_emg)

    else:
        kinematic_rnn_cell = LSTM(
            128,
            dropout=0.3,
            recurrent_dropout=0.2,
            return_sequences=True,
            stateful=False)(input_kinematic)
        emg_rnn_cell = LSTM(
            256,
            dropout=0.3,
            recurrent_dropout=0.2,
            return_sequences=True,
            stateful=False)(input_emg)

    merge_data = Concatenate()([kinematic_rnn_cell, emg_rnn_cell])

    hidden_1 = TimeDistributed(Dense(hidden_1_neurons), activation='selu')(merge_data)
    hidden_1 = Dropout(0.5)(hidden_1)

    hidden_2 = TimeDistributed(Dense(hidden_2_neurons), activation='selu')(hidden_1)
    hidden_2 = Dropout(0.5)(hidden_2)

    output = TimeDistributed(Dense(4, activation=None))(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model


def multi2one_stft(model_config, inference=False):
    """
    model input kinematic and emg in short-time Fourier transform, output the whole time step prediction
    :param dict model_config:
    :param bool inference:
    :return:
    """
    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 16, 10, 1))

    # # define structures
    embed = TimeDistributed(Dense(64, activation=None))(input_kinematic)

    cnn_1 = TimeDistributed(
        Conv2D(
            16,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(input_emg)
    cnn_2 = TimeDistributed(
        Conv2D(
            16,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(cnn_1)

    max_pool_1 = TimeDistributed(MaxPooling2D(pool_size=(2, 1)))(cnn_2)

    cnn_3 = TimeDistributed(
        Conv2D(
            32,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(max_pool_1)

    cnn_4 = TimeDistributed(
        Conv2D(
            32,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(cnn_3)

    max_pool_2 = TimeDistributed(MaxPooling2D(pool_size=(2, 1)))(cnn_4)

    # cnn_4 = TimeDistributed(
    #     Conv2D(
    #         64,
    #         3,
    #         data_format="channels_first",
    #         padding='same',
    #         activation='relu',
    #         strides=1))(max_pool_2)

    # flatten = TimeDistributed(Flatten())(cnn_4)

    kinematic_rnn_cell = LSTM(
        64,
        dropout=0.1,
        return_sequences=False,
        stateful=False)(embed)
    # emg_rnn_cell = LSTM(
    #     256,
    #     dropout=0.3,
    #     return_sequences=False,
    #     stateful=False)(cnn_4)
    emg_rnn_cell = ConvLSTM2D(
        64,  # # number of filters
        (3, 1),  # # kernel size
        strides=(1, 1),
        padding='same',
        dropout=0.1,
        data_format='channels_first',
        dilation_rate=(1, 1),
        return_sequences=False,
        stateful=False)(max_pool_2)

    emg_rnn_cell = Flatten()(emg_rnn_cell)

    emg_projection = Dense(64, activation='relu')(emg_rnn_cell)
    kinematic_projection = Dense(64, activation='linear')(kinematic_rnn_cell)

    merge_data = Concatenate()([kinematic_projection, emg_projection])

    hidden_1 = Dense(hidden_1_neurons, activation='linear')(merge_data)
    hidden_1 = Dropout(0.2)(hidden_1)

    hidden_2 = Dense(hidden_2_neurons, activation='linear')(hidden_1)
    hidden_2 = Dropout(0.2)(hidden_2)

    output = Dense(4, activation=None)(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])

    return model


def multi2one_simple_stft(model_config, inference=False):
    """
    model input kinematic and emg in short-time Fourier transform, output the whole time step prediction
    :param dict model_config:
    :param bool inference:
    :return:
    """

    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 16, 10, 1))

    # # define structures
    kinematic_rnn_cell = LSTM(
        128,
        dropout=0.1,
        return_sequences=False,
        stateful=False)(input_kinematic)
    emg_rnn_cell = ConvLSTM2D(
        128,                        # # number of filters
        (3, 1),                     # # kernel size
        strides=(1, 1),
        padding='same',
        dropout=0.3,
        data_format='channels_first',
        dilation_rate=(1, 1),
        return_sequences=False,
        stateful=False)(input_emg)

    emg_flatten = Flatten()(emg_rnn_cell)

    merge_data = Concatenate()([kinematic_rnn_cell, emg_flatten])

    hidden_1 = Dense(hidden_1_neurons, activation='selu')(merge_data)
    hidden_1 = Dropout(0.4)(hidden_1)

    hidden_2 = Dense(hidden_2_neurons, activation='selu')(hidden_1)
    hidden_2 = Dropout(0.4)(hidden_2)

    output = Dense(4, activation=None)(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

    return model


# # ====================================================================================================================
# # ============================================== previous fusion =====================================================
def one2one(model_config):
    """
    merge kinematic and emg before input model
    :param model_config:
    :return:
    """
    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 16))

    merge_data = Concatenate()([input_kinematic, input_emg])

    rnn_cell = LSTM(rnn_neurons, dropout=0.2, return_sequences=False)(merge_data)

    hidden_1 = Dense(hidden_1_neurons, activation='linear')(rnn_cell)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(hidden_2_neurons, activation='linear')(hidden_1)
    hidden_2 = Dropout(0.3)(hidden_2)
    output = Dense(4, activation=None)(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])

    return model


def one2one_stft(model_config):
    """
    merge kinematic and emg before input model
    :param model_config:
    :return:
    """
    rnn_neurons = model_config['rnn_neurons']
    hidden_1_neurons = model_config['hidden_1_neurons']
    hidden_2_neurons = model_config['hidden_2_neurons']

    # # define two separate inputs
    input_kinematic = Input(shape=(model_config['time_length'], 16))
    input_emg = Input(shape=(model_config['time_length'], 160))

    merge_data = Concatenate()([input_kinematic, input_emg])

    rnn_cell = LSTM(rnn_neurons, dropout=0.2, return_sequences=False)(merge_data)

    hidden_1 = Dense(hidden_1_neurons, activation='relu')(rnn_cell)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(hidden_2_neurons, activation='relu')(hidden_1)
    hidden_2 = Dropout(0.3)(hidden_2)
    output = Dense(4, activation=None)(hidden_2)

    model = Model([input_kinematic, input_emg], output)
    model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])

    return model


# # ====================================================================================================================
# # ============================================== classifier ==========================================================
def bin_cls(model_config):
    # from keras import metrics

    hidden_1_neurons = 128
    hidden_2_neurons = 128

    input_emg = Input(shape=(model_config['time_length'], 16, 10, 1))

    # # define structures
    cnn_1 = TimeDistributed(
        Conv2D(
            16,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(input_emg)
    cnn_2 = TimeDistributed(
        Conv2D(
            16,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(cnn_1)

    max_pool_1 = TimeDistributed(MaxPooling2D(pool_size=(2, 1)))(cnn_2)

    cnn_3 = TimeDistributed(
        Conv2D(
            32,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(max_pool_1)

    max_pool_2 = TimeDistributed(MaxPooling2D(pool_size=(2, 1)))(cnn_3)

    cnn_4 = TimeDistributed(
        Conv2D(
            64,
            3,
            data_format="channels_first",
            padding='same',
            activation='relu',
            strides=1))(max_pool_2)

    emg_rnn_cell = ConvLSTM2D(
        64,  # # number of filters
        (3, 1),  # # kernel size
        strides=(1, 1),
        padding='same',
        dropout=0.2,
        data_format='channels_first',
        dilation_rate=(1, 1),
        return_sequences=False,
        stateful=False)(cnn_4)

    emg_flatten = Flatten()(emg_rnn_cell)

    hidden_1 = Dense(hidden_1_neurons, activation='relu')(emg_flatten)
    hidden_1 = Dropout(0.4)(hidden_1)

    hidden_2 = Dense(hidden_2_neurons, activation='relu')(hidden_1)
    hidden_2 = Dropout(0.4)(hidden_2)

    output = Dense(4, activation='sigmoid')(hidden_2)

    model = Model(input_emg, output)
    model.summary()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', f1_score])

    return model

# # ====================================================================================================================


class DumpHistory(Callback):
    """
    Recode every epoch history
    """

    def __init__(self, file_path):
        super(self.__class__, self).__init__()
        self.file_path = file_path
        with open(file_path, 'w') as log_file:
            log_file.write('epoch,loss,val_loss\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.file_path, 'a') as log_file:
            log = "{0},{1},{2}\n".format(
                epoch,
                logs.get('loss'),
                logs.get('val_loss'),
            )
            log_file.write(log)


def f1_score(y_true, y_pred):
    thresh = 0.5
    y_pred = K.cast(K.greater(y_pred, thresh), dtype='float32')
    tp = K.sum(y_true * y_pred, axis=-1)

    precision = tp / (K.sum(y_pred, axis=-1) + K.epsilon())
    recall = tp / (K.sum(y_true, axis=-1) + K.epsilon())
    return K.mean(2 * ((precision * recall) /
                       (precision + recall + K.epsilon())))


if __name__ == "__main__":
    pass
