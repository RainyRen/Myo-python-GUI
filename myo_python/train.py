# coding: utf-8
import pdb
import argparse
import yaml
from pathlib import Path

import numpy as np
from keras.callbacks import ModelCheckpoint

from utils import data_io
from models import k_models

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"

# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train_reg", help='train_reg or train_cls or train_trad')
    parser.add_argument('--save_dir', default='./exp')
    args = parser.parse_args()

    # # ===== load config from config file =====
    if args.mode == 'train_reg':
        with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
            train_config = yaml.load(config_file)

        train_reg(args, train_config)

    elif args.mode == 'train_cls':
        with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
            train_config = yaml.load(config_file)

        train_cls(args, train_config)

    elif args.mode == 'train_trad':
        with open(str(CONFIG_PATH / 'train_trad.yml'), 'r') as config_file:
            train_config = yaml.load(config_file)

        train_trad(args, train_config)

    else:
        raise ValueError('No such mode, please check again')


def train_reg(args, train_config):
    """
    :param args:
    :param dict train_config:
    :return: None
    """
    save_folder = Path(args.save_dir) / train_config['exp_folder']

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=False
    )
    val_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/val',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=False
    )

    print("organising materials...\n")
    tr_data, _ = train_data_loader.get_all_data()
    val_data, _ = val_data_loader.get_all_data()
    pdb.set_trace()

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(train_config, write_file, default_flow_style=False)

    tr_kinematic, tr_emg, tr_target = tr_data
    val_kinematic, val_emg, val_target = val_data

    # # obtain a model
    model = k_models.multi2one_stft(train_config)

    checkpoint = ModelCheckpoint(
        filepath=str(save_folder / 'rnn_best.h5'),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_mean_absolute_error',
        mode='min'
    )

    save_history = k_models.DumpHistory(str(save_folder / 'logs.csv'))

    model.fit(
        x=[tr_kinematic, tr_emg], y=tr_target,
        validation_data=([val_kinematic, val_emg], val_target),
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        callbacks=[checkpoint, save_history],
        shuffle=False
    )


def train_cls(args, train_config):
    """
    training a simple classifier to classify each arm angle move or stop
    :param args:
    :param train_config:
    :return:
    """
    save_folder = Path(args.save_dir) / 'multi2one_cls'

    # # ===== get pre-processed data =====
    train_move_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=True
    )
    train_stop_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/stop',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=True
    )

    val_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/val',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=True
    )

    print("organising materials...\n")
    tr_move_data, _ = train_move_data_loader.get_all_data()
    tr_stop_data, _ = train_stop_data_loader.get_all_data()
    val_data, _ = val_data_loader.get_all_data()
    pdb.set_trace()

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(train_config, write_file, default_flow_style=False)

    _, tr_move_emg, tr_move_target = tr_move_data
    _, tr_stop_emg, tr_stop_target = tr_stop_data
    val_kinematic, val_emg, val_target = val_data

    tr_emg = np.concatenate([tr_move_emg, tr_stop_emg], axis=0)
    tr_target = np.concatenate([tr_move_target, tr_stop_target], axis=0)

    # # obtain a model
    model = k_models.bin_cls(train_config)

    checkpoint = ModelCheckpoint(
        filepath=str(save_folder / 'rnn_best.h5'),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        mode='min'
    )

    save_history = k_models.DumpHistory(str(save_folder / 'logs.csv'))

    model.fit(
        x=tr_emg, y=tr_target,
        validation_data=(val_emg, val_target),
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        callbacks=[checkpoint, save_history],
        shuffle=False
    )


def train_trad(args, train_config):
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.externals import joblib

    save_folder = Path(args.save_dir) / train_config['exp_folder']

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=False
    )
    val_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/val',
        separate_rate=0.,
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=False
    )

    print("organising materials...\n")
    tr_data, _ = train_data_loader.get_all_data(get_emg_raw=True, emg_3d=False)
    val_data, _ = val_data_loader.get_all_data(get_emg_raw=True, emg_3d=False)
    pdb.set_trace()

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(train_config, write_file, default_flow_style=False)

    tr_kinematic, tr_emg, tr_target = tr_data
    val_kinematic, val_emg, val_target = val_data

    train_x = np.hstack((tr_kinematic[:, -1, :], tr_emg[:, -1, :]))
    train_y = tr_target
    val_x = np.hstack((val_kinematic[:, -1, :], val_emg[:, -1, :]))
    val_y = val_target

    svr = SVR(C=1.0, epsilon=0.2, kernel='rbf', tol=1e-4)
    lin = LinearRegression(n_jobs=1)


if __name__ == "__main__":
    main()
