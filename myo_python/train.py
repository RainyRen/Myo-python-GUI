# coding: utf-8
import pdb
import argparse
import yaml
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

from utils import data_io
from models import k_models

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"

# # =============================
# # ========= GPU config ========
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=tf_config))
# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train_reg", help='train_reg or train_cls or train_trad')
    parser.add_argument('--fine_tune', default=None, help='JLW_stft_f4_L2')
    parser.add_argument('--save_dir', default='./exp')
    args = parser.parse_args()

    # # ===== load config from config file =====
    if args.mode == 'train_reg':
        print('\ntraining regression model')

        if args.fine_tune:
            print("Fine Tune -- {} deep model".format(args.fine_tune))
            with open(Path(args.save_dir) / args.fine_tune / 'config.yml', 'r') as config_file:
                train_config = yaml.load(config_file)
        else:
            # # load training config from file
            with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
                train_config = yaml.load(config_file)

        train_reg(args, train_config)

    elif args.mode == 'train_cls':
        print('\ntraining classification model')
        with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
            train_config = yaml.load(config_file)

        train_cls(args, train_config)

    elif args.mode == 'train_trad':
        print('\ntraining traditional regression model')

        if args.fine_tune:
            print("Fine Tune -- {} traditional model".format(args.fine_tune))
            with open(Path(args.save_dir) / args.fine_tune / 'config.yml', 'r') as config_file:
                train_config = yaml.load(config_file)
        else:
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
    training keras regression model
    """
    if args.fine_tune:
        model_folder_path = Path(args.save_dir) / args.fine_tune

        model_weight_path = model_folder_path / 'rnn_weight.h5'
        save_folder = model_folder_path / 'fine_tune'
        model_name = 'rnn_best.h5'

        model = k_models.multi2one_stft_dueling(train_config, fine_tune=True)
        model.load_weights(str(model_weight_path))

    else:
        save_folder = Path(args.save_dir) / train_config['exp_folder']
        model_name = 'rnn_best.h5'
        model = k_models.multi2one_stft_dueling(train_config)

    # # ===== get training and validation data =====
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
    tr_data, _ = train_data_loader.get_all_data(get_emg_raw=False, emg_3d=True)
    val_data, _ = val_data_loader.get_all_data(get_emg_raw=False, emg_3d=True)
    pdb.set_trace()

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(train_config, write_file, default_flow_style=False)

    # # unpack data to get each component composition
    tr_kinematic, tr_emg, tr_target = tr_data
    val_kinematic, val_emg, val_target = val_data

    checkpoint = ModelCheckpoint(
        filepath=str(save_folder / model_name),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_mean_absolute_error',
        mode='min'
    )

    # # define save history callback, we can save history in every epochs
    save_history = k_models.DumpHistory(str(save_folder / 'logs.csv'))

    # # start training
    model.fit(
        x=[tr_kinematic, tr_emg], y=tr_target,
        validation_data=([val_kinematic, val_emg], val_target),
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        callbacks=[checkpoint, save_history],
        shuffle=True
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
        shuffle=True
    )


def train_trad(args, train_config):
    import math

    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.externals import joblib
    from sklearn.metrics import mean_absolute_error

    if args.fine_tune:
        save_folder = Path(args.save_dir) / train_config['exp_folder'] / 'fine_tune'
        lin = joblib.load(str(save_folder.parent / 'lin_model.p'))
        svr = joblib.load(str(save_folder.parent / 'svr_model.p'))
        knr = joblib.load(str(save_folder.parent / 'knr_model.p'))
    else:
        save_folder = Path(args.save_dir) / train_config['exp_folder']
        # # ===== obtain models =====
        lin = [LinearRegression(n_jobs=2) for _ in range(4)]
        svr = [SVR(C=1.0, epsilon=0.2, kernel='rbf', tol=1e-4) for _ in range(4)]
        knr = [KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=2)
               for _ in range(4)]
        # gpr = [GaussianProcessRegressor() for _ in range(4)]

    # # ===== get taining and validation data =====
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

    # # ===== training linear models and evaluate =====
    select_col = list(range(3, train_x.shape[-1]))

    print('\ntraining linear mode...')
    lin_score = list()
    lin_mae = list()
    # # each joint have a model, totally we have four models
    for i in range(4):
        select_col[0] = i
        lin[i].fit(train_x[:, select_col], train_y[:, i])

        lin_score.append(lin[i].score(val_x[:, select_col], val_y[:, i]))

        lin_val_y = lin[i].predict(val_x[:, select_col])
        lin_mae.append(math.degrees(mean_absolute_error(val_y[:, i], lin_val_y)))

        print("linear model {} axis -- mae: {} -- R2: {}".format(i, lin_mae[i], lin_score[i]))

    # # save model
    joblib.dump(lin, str(save_folder / 'lin_model.p'))

    # # ===== training SVM models and evaluate =====
    print('\ntraining SVR model')
    svr_score = list()
    svr_mae = list()
    for i in range(4):
        select_col[0] = i
        svr[i].fit(train_x[:, select_col], train_y[:, i])

        svr_score.append(svr[i].score(val_x[:, select_col], val_y[:, i]))

        svr_val_y = svr[i].predict(val_x[:, select_col])
        svr_mae.append(math.degrees(mean_absolute_error(val_y[:, i], svr_val_y)))
        print("SVR model {} axis -- mae: {} -- R2: {}".format(i, svr_mae[i], svr_score[i]))

    # # save model
    joblib.dump(svr, str(save_folder / 'svr_model.p'))

    # # ===== training KNN models and evaluate =====
    print('\ntraining KNR model')
    knr_score = list()
    knr_mae = list()
    for i in range(4):
        select_col[0] = i
        knr[i].fit(train_x[:, select_col], train_y[:, i])

        knr_score.append(knr[i].score(val_x[:, select_col], val_y[:, i]))

        knr_val_y = knr[i].predict(val_x[:, select_col])
        knr_mae.append(math.degrees(mean_absolute_error(val_y[:, i], knr_val_y)))
        print("KNR model {} axis -- mae: {} -- R2: {}".format(i, knr_mae[i], knr_score[i]))

    joblib.dump(knr, str(save_folder / 'knr_model.p'))

    # # ===== train GPR models and evaluate =====
    # print('\ntraining GPR model')
    # gpr_score = list()
    # gpr_mae = list()
    # for i in range(4):
    #     select_col[0] = i
    #     gpr[i].fit(train_x[:, select_col], train_y[:, i])
    #
    #     gpr_score.append(gpr[i].score(val_x[:, select_col], val_y[:, i]))
    #
    #     gpr_val_y = gpr[i].predict(val_x[:, select_col])
    #     gpr_mae.append(math.degrees(mean_absolute_error(val_y[:, i], gpr_val_y)))
    #     print("GPR model {} axis -- mae: {} -- R2: {}".format(i, gpr_mae[i], gpr_score[i]))


if __name__ == "__main__":
    main()
