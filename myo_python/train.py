# coding: utf-8
import pdb
import argparse
import yaml
from pathlib import Path

# import numpy as np
from keras.callbacks import ModelCheckpoint

from utils import data_io
from models import k_models

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"

# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train_reg", help='train_reg or train_cls')
    parser.add_argument('--save_dir', default='./exp')
    args = parser.parse_args()

    # # ===== load config from config file =====
    with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
        train_config = yaml.load(config_file)

    if args.mode == 'train_reg':
        train_reg(args, train_config)

    elif args.mode == 'test_cls':
        test_cls(args, train_config)


def train_reg(args, train_config):
    """
    :param tuple train_data: tuple contain three type of training data
    :param tuple validation_data:
    :param dict config:
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

    # # check folder exist, if not creat new one
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

    model.fit(
        x=[tr_kinematic, tr_emg], y=tr_target,
        validation_data=([val_kinematic, val_emg], val_target),
        batch_size=train_config['batch_size'],
        epochs=train_config['epochs'],
        callbacks=[checkpoint],
        shuffle=False
    )


def train_cls(args, train_config):
    save_folder = Path(args.save_dir) / train_config['exp_folder']

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
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


if __name__ == "__main__":
    main()
