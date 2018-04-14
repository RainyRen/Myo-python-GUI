# coding: utf-8
import pdb
import yaml
from pathlib import Path

# import numpy as np
# import pandas as pd
from keras.callbacks import ModelCheckpoint

from utils.data_io import DataManager
from models import k_models

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"
EXP_PATH = ROOT_PATH / "exp"

# # =============================


def train(train_data, validation_data, config):
    """
    :param tuple train_data: tuple contain three type of training data
    :param tuple validation_data:
    :param dict config:
    """
    save_folder = EXP_PATH / config['exp_folder']

    if save_folder.exists():
        user_input = input('folder {} already exist, do you want to overrider? [y/n] '
                           .format(save_folder.stem))
        if user_input in ('n', 'no', 'N', 'No'):
            print('please reset your arguments')
            exit(0)
    else:
        print('create new folder {}'.format(save_folder.stem))
        save_folder.mkdir()
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(config, write_file, default_flow_style=False)

    tr_kinematic, tr_emg, tr_target = train_data
    val_kinematic, val_emg, val_target = validation_data
    model = k_models.multi2one_stft2(config)

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
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=[checkpoint],
        shuffle=False
    )


if __name__ == "__main__":
    # # ===== load config from config file =====
    with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
        train_config = yaml.load(config_file)

    # # ===== get pre-processed data =====
    data_mg = DataManager(
        './data/' + str(train_config['fs']) + 'hz',
        separate_rate=train_config['separate_rate'],
        time_length=train_config['time_length'],
        future_time=train_config['future_time'],
        one_target=train_config['one_target'],
        degree2rad=train_config['degree2rad'],
        use_direction=train_config['use_direction']
    )
    print("organising materials...\n")
    tr_data, val_data = data_mg.get_all_data()
    pdb.set_trace()

    # # ===== training model =====
    train(tr_data, val_data, train_config)
    print("Finished training!")
