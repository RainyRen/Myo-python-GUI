# coding: utf-8
import pdb
import yaml
from pathlib import Path

# import numpy as np
# import pandas as pd
import tensorflow as tf

from utils.data_io import DataManager
from models import tf_models

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"
EXP_PATH = ROOT_PATH / "exp"

# # =============================


def train_tf(config):
    """

    :param dict config:
    :return:
    """
    # # obtain saving folder
    save_folder = EXP_PATH / config['exp_folder']

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

    model = multi2one_stft(config)



if __name__ == "__main__":
    # # ===== load config from config file =====
    with open(str(CONFIG_PATH / 'train_keras.yml'), 'r') as config_file:
        train_config = yaml.load(config_file)

    # # ===== training model =====
    train_tf(train_config)
    print("Finished training!")
