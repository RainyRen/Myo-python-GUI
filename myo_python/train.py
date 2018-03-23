# coding: utf-8
import pdb
import yaml
from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from utils.data_io import DataManager
from models.post_fusion import basic_model

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"
EXP_PATH = ROOT_PATH / "exp"

# # =============================


def train(train_data, validation_data, config):
    """
    :param tuple train_data: tuple contain three type of training data
    :param tuple validation_data:
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
    # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(config, write_file, default_flow_style=False)

    tr_kinematic, tr_emg, tr_target = train_data
    val_kinematice, val_emg, val_target = validation_data
    model = basic_model(config)

    checkpoint = ModelCheckpoint(
        filepath=str(save_folder / 'rnn_best.h5'),
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_acc',
        mode='max')

    model.fit(
        x=[tr_kinematic, tr_emg], y=tr_target,
        validation_data=([val_kinematice, val_emg], val_target),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=[checkpoint],
        shuffle=True
    )


if __name__ == "__main__":
    with open(str(CONFIG_PATH / 'train_default.yml'), 'r') as config_file:
        config = yaml.load(config_file)

    data_mg = DataManager('./data/20hz', time_length=config['time_length'])
    print("organising materials...")
    tr_data, val_data = data_mg.get_all_data()
    pdb.set_trace()
    train(tr_data, val_data, config)
    print("Finished training!")
