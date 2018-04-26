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

# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--label_type', default="one_hot", help='one_hot or five_hot')
    parser.add_argument('--n_classes', default=5)
    parser.add_argument('--seq_length', default=20)
    parser.add_argument('--future_time', default=1)
    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', default=4)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--rnn_size', default=200)
    parser.add_argument('--image_width', default=20)
    parser.add_argument('--image_height', default=20)
    parser.add_argument('--rnn_num_layers', default=1)
    parser.add_argument('--memory_size', default=128)
    parser.add_argument('--memory_vector_dim', default=40)
    parser.add_argument('--shift_range', default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')
    parser.add_argument('--save_dir', default='./exp/mann')
    parser.add_argument('--tensorboard_dir', default='./summary/one_shot_learning')
    args = parser.parse_args()

    # # ===== load config from config file =====
    print('training tf regression model')
    train_tf_reg(args)


def train_tf_reg(args, config):
    """
    :param args:
    :param dict config:
    :return:
    """
    # # ===== obtain saving folder =====
    save_folder = Path(args.save_dir) / train_config['exp_folder']

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
        separate_rate=0.,
        time_length=args.seq_length,
        future_time=args.future_time,
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

    # # ===== obtain model =====


if __name__ == "__main__":
    main()
