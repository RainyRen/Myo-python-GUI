# coding: utf-8
import pdb
import yaml
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from utils.data_io import DataManager
from models.tf_models import NTMOneShotLearningModel

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"

# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--seq_length', default=20)
    parser.add_argument('--future_time', default=1)
    parser.add_argument('--degree2rad', default=True)
    parser.add_argument('--emg_raw', default=False)
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
    parser.add_argument('--tensorboard_dir', default='./exp/mann/summary')
    args = parser.parse_args()

    # # ===== load config from config file =====
    print('training tf regression model')
    train_tf_reg(args)


def train_tf_reg(args):
    """
    :param args:
    :return:
    """
    # # ===== obtain saving folder =====
    save_folder = Path(args.save_dir)

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/train',
        separate_rate=0.,
        time_length=args.seq_length,
        future_time=args.future_time,
        one_target=False,
        degree2rad=args.degree2rad,
        use_direction=False
    )
    val_data_loader = data_io.DataManager(
        './data/' + str(train_config['fs']) + 'hz' + '/val',
        separate_rate=0.,
        time_length=args.seq_length,
        future_time=args.future_time,
        one_target=False,
        degree2rad=args.degree2rad,
        use_direction=False
    )

    print("organising materials...\n")
    tr_data_generator = train_data_loader.data_generator(
        get_emg_raw=args.emg_raw, emg_3d=False, batch_size=args.batch_size, shuffle=False
    )
    val_data_generator = val_data_loader.data_generator(
        get_emg_raw=args.emg_raw, emg_3d=False, batch_size=args.batch_size, shuffle=False
    )

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(vars(args), write_file, default_flow_style=False)

    # # ===== obtain model =====
    model = NTMOneShotLearningModel(args)

    with tf.Session() as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        print('\n', args, '\n')

        for b in range(args.num_epoches):
            # # ---------- Train ----------
            tr_kinematic, tr_emg, tr_target = next(tr_data_generator)
            x = np.concatenate((tr_kinematic, tr_emg), axis=-1)
            x_label = np.concatenate([np.zeros(shape=[batch_size, 1, 4]), tr_target[:, :-1, :]], axis=1)
            y = tr_target

            feed_dict = {model.x: , model.x_label: x_label, model.y: y}
            sess.run(model.train_op, feed_dict=feed_dict)

            # # --------------------------------
            # # ---------- Test ---------
            if b % 100 == 0:
                val_kinematic, val_emg, val_target = next(val_data_generator)
                x = np.concatenate((tr_kinematic, tr_emg), axis=-1)
                x_label = np.concatenate([np.zeros(shape=[batch_size, 1, 4]), tr_target[:, :-1, :]], axis=1)
                y = tr_target

                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)

            # # --------------------------------
            # # ---------- Save model ----------
            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # # --------------------------------

    print('Finished Training!')


if __name__ == "__main__":
    main()
