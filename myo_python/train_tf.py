# coding: utf-8
import pdb
import yaml
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.metrics import mean_absolute_error

from utils import data_io
from models.tf_models import NTMOneShotLearningModel

# # ===== global parameters =====
ROOT_PATH = Path(__file__).parent
CONFIG_PATH = ROOT_PATH / "config"

# # =============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train", help='train or extract')
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--seq_length', default=10)
    parser.add_argument('--future_time', default=4)
    parser.add_argument('--degree2rad', default=True)
    parser.add_argument('--emg_raw', default=True)
    parser.add_argument('--one_target', default=True)
    parser.add_argument('--model', default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', default=2)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_epoches', default=300000)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--rnn_size', default=32)
    parser.add_argument('--rnn_num_layers', default=1)
    parser.add_argument('--memory_size', default=256)
    parser.add_argument('--memory_vector_dim', default=32)
    parser.add_argument('--shift_range', default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')
    parser.add_argument('--save_dir', default='./exp/mann_s10_h2_f4')
    parser.add_argument('--tensorboard_dir', default='./exp/mann_s10_h2_f4/summary')
    parser.add_argument('--extract_dir', default='./exp/mann_s10_h2_f4/MANN')
    args = parser.parse_args()

    # # ===== determine mode =====
    if args.mode == 'train':
        print('training tf regression model')
        train_tf_reg(args)

    elif args.mode == 'extract':
        print('freeze {} model'.format(args.extract_dir))
        # freeze_graph(args.extract_dir)
        freeze_graph2(Path(args.extract_dir))

    else:
        raise ValueError('No such mode')


def train_tf_reg(args):
    """
    :param args:
    :return:
    """
    # # ===== obtain saving folder =====
    save_folder = Path(args.save_dir)

    # # ===== obtain model =====
    args.output_dim = 4
    model = NTMOneShotLearningModel(args)

    usefule_label_num = args.seq_length - args.future_time

    # # ===== get pre-processed data =====
    train_data_loader = data_io.DataManager(
        './data/20hz/train',
        separate_rate=0.,
        time_length=args.seq_length,
        future_time=args.future_time,
        one_target=args.one_target,
        degree2rad=args.degree2rad,
        use_direction=False
    )
    val_data_loader = data_io.DataManager(
        './data/20hz/val',
        separate_rate=0.,
        time_length=args.seq_length,
        future_time=args.future_time,
        one_target=args.one_target,
        degree2rad=args.degree2rad,
        use_direction=False
    )

    print("organising materials...\n")
    tr_data_generator = train_data_loader.data_generator(
        get_emg_raw=args.emg_raw, emg_3d=False, batch_size=args.batch_size, shuffle=True
    )
    val_data_generator = val_data_loader.data_generator(
        get_emg_raw=args.emg_raw, emg_3d=False, batch_size=args.batch_size, shuffle=True
    )

    # # check folder exist, if not create new one
    data_io.folder_check(save_folder)
    # # save training config to file
    with open(str(save_folder / 'config.yml'), 'w') as write_file:
        yaml.dump(vars(args), write_file, default_flow_style=False)

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
            tr_data, _ = next(tr_data_generator)
            tr_kinematic, tr_emg, tr_target = tr_data
            x = np.concatenate((tr_kinematic, tr_emg), axis=-1)
            x_label = np.zeros((args.batch_size, args.seq_length, 4))
            x_label[:, 1:usefule_label_num + 1, :] = tr_kinematic[:, -usefule_label_num:, :4]
            # x_label = np.concatenate([np.zeros((args.batch_size, 1, 4)), tr_target[:, :-1, :]], axis=1)
            y = tr_target

            feed_dict = {model.x: x, model.x_label: x_label, model.y: y}
            sess.run(model.train_op, feed_dict=feed_dict)

            # # --------------------------------
            # # ---------- Test ---------
            if b % 500 == 0:
                val_data, _ = next(val_data_generator)
                val_kinematic, val_emg, val_target = val_data
                x_val = np.concatenate((val_kinematic, val_emg), axis=-1)
                x_label_val = np.zeros((args.batch_size, args.seq_length, 4))
                x_label_val[:, 1:usefule_label_num + 1, :] = val_kinematic[:, -usefule_label_num:, :4]
                # x_label_val = np.concatenate([np.zeros((args.batch_size, 1, 4)), val_target[:, :-1, :]], axis=1)
                y_val = val_target

                feed_dict = {model.x: x_val, model.x_label: x_label_val, model.y: y_val}
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)

                if args.one_target:
                    mae = mean_absolute_error(y_val[:, :], output[:, :])
                else:
                    mae = mean_absolute_error(y_val[:, -1, :], output[:, -1, :])
                print('episode: {} -- loss: {} -- mae: {}'.format(b, learning_loss, mae))

            # # --------------------------------
            # # ---------- Save model ----------
            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # # --------------------------------

    print('Finished Training!')


def freeze_graph(model_dir, output_node_names='output'):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    :param model_dir: the root folder containing the checkpoint state file
    :param output_node_names: a string, containing all the output node's names, comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # # We restore the weights
        saver.restore(sess, input_checkpoint)

        # # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )

        # # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def freeze_graph2(model_dir, output_node_names='output'):
    """
    change mode to batch size 1 for real-time inference
    :param model_dir:
    :param output_node_names:
    :return:
    """
    from collections import namedtuple

    with open(model_dir.parent / 'config.yml', 'r') as config_file:
        config = yaml.load(config_file)

    config['batch_size'] = 1
    model_args = namedtuple('args', config.keys())(*config.values())

    # # ===== load model =====
    model = NTMOneShotLearningModel(model_args)
    saver = tf.train.Saver()

    output_node_names = 'output'
    output_graph = str(model_dir.parent / "frozen_model2.pb")

    with tf.Session() as sess:
        saver.restore(sess, str(model_dir / 'model.tfmodel-295000'))

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names.split(",")
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()
