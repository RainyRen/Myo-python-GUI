# coding: utf-8
import pdb
import argparse
import yaml
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error

from utils.data_io import DataManager, angle2position

# # ===== define global variables =====
UPPER_ARM_LEN = 32      # # unit: cm
FOREARM_LEN = 33        # # unit: cm

MODEL_DIR_NAME = "JLW_stft_f4_dueling"
# # ==================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test_reg", help='test_reg or test_cls or test_trad or test_mann')
    parser.add_argument('--save_dir', default='./exp')
    parser.add_argument('--mann_dir', default='./exp/mann_s10_h2_f4')
    args = parser.parse_args()

    # # ===== load model config from saved config file =====
    if args.mode == 'test_reg':
        print('\ntest keras regression model -- ', MODEL_DIR_NAME)

        model_path = Path(args.save_dir) / MODEL_DIR_NAME
        with open(model_path / 'config.yml') as config_file:
            test_config = yaml.load(config_file)

        test_config['model_path'] = str(model_path / 'rnn_best.h5')
        test_reg(test_config)

    elif args.mode == 'test_cls':
        print('\ntest keras classification model')

        model_path = Path(args.save_dir) / MODEL_DIR_NAME
        with open(model_path / 'config.yml') as config_file:
            test_config = yaml.load(config_file)

        test_cls(test_config)

    elif args.mode == 'test_trad':
        print('\ntest traditional model -- ', MODEL_DIR_NAME)

        model_path = Path(args.save_dir) / MODEL_DIR_NAME
        with open(model_path / 'config.yml') as config_file:
            test_config = yaml.load(config_file)

        test_config['all_model_path'] = model_path
        test_trad(test_config)

    elif args.mode == 'test_mann':
        print('\ntest mann model -- ', args.mann_dir)

        model_path = Path(args.mann_dir)
        with open(model_path / 'config.yml') as config_file:
            test_config = yaml.load(config_file)

        test_mann(args, test_config)

    else:
        raise ValueError('No such mode, please check again')


# # ================================================ Keras ===========================================================
def test_reg(config):
    test_data_loader = DataManager(
        './data/' + str(config['fs']) + 'hz' + '/test',
        separate_rate=0,
        time_length=config['time_length'],
        future_time=config['future_time'],
        one_target=config['one_target'],
        degree2rad=config['degree2rad'],
        use_direction=False,
    )
    dt = 1 / config['fs']

    # train_data, test_data = data_mg.get_all_data()
    test_data, _ = test_data_loader.get_all_data(get_emg_raw=False, emg_3d=True)
    ts_kinematic, ts_emg, ts_target = test_data
    ts_orbit = ts_kinematic[:, -1, :4]

    model = load_model(config['model_path'])
    result = model.predict([ts_kinematic, ts_emg], batch_size=128, verbose=1)
    # result = model.predict_on_batch([ts_kinematic, ts_emg])

    if not config['one_target']:
        result = result[:, -1, :]
        ts_target = ts_target[:, -1, :]

    if config['degree2rad']:
        ts_orbit_degrees = np.degrees(ts_orbit)
        ts_target_degrees = np.degrees(ts_target)
        result_degrees = np.degrees(result)
    else:
        ts_orbit_degrees = ts_orbit
        ts_target_degrees = ts_target
        result_degrees = result

        ts_orbit = np.radians(ts_orbit)
        ts_target = np.radians(ts_target)
        result = np.radians(result)

    # # =========================== metrics for model ===================================
    _evaluate(ts_target_degrees, result_degrees, ts_orbit_degrees)
    pdb.set_trace()

    _plot_single_fig(ts_target_degrees[-800:-400], result_degrees[-800:-400], ts_orbit_degrees[-800:-400], dt)
    _plot_3d_fig(ts_target[200:260], result[200:260], ts_orbit[200:260])


def test_cls(config):
    from sklearn.metrics import accuracy_score, average_precision_score
    from sklearn.metrics import f1_score as f1score

    from models.k_models import f1_score

    test_data_loader = DataManager(
        './data/' + str(config['fs']) + 'hz' + '/test',
        separate_rate=0,
        time_length=config['time_length'],
        future_time=config['future_time'],
        one_target=config['one_target'],
        degree2rad=config['degree2rad'],
        use_direction=True,
    )
    # dt = 1 / config['fs']

    test_data, _ = test_data_loader.get_all_data()
    _, ts_emg, ts_target = test_data

    model = load_model(config['model_path'], custom_objects={'f1_score': f1_score})
    result = model.predict(ts_emg, batch_size=128, verbose=1)

    result = np.where(result > 0.5, 1, 0)

    acc = accuracy_score(ts_target, result)
    avg_prec = average_precision_score(ts_target, result)
    f1 = f1score(ts_target, result, average='macro')

    print('accuracy: ', acc)
    print('average precision: ', avg_prec)
    print('f1 score:', f1)


def test_trad(config):
    from sklearn.externals import joblib

    test_data_loader = DataManager(
        './data/' + str(config['fs']) + 'hz' + '/test',
        separate_rate=0,
        time_length=config['time_length'],
        future_time=config['future_time'],
        one_target=config['one_target'],
        degree2rad=config['degree2rad'],
        use_direction=False,
    )
    dt = 1 / config['fs']

    test_data, _ = test_data_loader.get_all_data(get_emg_raw=True, emg_3d=False)
    ts_kinematic, ts_emg, ts_target = test_data

    test_x = np.hstack((ts_kinematic[:, -1, :], ts_emg[:, -1, :]))
    ts_orbit = ts_kinematic[:, -1, :4]

    # # ===== load model =====
    lin = joblib.load(str(config['all_model_path'] / 'lin_model.p'))

    select_col = list(range(3, test_x.shape[-1]))
    result = list()
    for i in range(4):
        select_col[0] = i
        result.append(lin[i].predict(test_x[:, select_col]))
    result = np.asarray(result).T

    if config['degree2rad']:
        ts_orbit_degrees = np.degrees(ts_orbit)
        ts_target_degrees = np.degrees(ts_target)
        result_degrees = np.degrees(result)
    else:
        ts_orbit_degrees = ts_orbit
        ts_target_degrees = ts_target
        result_degrees = result

        ts_orbit = np.radians(ts_orbit)
        ts_target = np.radians(ts_target)
        result = np.radians(result)

    _evaluate(ts_target_degrees, result_degrees, ts_orbit_degrees)
    pdb.set_trace()

    _plot_single_fig(ts_target_degrees[-800:-400], result_degrees[-800:-400], ts_orbit_degrees[-800:-400], dt=dt)
    _plot_3d_fig(ts_target[200:260], result[200:260], ts_orbit[200:260])


def test_mann(args, config):
    from tqdm import tqdm

    frozen_model_path = Path(args.mann_dir) / 'frozen_model2.pb'
    graph = _load_graph(str(frozen_model_path))

    # # We can verify that we can access the list of operations in the graph
    # # for op in graph.get_operations():
    # #     print(op.name)
    # # prefix/Placeholder/inputs_placeholder
    # # ...
    # # prefix/Accuracy/predictions

    # # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    x_label = graph.get_tensor_by_name('prefix/Placeholder_1:0')
    prediction = graph.get_tensor_by_name('prefix/output:0')

    # # load test data
    test_data_loader = DataManager(
        './data/20hz/test',
        separate_rate=0.,
        time_length=config['seq_length'],
        future_time=config['future_time'],
        one_target=False,
        degree2rad=config['degree2rad'],
        use_direction=False
    )
    dt = 1 / 20
    print("organising materials...\n")
    test_data, _ = test_data_loader.get_all_data(get_emg_raw=config['emg_raw'], emg_3d=False)
    ts_kinematic, ts_emg, ts_target = test_data
    ts_orbit = ts_kinematic[:, -1, :4]

    batch_size = 1  # config['batch_size']
    n_batch = ts_target.shape[0] // batch_size
    total_ts_num = n_batch * batch_size

    ts_orbit = ts_orbit[:total_ts_num]
    ts_target = ts_target[:total_ts_num]

    ts_x = np.concatenate((ts_kinematic, ts_emg), axis=-1)
    ts_x_label = np.concatenate([np.zeros(shape=[ts_target.shape[0], 1, 4]), ts_target[:, :-1, :]], axis=1)

    # # We launch a Session
    with tf.Session(graph=graph) as sess:
        # # Note: we don't nee to initialize/restore anything
        # # There is no Variables in this graph, only hardcoded constants
        output = []
        for i in tqdm(range(n_batch), ncols=60):
            s, e = i * batch_size, (i + 1) * batch_size
            feed_dict = {x: ts_x[s:e], x_label: ts_x_label[s:e]}
            output.append(sess.run(prediction, feed_dict=feed_dict))

    result = np.concatenate(output, axis=0)
    result = result[:, -1, :]
    ts_target = ts_target[:, -1, :]

    ts_orbit_degrees = np.degrees(ts_orbit)
    ts_target_degrees = np.degrees(ts_target)
    result_degrees = np.degrees(result)

    _evaluate(ts_target_degrees, result_degrees, ts_orbit_degrees)

    pdb.set_trace()

    _plot_single_fig(ts_target_degrees[-800:-400], result_degrees[-800:-400], ts_orbit_degrees[-800:-400], dt=dt)
    _plot_3d_fig(ts_target[200:260], result[200:260], ts_orbit[200:260])


def _evaluate(target_deg, estimate_deg, orbit_deg):
    r2 = r2_score(target_deg, estimate_deg)
    r2_2 = r2_score(target_deg[:, 0], estimate_deg[:, 0])
    r2_5 = r2_score(target_deg[:, 1], estimate_deg[:, 1])
    r2_6 = r2_score(target_deg[:, 2], estimate_deg[:, 2])
    r2_7 = r2_score(target_deg[:, 3], estimate_deg[:, 3])
    print("\n--------- r2 score ---------")
    print("r2 all: {:.2f}".format(r2))
    print("r2 for 2 axis: {:.2f}, r2 for 5 axis: {:.2f}, r2 for 6 axis {:.2f}, r2 for 7 axis {:.2f}"
          .format(r2_2, r2_5, r2_6, r2_7))
    print("----------------------------")

    mae = mean_absolute_error(target_deg, estimate_deg)
    mae_single = map(mean_absolute_error, target_deg.T, estimate_deg.T)
    print("\n-------- mean absolute error in degrees---------")
    print("mae all: {:.2f}".format(mae))
    print("2 axis mae: {:.2f}, 5 axis mae: {:.2f}, 6 axis mae: {:.2f}, 7 axis mae {:.2f}".format(*mae_single))
    print("------------------------------------------------")

    fake_result_degrees = orbit_deg
    mae_fake = mean_absolute_error(target_deg, fake_result_degrees)
    mae_fake_single = map(mean_absolute_error, target_deg.T, fake_result_degrees.T)
    print("\n-------- fake result mean absolute error in degrees ---------")
    print("mae all: {:.2f}".format(mae_fake))
    print("2 axis mae: {:.2f}, 5 axis mae: {:.2f}, 6 axis mae: {:.2f}, 7 axis mae {:.2f}".format(*mae_fake_single))
    print("-------------------------------------------------------------")


def _plot_single_fig(target_deg, estimate_deg, orbit_deg, dt=0.05):
    # show_interval = 200
    show_interval = target_deg.shape[0]
    x = [i * dt for i in range(show_interval)]
    # # ----- plot single axis -----
    plt.figure(0)
    plt.subplot(411)
    plt.plot(x, orbit_deg[:show_interval, 0], 'g-', label='current')
    plt.plot(x, target_deg[:show_interval, 0], 'k-', label='target')
    plt.plot(x, estimate_deg[:show_interval, 0], 'r--', label='predict')
    plt.legend(loc=0)
    plt.ylabel('degree')
    plt.subplot(412)
    plt.plot(x, orbit_deg[:show_interval, 1], 'g-', label='current')
    plt.plot(x, target_deg[:show_interval, 1], 'k-', label='target')
    plt.plot(x, estimate_deg[:show_interval, 1], 'r--', label='predict')
    plt.legend(loc=0)
    plt.ylabel('degree')
    plt.subplot(413)
    plt.plot(x, orbit_deg[:show_interval, 2], 'g-', label='current')
    plt.plot(x, target_deg[:show_interval, 2], 'k-', label='target')
    plt.plot(x, estimate_deg[:show_interval, 2], 'r--', label='predict')
    plt.legend(loc=0)
    plt.ylabel('degree')
    plt.subplot(414)
    plt.plot(x, orbit_deg[:show_interval, 3], 'g-', label='current')
    plt.plot(x, target_deg[:show_interval, 3], 'k-', label='target')
    plt.plot(x, estimate_deg[:show_interval, 3], 'r--', label='predict')
    plt.legend(loc=0)
    plt.xlabel('time')
    plt.ylabel('degree')


def _plot_3d_fig(target_rad, estimate_rad, orbit_rad):
    x_gt, y_gt, z_gt = angle2position(target_rad)
    x_es, y_es, z_es = angle2position(estimate_rad)
    x_or, y_or, z_or = angle2position(orbit_rad)

    fig3d = plt.figure(1)
    ax = Axes3D(fig3d)

    ax.plot(x_or, y_or, z_or, 'g-', label='current')
    ax.plot(x_gt, y_gt, z_gt, 'b-', label='target')
    ax.plot(x_es, y_es, z_es, 'r--', label='predict')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')

    plt.legend(loc=0)
    plt.show()


def _load_graph(frozen_graph_filename):
    # # We load the protobuf file from the disk and parse it to retrieve the
    # # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # # The name var will prefix every op/nodes in your graph
        # # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == "__main__":
    main()
