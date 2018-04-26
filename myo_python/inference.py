# coding: utf-8
import pdb
import argparse
import yaml
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error

from utils.data_io import DataManager, angle2position

# # ===== define global varibles =====
UPPER_ARM_LEN = 32      # # unit: cm
FOREARM_LEN = 33        # # unit: cm

MODEL_DIR_NAME = "traditional_f4"
# # ==================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test_reg", help='test_reg or test_cls or test_trad')
    parser.add_argument('--save_dir', default='./exp')
    args = parser.parse_args()

    # # ===== load model config from saved config file =====
    model_path = Path(args.save_dir) / MODEL_DIR_NAME

    with open(model_path / 'config.yml') as config_file:
        test_config = yaml.load(config_file)

    if args.mode == 'test_reg':
        test_config['model_path'] = str(model_path / 'rnn_best.h5')
        test_reg(test_config)

    elif args.mode == 'test_cls':
        test_cls(test_config)

    elif args.mode == 'test_trad':
        test_config['all_model_path'] = model_path
        test_trad(test_config)

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
    test_data, _ = test_data_loader.get_all_data()
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

    _plot_single_fig(ts_target_degrees, result_degrees, ts_orbit_degrees, dt)
    _plot_3d_fig(ts_target, result, ts_orbit)


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

    _plot_single_fig(ts_target_degrees, result_degrees, ts_orbit_degrees, dt=dt)
    _plot_3d_fig(ts_target, result, ts_orbit)


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
    plt.plot(x, orbit_deg[:show_interval, 0], 'g-')
    plt.plot(x, target_deg[:show_interval, 0], 'k-')
    plt.plot(x, estimate_deg[:show_interval, 0], 'r--')
    plt.subplot(412)
    plt.plot(x, orbit_deg[:show_interval, 1], 'g-')
    plt.plot(x, target_deg[:show_interval, 1], 'k-')
    plt.plot(x, estimate_deg[:show_interval, 1], 'r--')
    plt.subplot(413)
    plt.plot(x, orbit_deg[:show_interval, 2], 'g-')
    plt.plot(x, target_deg[:show_interval, 2], 'k-')
    plt.plot(x, estimate_deg[:show_interval, 2], 'r--')
    plt.subplot(414)
    plt.plot(x, orbit_deg[:show_interval, 3], 'g-')
    plt.plot(x, target_deg[:show_interval, 3], 'k-')
    plt.plot(x, estimate_deg[:show_interval, 3], 'r--')


def _plot_3d_fig(target_rad, estimate_rad, orbit_rad):
    x_gt, y_gt, z_gt = angle2position(target_rad[200:260])
    x_es, y_es, z_es = angle2position(estimate_rad[200:260])
    x_or, y_or, z_or = angle2position(orbit_rad[200:260])

    fig3d = plt.figure(1)
    ax = Axes3D(fig3d)

    ax.plot(x_or, y_or, z_or, 'g-')
    ax.plot(x_gt, y_gt, z_gt, 'b-')
    ax.plot(x_es, y_es, z_es, 'r--')

    plt.show()


if __name__ == "__main__":
    main()
