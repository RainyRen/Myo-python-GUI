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

from utils.data_io import DataManager

# # ===== define global varibles =====
UPPER_ARM_LEN = 32      # # unit: cm
FOREARM_LEN = 33        # # unit: cm

# # ==================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test_reg", help='test_reg or test_cls')
    parser.add_argument('--save_dir', default='./exp')
    args = parser.parse_args()

    # # ===== load model config from saved config file =====
    model_path = Path(args.save_dir) / 'multi2one_convlstm_stft_k'

    with open(model_path / 'config.yml') as config_file:
        test_config = yaml.load(config_file)
    test_config['model_path'] = str(model_path / 'rnn_best.h5')

    if args.mode == 'test_reg':
        test_reg(test_config)

    elif args.mode == 'test_cls':
        test_cls(test_config)

    else:
        raise ValueError('No such mode, please check again')


# # ================================================ Keras ===========================================================
def test_reg(config):
    from sklearn.metrics import r2_score, mean_absolute_error

    from utils.data_io import angle2position

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
    r2 = r2_score(ts_target, result)
    r2_2 = r2_score(ts_target[:, 0], result[:, 0])
    r2_5 = r2_score(ts_target[:, 1], result[:, 1])
    r2_6 = r2_score(ts_target[:, 2], result[:, 2])
    r2_7 = r2_score(ts_target[:, 3], result[:, 3])
    print("--------- r2 score ---------\n")
    print("r2 all: {:.2f}".format(r2))
    print("r2 for 2 axis: {:.2f}, r2 for 5 axis: {:.2f}, r2 for 6 axis {:.2f}, r2 for 7 axis {:.2f}"
          .format(r2_2, r2_5, r2_6, r2_7))
    print("----------------------------")

    mae = mean_absolute_error(ts_target_degrees, result_degrees)
    mae_single = map(mean_absolute_error, ts_target_degrees.T, result_degrees.T)
    print("-------- mean absolute error in degrees---------\n")
    print("mae all: {:.2f}".format(mae))
    print("2 axis mae: {:.2f}, 5 axis mae: {:.2f}, 6 axis mae: {:.2f}, 7 axis mae {:.2f}".format(*mae_single))
    print("------------------------------------------------")

    fake_result_degrees = ts_orbit_degrees
    mae_fake = mean_absolute_error(ts_target_degrees, fake_result_degrees)
    mae_fake_single = map(mean_absolute_error, ts_target_degrees.T, fake_result_degrees.T)
    print("-------- fake result mean absolute error in degrees ---------\n")
    print("mae all: {:.2f}".format(mae_fake))
    print("2 axis mae: {:.2f}, 5 axis mae: {:.2f}, 6 axis mae: {:.2f}, 7 axis mae {:.2f}".format(*mae_fake_single))
    print("-------------------------------------------------------------")

    pdb.set_trace()
    # # =========================================================================================
    # # ============================== plot fig for each angle ==================================
    x = [i * (1 / config['time_length']) for i in range(ts_target.shape[0])]
    # # ----- plot single axis -----
    plt.figure(0)
    plt.subplot(411)
    plt.plot(x, ts_orbit_degrees[:, 0], 'g-')
    plt.plot(x, ts_target_degrees[:, 0], 'k-')
    plt.plot(x, result_degrees[:, 0], 'r--')
    plt.subplot(412)
    plt.plot(x, ts_orbit_degrees[:, 1], 'g-')
    plt.plot(x, ts_target_degrees[:, 1], 'k-')
    plt.plot(x, result_degrees[:, 1], 'r--')
    plt.subplot(413)
    plt.plot(x, ts_orbit_degrees[:, 2], 'g-')
    plt.plot(x, ts_target_degrees[:, 2], 'k-')
    plt.plot(x, result_degrees[:, 2], 'r--')
    plt.subplot(414)
    plt.plot(x, ts_orbit_degrees[:, 3], 'g-')
    plt.plot(x, ts_target_degrees[:, 3], 'k-')
    plt.plot(x, result_degrees[:, 3], 'r--')

    # # ----- plot 3d space -----
    # # convert degree to radius

    x_or, y_or, z_or = angle2position(ts_orbit[200:260])
    x_gt, y_gt, z_gt = angle2position(ts_target[200:260])
    x_es, y_es, z_es = angle2position(result[200:260])

    fig3d = plt.figure(1)
    ax = Axes3D(fig3d)

    ax.plot(x_or, y_or, z_or, 'g-')
    ax.plot(x_gt, y_gt, z_gt, 'b-')
    ax.plot(x_es, y_es, z_es, 'r--')

    plt.show()


def test_cls(config):
    from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
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


if __name__ == "__main__":
    main()
