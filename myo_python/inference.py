# coding: utf-8
import pdb
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # ===== define global varibles =====
UPPER_ARM_LEN = 32      # # unit: cm
FOREARM_LEN = 33        # # unit: cm

# # ==================================


class EstimatorTF(object):
    def __init__(self):
        pass

    def predict(self):
        pass


def k_test(config):
    from utils.data_io import DataManager
    from keras.models import load_model

    data_mg = DataManager(
        './data/' + str(config['fs']) + 'hz',
        separate_rate=config['separate_rate'],
        time_length=config['time_length'],
        future_time=config['future_time'],
        one_target=True,
        degree2rad=config['degree2rad']
    )

    train_data, test_data = data_mg.get_all_data()
    ts_kinematic, ts_emg, ts_target = test_data

    pdb.set_trace()

    model = load_model(config['model_path'])
    result = model.predict([ts_kinematic, ts_emg], batch_size=128, verbose=1)

    x = [i * (1 / config['time_length']) for i in range(ts_target.shape[0])]
    # # ----- plot single axis -----
    plt.figure(0)
    plt.subplot(311)
    plt.plot(x, np.degrees(ts_target[:, 0]), 'k-')
    plt.plot(x, np.degrees(result[:, 0]), 'r--')
    plt.subplot(312)
    plt.plot(x, np.degrees(ts_target[:, 1]), 'k-')
    plt.plot(x, np.degrees(result[:, 1]), 'r--')
    plt.subplot(313)
    plt.plot(x, np.degrees(ts_target[:, 2]), 'k-')
    plt.plot(x, np.degrees(result[:, 2]), 'r--')

    # # ----- plot 3d space -----
    # # convert degree to radius
    if not config['degree2rad']:
        ts_target = np.radians(ts_target[200: 300])
        result = np.radians(result[200: 300])
    else:
        ts_target = ts_target[200: 300]
        result = result[200: 300]

    xy_gt = FOREARM_LEN * np.cos(ts_target[:, 1] + ts_target[:, 2]) + UPPER_ARM_LEN * np.cos(ts_target[:, 1])
    x_gt = xy_gt * np.cos(ts_target[:, 0])
    y_gt = xy_gt * np.sin(ts_target[:, 0])
    z_gt = FOREARM_LEN * np.sin(ts_target[:, 1] + ts_target[:, 2]) + UPPER_ARM_LEN * np.sin(ts_target[:, 1])

    xy_es = FOREARM_LEN * np.cos(result[:, 1] + result[:, 2]) + UPPER_ARM_LEN * np.cos(result[:, 1])
    x_es = xy_es * np.cos(result[:, 0])
    y_es = xy_es * np.sin(result[:, 0])
    z_es = FOREARM_LEN * np.sin(result[:, 1] + result[:, 2]) + UPPER_ARM_LEN * np.sin(result[:, 1])

    fig3d = plt.figure(1)
    ax = Axes3D(fig3d)

    ax.plot(x_gt, y_gt, z_gt)
    ax.plot(x_es, y_es, z_es)

    plt.show()


if __name__ == "__main__":
    """
    first we have to select model path
    """
    model_path = Path('./exp/multi2one_k')
    with open(model_path / 'config.yml') as config_file:
        test_config = yaml.load(config_file)
    test_config['model_path'] = str(model_path / 'rnn_best.h5')

    k_test(test_config)
