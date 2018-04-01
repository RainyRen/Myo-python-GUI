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
    from utils.data_io import DataManager, degree2position
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
    plt.subplot(411)
    plt.plot(x, np.degrees(ts_target[:, 0]), 'k-')
    plt.plot(x, np.degrees(result[:, 0]), 'r--')
    plt.subplot(412)
    plt.plot(x, np.degrees(ts_target[:, 1]), 'k-')
    plt.plot(x, np.degrees(result[:, 1]), 'r--')
    plt.subplot(413)
    plt.plot(x, np.degrees(ts_target[:, 2]), 'k-')
    plt.plot(x, np.degrees(result[:, 2]), 'r--')
    plt.subplot(414)
    plt.plot(x, np.degrees(ts_target[:, 3]), 'k-')
    plt.plot(x, np.degrees(result[:, 3]), 'r--')

    # # ----- plot 3d space -----
    # # convert degree to radius
    if not config['degree2rad']:
        ts_target = np.radians(ts_target[200: 300])
        result = np.radians(result[200: 300])
    else:
        ts_target = ts_target[200: 300]
        result = result[200: 300]

    x_gt, y_gt, z_gt = degree2position(ts_target)
    x_es, y_es, z_es = degree2position(result)

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
