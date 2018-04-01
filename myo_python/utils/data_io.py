# coding: utf-8
from pathlib import Path
from itertools import cycle

import pandas as pd
import numpy as np


class DataManager(object):
    def __init__(self, file_path,
                 separate_rate=0.2, time_length=15, future_time=1,
                 one_target=True, degree2rad=False):
        """
        data pre-processing, sort out to the format we needed
        :param file_path:
        :param separate_rate:
        :param time_length:
        :param future_time:
        :param one_target:
        """
        self.file_path = Path(file_path)
        self.time_length = time_length
        self.future_time = future_time
        self.one_target = one_target
        self.degree2rad = degree2rad

        self.tr_kinematic = list()
        self.tr_emg = list()
        self.tr_target = list()

        self.val_kinematic = list()
        self.val_emg = list()
        self.val_target = list()

        self.separate_rate = separate_rate
        all_file_list = list(self.file_path.glob('jl_*.csv'))
        print("found files: ", all_file_list)

        all_file_num = len(all_file_list)
        val_file_num = int(all_file_num * separate_rate)
        self.tr_file_list = all_file_list[val_file_num:]
        self.val_file_list = all_file_list[:val_file_num]

    def get_all_data(self):
        for current_file in self.tr_file_list:
            kinematic, emg, target = self._load_data(current_file)
            self.tr_kinematic.append(kinematic)
            self.tr_emg.append(emg)
            self.tr_target.append(target)

        for current_file in self.val_file_list:
            kinematic, emg, target = self._load_data(current_file)
            self.val_kinematic.append(kinematic)
            self.val_emg.append(emg)
            self.val_target.append(target)

        tr = list(map(lambda x: np.concatenate(x, axis=0), [self.tr_kinematic, self.tr_emg, self.tr_target]))
        val = list(map(lambda x: np.concatenate(x, axis=0), [self.val_kinematic, self.val_emg, self.val_target]))\
            if self.separate_rate > 0 else None
                
        if self.degree2rad:
            tr = list(map(np.radians, tr))
            if val:
                val = list(map(np.radians, val))

        return tr, val

    def data_generator(self, batch_size=32):
        tr_file_cycle = cycle(self.tr_file_list)

        while True:
            current_file = next(tr_file_cycle)
            kinematic, emg, target = self._load_data(current_file)

            n_batch = target.shape[0] // batch_size

    def _load_data(self, file_path):
        arm_angle_samples = list()
        gyro_samples = list()
        acc_samples = list()
        emg_samples = list()
        target_samples = list()

        data_df = pd.read_csv(file_path, skiprows=1, header=None)
        n_sample = data_df.shape[0] - self.time_length

        # samples_pool = [(i * self.time_length, (i + 1) * self.time_length) for i in range(n_sample)]

        arm_angle = data_df.iloc[:, :4].values
        gyro = data_df.iloc[:, 10:16].values
        acc = data_df.iloc[:, 16:22].values
        emg = data_df.iloc[:, 22:38].values

        for i in range(n_sample):
            arm_angle_samples.append(arm_angle[i:i + self.time_length])
            gyro_samples.append(gyro[i:i + self.time_length])
            acc_samples.append(acc[i:i + self.time_length])
            emg_samples.append(emg[i:i + self.time_length])
            if self.one_target:
                target_samples.append(arm_angle[i + self.time_length + self.future_time - 1])
            else:
                # # get multi output target
                target_samples.append(
                    arm_angle[i + self.future_time: i + self.time_length + self.future_time]
                )

        arm_angle_samples = np.asarray(arm_angle_samples)
        gyro_samples = np.asarray(gyro_samples)
        acc_samples = np.asarray(acc_samples)

        kinematic_samples = np.concatenate((arm_angle_samples, gyro_samples, acc_samples), axis=-1)
        emg_samples = np.asarray(emg_samples)
        target_samples = np.asarray(target_samples)

        return kinematic_samples, emg_samples, target_samples


def degree2position(arm_angle, forearm_len=33, upper_arm_len=32):
    """
    convert arm angle to hand position in 3D space
    a1: forearm, a2: upper arm
    x = a1 * cos(5+7) * cos(2) - a1 * sin(7) * sin(6) * cos(2) + a2 * cos(5) * cos(2)
    y =  a1 * ocs(5+7) * sin(2) - a1 * sin(7) * sin(6) * sin(2) + a2 * cos(5) * sin(2)
    z = a1 * sin(5+7) * cos(6) + a2 * sin(5)
    :param np.array arm_angle: n x 4 dim, for 2, 5, ,6 ,7 angle in exoskeleton present in rad! rad! rad!
    :param float forearm_len: forearm length for the subject
    :param float upper_arm_len: upper arm length for the subject
    :return: np.array: m x 3 dim, for x, y, z position in euler space
    """
    cos57, sin57 = np.cos(arm_angle[:, 1] + arm_angle[:, 3]), np.sin(arm_angle[:, 1] + arm_angle[:, 3])
    sin7sin6 = np.sin(arm_angle[:, 3]) * np.sin(arm_angle[:, 2])
    cos6 = np.cos(arm_angle[:, 2])
    cos2, sin2 = np.cos(arm_angle[:, 0]), np.sin(arm_angle[:, 0])
    cos5, sin5 = np.cos(arm_angle[:, 1]), np.sin(arm_angle[:, 1])

    x = forearm_len * cos2 * (cos57 - sin7sin6) + upper_arm_len * cos5 * cos2
    y = forearm_len * sin2 * (cos57 - sin7sin6) + upper_arm_len * cos5 * sin2
    z = forearm_len * sin57 * cos6 + upper_arm_len * sin5

    return x, y, z


if __name__ == "__main__":
    pass
