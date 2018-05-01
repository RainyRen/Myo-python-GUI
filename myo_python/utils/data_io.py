# coding: utf-8
import pdb
from pathlib import Path
from itertools import cycle

import pandas as pd
import numpy as np
# from keras.utils import to_categorical


class DataManager(object):
    def __init__(self, file_path,
                 separate_rate=0.2, time_length=15, future_time=1,
                 one_target=True, degree2rad=False, use_direction=False, direction_threshold=0.4):
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

        self.use_direction = use_direction
        self.direction_threshold = direction_threshold

        self.tr_kinematic = list()
        self.tr_emg = list()
        self.tr_target = list()

        self.val_kinematic = list()
        self.val_emg = list()
        self.val_target = list()

        self.separate_rate = separate_rate
        all_file_list = list(self.file_path.glob('*.csv'))
        print("Found {} files".format(len(all_file_list)))
        print(all_file_list, end='\n\n')

        all_file_num = len(all_file_list)
        val_file_num = int(all_file_num * separate_rate)
        self.tr_file_list = all_file_list[val_file_num:]
        self.val_file_list = all_file_list[:val_file_num]

    def get_all_data(self, get_emg_raw=False, emg_3d=True):
        """
        get all training and validation data, this method is for keras training,
        not appropriate for tensorflow
        :param bool get_emg_raw:
        :param bool emg_3d:
        :return:
        """
        # # extract data from different files and append to list
        for current_file in self.tr_file_list:
            kinematic, emg, target = self._load_data(current_file, get_emg_raw=get_emg_raw, emg_3d=emg_3d)
            self.tr_kinematic.append(kinematic)
            self.tr_emg.append(emg)
            self.tr_target.append(target)

        for current_file in self.val_file_list:
            kinematic, emg, target = self._load_data(current_file, get_emg_raw=get_emg_raw, emg_3d=emg_3d)
            self.val_kinematic.append(kinematic)
            self.val_emg.append(emg)
            self.val_target.append(target)

        # # self.kinematic, self.emg, self.target now are list contain different files data, index by file order
        # # we need concatenate all different data to one matrix as model inputs
        tr = list(map(lambda x: np.concatenate(x, axis=0), [self.tr_kinematic, self.tr_emg, self.tr_target]))
        val = list(map(lambda x: np.concatenate(x, axis=0), [self.val_kinematic, self.val_emg, self.val_target]))\
            if self.separate_rate > 0 else None

        # # normalize emg
        tr[1] = tr[1] / 60.0
        if val:
            val[1] = val[1] / 60.0

        print("train kinematic data shape: ", tr[0].shape)
        print("train emg data shape: ", tr[1].shape)
        print("train target data shape: ", tr[2].shape)

        if self.separate_rate > 0:
            print("validation kinematic data shape: ", val[0].shape)
            print("validation emg data shape: ", val[1].shape)
            print("validation target data shape: ", val[2].shape)

        else:
            print("no validation data")

        return tr, val

    def data_generator(self, get_emg_raw=False, emg_3d=False, batch_size=32, shuffle=False, less_memory=False):
        """
        this is design for tensorflow training
        :return:
        """
        if less_memory:
            tr_file_cycle = cycle(self.tr_file_list)
            val_file_cycle = cycle(self.val_file_list)
            return
        else:
            tr_data, val_data = self.get_all_data(get_emg_raw=get_emg_raw, emg_3d=emg_3d)

        while True:
            samples_num = tr_data[0].shape[0]
            n_batch = samples_num // batch_size
            indices_pool = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batch)]
            indices_order = np.random.permutation(n_batch) if shuffle else range(n_batch)

            for index in indices_order:
                split_pair = indices_pool[index]
                s, e = split_pair

                tr_batch = tuple(map(lambda x: x[s:e], tr_data))
                val_batch = tuple(map(lambda x: x[s:e], val_data)) if self.separate_rate > 0 else None

                yield tr_batch, val_batch

    def _load_data(self, file_path, get_emg_raw=False, emg_3d=True):
        """
        internal use function, for each file get data we need
        return in nd_array form
        :param bool get_emg_raw: get emg raw data or time-frequency features
        :param bool emg_3d: ignored if get_emg_raw is True, reshape emg time-frequency features to multi-dim
        :return:
        """
        arm_angle_samples = list()
        gyro_samples = list()
        acc_samples = list()
        emg_raw_samples = list()
        emg_features_samples = list()
        target_samples = list()

        data_df = pd.read_csv(file_path, skiprows=1, header=None)
        n_sample = data_df.shape[0] - self.time_length - (self.future_time - 1)

        # samples_pool = [(i * self.time_length, (i + 1) * self.time_length) for i in range(n_sample)]

        arm_angle = data_df.iloc[:, :4].values
        gyro = data_df.iloc[:, 10:16].values
        acc = data_df.iloc[:, 16:22].values
        emg = data_df.iloc[:, 22:38].values
        emg_features_real = data_df.iloc[:, 38:198].values
        emg_features_imag = data_df.iloc[:, 198:358].values

        if self.degree2rad:
            arm_angle = np.radians(arm_angle)
            gyro = np.radians(gyro)

        # # combine complex value as a image, shape of (samples, channels, frequence band, real or imag)
        emg_features_3d = np.concatenate(
            (emg_features_real.reshape(emg_features_real.shape[0], 16, -1, 1),
             emg_features_imag.reshape(emg_features_imag.shape[0], 16, -1, 1)),
            axis=-1
        )

        emg_features_complex = emg_features_real + 1j * emg_features_imag
        # # convert complex to magnitude and all channels are flattened to one row
        emg_features_mag = np.abs(emg_features_complex)

        if emg_3d:
            # # reshape magnitude to 2d format, shape of (samples, channels, frequence band)
            emg_features_mag = emg_features_mag.reshape(emg_features_mag.shape[0], 16, -1, 1)

        if self.use_direction:
            # print("convert postion to direction")
            target_direction_value = np.abs(arm_angle[1:] - arm_angle[:-1])
            target = np.where(target_direction_value > self.direction_threshold, 1, 0)

        else:
            target = arm_angle[1:]

        for i in range(n_sample):
            arm_angle_samples.append(arm_angle[i:i + self.time_length])
            gyro_samples.append(gyro[i:i + self.time_length])
            acc_samples.append(acc[i:i + self.time_length])
            emg_raw_samples.append(emg[i:i + self.time_length])
            # emg_features_samples.append(
            #     np.hstack(
            #         (emg_features_real[i:i + self.time_length], emg_features_imag[i:i + self.time_length]))
            # )
            emg_features_samples.append(emg_features_mag[i:i + self.time_length])

            # # if we do regression, we need get next time step position as our labels
            if self.one_target:
                target_samples.append(target[i + self.time_length + self.future_time - 1 - 1])
            else:
                # # get multi output target
                target_samples.append(
                    target[i + self.future_time - 1: i + self.time_length + self.future_time - 1]
                )

        # # convert list to numpy array
        arm_angle_samples = np.asarray(arm_angle_samples)
        gyro_samples = np.asarray(gyro_samples)
        acc_samples = np.asarray(acc_samples)

        # # merge relative information
        kinematic_samples = np.concatenate((arm_angle_samples, gyro_samples, acc_samples), axis=-1)
        emg_raw_samples = np.asarray(emg_raw_samples)
        emg_features_samples = np.asarray(emg_features_samples)
        target_samples = np.asarray(target_samples)

        # if self.use_direction:
        #     target_samples = to_categorical(target_samples, num_classes=3)

        emg_samples = emg_raw_samples if get_emg_raw else emg_features_samples

        return kinematic_samples, emg_samples, target_samples


# # ====================================== functions ===================================================================
def angle2position(arm_angle, forearm_len=33, upper_arm_len=32):
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


def folder_check(folder_path):
    """
    :param folder_path: type of str or Path, which need to check if exists
    :retrun: None
    """
    folder_path = Path(folder_path)

    if folder_path.exists():
        user_input = input('folder {} already exist, do you want to overrider? [y/n] '
                           .format(folder_path.stem))
        if user_input in ('n', 'no', 'N', 'No'):
            print('please reset your arguments')
            exit(0)
        else:
            print('\n')
    else:
        print('create new folder {}'.format(folder_path.stem))
        folder_path.mkdir()


if __name__ == "__main__":
    pass
