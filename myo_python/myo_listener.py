# coding: utf-8
"""
The Script are writen in python 3
Basic threading event for multi sensor on myo
Also include method reconstruct arm angle
"""
import math
import threading
from collections import deque

import numpy as np

import myo
from myo_filter import Filter, Complementary, Kalman, STFT


class Listener(myo.DeviceListener):
    def __init__(self, do_filter=False, moving_ave=False, filter_fs=200, filter_order=4, low_cutoff=2, window_size=11):
        super(self.__class__, self).__init__()
        # # ===== EMG pre-processing init =====
        self.do_filter = do_filter
        if do_filter:
            self.high_pass_filter = Filter(filter_fs, 'highpass', low_cut=low_cutoff, high_cut=4, order=filter_order)
            self.notch_filter = Filter(filter_fs, 'notch')

        self.moving_ave = moving_ave

        # # if moving_ave is False. this will ignore
        self.window_size = window_size
        self.stft = STFT(self.window_size, window_name='hamming')

        # # ===== property init ======
        self.lock = threading.Lock()

        self.known_myos = list()
        self.arm = list()
        self.orientation = list()
        self.acceleration = list()
        self.gyroscope = list()
        self.emg = list()
        self.rssi = list()
        self.battery_level = list()

    def on_connect(self, device, timestamp, firmware_version):
        # print(device)
        device.set_stream_emg(myo.StreamEmg.enabled)
        # device.request_rssi()
        # device.request_battery_level()

    def on_pair(self, device, timestamp, firmware_version):
        # pass
        # print(device)
        self.known_myos.append(device)
        self.arm.append(None)
        self.emg.append(deque(maxlen=self.window_size))
        self.orientation.append(None)
        self.acceleration.append(None)
        self.gyroscope.append(None)
        self.rssi.append(None)
        self.battery_level.append(None)

    def on_unpair(self, device, timestamp):
        pass

    def on_arm_sync(self, device, timestamp, arm, x_direction, rotation,
                    warmup_state):
        device_num = self.identify_myo(device)
        self.arm[device_num] = arm

    def on_emg_data(self, device, timestamp, emg_data):
        # print(device.value)
        # print(emg_data)
        with self.lock:
            device_num = self.identify_myo(device)
            self.emg[device_num].append(emg_data)

    def on_orientation_data(self, device, timestamp, orientation):
        """
        get myo orientation respect to world frame
        :param device:
        :param timestamp:
        :param orientation: type of myo Quaternion object
        :return:
        """
        device_num = self.identify_myo(device)
        self.orientation[device_num] = orientation

    def on_accelerometor_data(self, device, timestamp, acceleration):
        """
        get accelerometer data on myo
        :param device:
        :param timestamp:
        :param acceleration: type of myo vector object
        :return:
        """
        device_num = self.identify_myo(device)
        self.acceleration[device_num] = list(acceleration)

    def on_gyroscope_data(self, device, timestamp, gyroscope):
        device_num = self.identify_myo(device)
        self.gyroscope[device_num] = list(gyroscope)

    def on_rssi(self, device, timestamp, rssi):
        device_num = self.identify_myo(device)
        self.rssi[device_num] = rssi

    def on_battery_level_received(self, device, timestamp, level):
        device_num = self.identify_myo(device)
        self.battery_level[device_num] = level

    # # ===== below are method we can get from outside =====
    def identify_myo(self, device):
        for idx, one_device in enumerate(self.known_myos):
            if one_device.value == device.value:
                return idx

        return -1

    @property
    def get_connected_devices(self):
        # pdb.set_trace()
        with self.lock:
            return self.known_myos

    @property
    def get_emg_data(self):
        # pdb.set_trace()
        with self.lock:
            raw_emg = np.asarray([tuple(one_emg) for one_emg in self.emg])
            # print("=======")
            # print(raw_emg)
            # print("=======")
            # print(raw_emg.shape)

            if len(raw_emg.shape) == 3:
                raw_emg = raw_emg.swapaxes(1, 2)
            else:
                print(raw_emg.shape)
                print("get emg loss data")
                return None

            # raw_emg = np.fabs(raw_emg)    # used for using low pass filer method
            if self.do_filter:
                filtered_emg = np.asarray([self.high_pass_filter.filtfilt(emg) for emg in raw_emg])

            else:
                filtered_emg = raw_emg

            # # full wave rectification
            filtered_emg = np.fabs(filtered_emg)

            if self.moving_ave:
                ma_emg = np.asarray(
                    np.ma.average(filtered_emg, axis=-1, weights=np.arange(filtered_emg.shape[-1]))
                )
                filter_emg_list = ma_emg.tolist()

            # # if don't use moving avg
            else:
                filter_emg_list = filtered_emg[:, :, -1].tolist()

            return filter_emg_list

    @property
    def get_emg_features(self):
        """
        get emg time-frequency features from shot-time Fourier transform
        :return:
        """
        raw_emg = np.asarray([tuple(one_emg) for one_emg in self.emg])

        if len(raw_emg.shape) == 3:
            raw_emg = raw_emg.swapaxes(1, 2)
            raw_emg = raw_emg.reshape(-1, self.window_size)

            # # check shape is correct
            assert raw_emg.shape == (len(self.known_myos) * 8, self.window_size)

        else:
            print(raw_emg.shape)
            print("get emg loss data")
            return None

        return self.stft(raw_emg)

    @property
    def get_orientation(self):
        with self.lock:
            return [list(quaternion) for quaternion in self.orientation]

    @property
    def get_rpy(self):
        with self.lock:
            return [list(map(math.degrees, quaternion.rpy)) for quaternion in self.orientation]

    @property
    def get_acceleration(self):
        with self.lock:
            return self.acceleration

    @property
    def get_gyroscope(self):
        with self.lock:
            return self.gyroscope

    @property
    def get_device_state(self):
        with self.lock:
            return self.battery_level, self.rssi


class ArmAngle(object):
    def __init__(self, rotation, dt=0.02):
        """
        This is an implementation of author Lin, Xuan-Zhu method based on C++
        Using Kalman filter combine accelerometor and gyroscope
        Do some optimization than origin code
        :param rotation: Quaternion of angle 2 on Magnetometer
        :param float dt: sampling time in millisecond
        """
        self.yaw_init = [0., 0.]
        self.acc_angle = [[0., 0.], [0., 0.]]
        self.euler = [[0., 0., 0.], [0., 0., 0.]]
        self.euler_filter = [[Kalman(), Kalman()], [Kalman(), Kalman()]]
        self.dt = dt
        self.is_init = False

        self.angle_2 = 0.
        self.angle_5 = 0.
        self.angle_6 = 0.
        self.angle_7 = 0.
        self.calibration(rotation)

    def calibration(self, rotation):
        """

        :param list rotation: tow list contain 8 element of two arm Quaternion
        :return: None
        """
        # self.angle_2_bias = self.angle_2
        # self.angle_5_bias = self.angle_5
        # self.angle_6_bias = self.angle_6
        # self.angle_7_bias = self.angle_7
        self.yaw_init = [self._cal_yaw(one_rotate) for one_rotate in rotation]
        self.is_init = True

    def cal_arm_angle(self, rotation, acceleration, gyroscope):
        self._set_acc_angle(acceleration)
        yaw = [self._cal_yaw(one_rotate) for one_rotate in rotation]
        self._set_euler(yaw, gyroscope)

        self.angle_2 = -self.euler[1][2]
        self.angle_5 = self.euler[1][1]
        self.angle_6 = -self.euler[1][0]
        self.angle_7 = abs(self.euler[1][1] - self.euler[0][1]) + abs(self.euler[1][2] - self.euler[0][2])

        # if self.angle_7 > 180:
        #     print("angle 7 large than 180")
        #     self.angle_7 = 178 + ((self.angle_7 - 180) / 90)

        return self.angle_2, self.angle_5, self.angle_6, self.angle_7

    def _set_euler(self, yaw, gyroscope):
        for myo_idx in range(2):
            for angle_idx in range(2):
                self.euler[myo_idx][angle_idx] = self.euler_filter[myo_idx][angle_idx].get_angle(
                    self.acc_angle[myo_idx][angle_idx], gyroscope[myo_idx][angle_idx], self.dt
                )
            self.euler[myo_idx][2] = yaw[myo_idx] - self.yaw_init[myo_idx]

            if self.euler[myo_idx][2] > 180:
                self.euler[myo_idx][2] -= 360

    def _set_acc_angle(self, acceleration):
        """
        upper arm must wear primary myo and device is number 1
        down arm must wear non-primary myo and device is number 0
        :return: None
        """
        for myo_idx in range(2):
            self.acc_angle[myo_idx][0] = math.degrees(
                math.atan2(
                    acceleration[myo_idx][1],
                    math.sqrt(acceleration[myo_idx][0] ** 2 + acceleration[myo_idx][2] ** 2)
                )
            )

            self.acc_angle[myo_idx][1] = math.degrees(
                math.atan2(
                    -acceleration[myo_idx][0],
                    math.sqrt(acceleration[myo_idx][1] ** 2 + acceleration[myo_idx][2] ** 2)
                )
            )

    @staticmethod
    def _cal_yaw(rotate):
        yaw_degree = math.degrees(
            math.atan2(
                2 * (rotate[3] * rotate[2] + rotate[0] * rotate[1]),
                1 - 2 * (rotate[1] ** 2 + rotate[2] ** 2)
            )
        )

        return yaw_degree


class ArmAngle2(object):
    def __init__(self, rpys, compensate_k=None, use_filter=True, dt=0.05):
        """
        implementation of complementary filter, contain muscle deformation compensate method
        only using linear compensate method
        :param rpys:
        :param compensate_k: tuple contain four arm angle compensate value k
        :param use_filter: if true, use complementary filter
        :param float dt: sampling time
        """
        self.upper_arm_bias = (0., 0., 0.)
        self.forearm_bias = (0., 0., 0.)
        self.compensate_k = compensate_k if compensate_k is not None else (0., 0., 0., 0.)
        self.use_filter = use_filter

        self.angle_2 = 0.
        self.angle_5 = 0.
        self.angle_6 = 0.
        self.angle_7 = 0.

        self.angle_2_cps_k, self.angle_5_cps_k, self.angle_6_cps_k, self.angle_7_cps_k = self.compensate_k

        if self.use_filter:
            self.cpl_filter_2 = Complementary(dt, self.angle_2)
            self.cpl_filter_5 = Complementary(dt, self.angle_5)
            self.cpl_filter_7 = Complementary(dt, self.angle_7)

        self.calibration(rpys)

    def calibration(self, rpys):
        """
        fast calibration on different initial point
        :param rpys: list of list contain current arm euler angle
        :return: None
        """
        self.forearm_bias, self.upper_arm_bias = rpys
        if self.use_filter:
            self.cpl_filter_2.angle = 0
            self.cpl_filter_5.angle = 0
            self.cpl_filter_7.angle = 0

    def cal_arm_angle(self, rpys, gyr=None):
        """
        :param rpys: include two arms rpy, in order pitch, roll, yaw
        :param gyr: two myo arm's angular velocity (gyroscope)
        :return tuple: four current arm angle
        """
        forearm, upper_arm = rpys

        self.angle_2 = upper_arm[2] - self.upper_arm_bias[2]
        self.angle_5 = upper_arm[1] - self.upper_arm_bias[1]
        self.angle_6 = upper_arm[0] - self.upper_arm_bias[0]

        if self.use_filter:
            self.angle_2 = self.cpl_filter_2.get_angle(self.angle_2, gyr[1][2])
            self.angle_5 = self.cpl_filter_5.get_angle(self.angle_5, gyr[1][1])
            self.angle_7 = self.cpl_filter_7.get_angle(forearm[1] - self.forearm_bias[1], gyr[0][1]) - self.angle_5

        else:
            self.angle_7 = (forearm[1] - self.forearm_bias[1]) - self.angle_5

        # # ----- calculate muscle deformation compensate -----
        self.angle_2 *= (1 + self.angle_2_cps_k)
        self.angle_5 *= (1 + self.angle_5_cps_k)
        self.angle_6 *= (1 + self.angle_6_cps_k)
        self.angle_7 *= (1 + self.angle_7_cps_k)

        # # ----- bound each angle value -----
        self.angle_2 = self._bound(0., 90., self.angle_2)
        self.angle_5 = self._bound(0., 90., self.angle_5)
        # self.angle_6 = self._bound(0., 90., self.angle_6)
        self.angle_7 = self._bound(0., 90., self.angle_7)

        # # Keep two decimal
        self.angle_2, self.angle_5, self.angle_6, self.angle_7 = map(
                lambda x: round(x, 2), [self.angle_2, self.angle_5, self.angle_6, self.angle_7]
        )

        return self.angle_2, self.angle_5, self.angle_6, self.angle_7

    @staticmethod
    def _bound(low, high, value):
        return max(low, min(high, value))
