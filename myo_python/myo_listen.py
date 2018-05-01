# coding: utf-8
"""
The Script are writen in python 3
"""
# # ===== build-in library =====
import pdb
import time
import yaml
import json
import math
from collections import deque

# # ===== third party library =====
import socket
import zmq
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib

# # ===== own library =====
import myo
from myo_listener import Listener, ArmAngle2


class MyoListen(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, ini_config, parent=None):
        super(self.__class__, self).__init__(parent)
        # # ===== load parameter from config file =====
        self.is_tcp = ini_config['tcp_mode']                    # type: bool
        self.is_req_mode = ini_config['req_mode']               # type: bool
        self.socket_address = ini_config['socket_address']      # type: str
        self.emg_filter = ini_config['emg_filter']              # type: bool
        self.moving_ave = ini_config['moving_ave']              # type: bool
        self.filter_order = ini_config['filter_order']          # type: int
        self.low_cutoff = ini_config['low_cutoff']              # type: int
        self.window_size = ini_config['window_size']            # type: int
        self.send_fs = ini_config['send_fs']                    # type: int
        self.imu_filter = ini_config['imu_filter']              # type: bool
        self.complementary_a = ini_config['complementary_a']    # type: float
        self.ha_compensate_k = ini_config['ha_compensate_k']    # type: float
        self.sf_compensate_k = ini_config['sf_compensate_k']    # type: float
        self.er_compensate_k = ini_config['er_compensate_k']    # type: float
        self.ef_compensate_k = ini_config['ef_compensate_k']    # type: float
        self.send_save = ini_config['send_save']                # type: list
        self.emg_fs = 200       # device EMG sampling frequency
        self.connect_state = False
        print('save content: ', self.send_save)

        # # ===== socket initial =====
        self.socket = None
        if not self.is_tcp:
            self.udp_address = None

        # # ===== try initial myo =====
        if not myo.myo_initialized():
            print("init myo")
            myo.init()

        # # try to open bluetooth protocol =====
        try:
            self.hub = myo.Hub()

        except MemoryError:
            print("Myo Hub could not be created. Make sure Myo Connect is running.")
            self.hub = None
            return

        # # ===== using event monitor =====
        # hub.set_locking_policy(libmyo.LockingPolicy.none)
        self.listener = Listener(
            do_filter=self.emg_filter, moving_ave=self.moving_ave,
            filter_fs=self.emg_fs, filter_order=self.filter_order, low_cutoff=self.low_cutoff,
            window_size=self.window_size
        )
        self.device_data = dict()
        self.devices_num = 0
        # # ===== thread multi task =====
        self.stop_signal = True

        self.get_arm_angle_signal = False
        self.arm_angle = None

        self.estimate_signal = False
        self.estimator = None
        self._kinematic_window = None
        self._emg_window = None
        self._model_window_size = 0

        self.send_signal = False
        self.record_signal = False
        self.record_file = None
        self.mutex = QMutex()
        self.connect_state = True

    def run(self):
        with QMutexLocker(self.mutex):
            self.stop_signal = False
            self.record_signal = False

        try:
            self.hub.run(20, self.listener)
            # # ===== wait a moment to start event =====
            print("Waiting for Myo to connect...")
            time.sleep(0.5)

            # # ===== get connected devices =====
            self.devices_num = len(self.listener.get_connected_devices)
            print("{} myo devices connect".format(self.devices_num))
            self.msg_signal.emit("{} myo devices connect".format(self.devices_num))

            if self.devices_num == 0:
                print("No Myo connected")
                self.msg_signal.emit("No Myo connected")
                return

            for device_id, device_arm in enumerate(self.listener.arm):
                print('Myo {} wear on {}'.format(device_id, device_arm))
                self.msg_signal.emit('Myo {} wear on {}'.format(device_id, device_arm))

            print("Start Dongle")
            time.sleep(0.5)
            if self.is_req_mode:
                print("request mode on")
                self._req_mode()

            else:
                print("publish mode on")
                self._pub_mode()

        except KeyboardInterrupt:
            print("\nQuitting ...")

        finally:
            print("close hub & socket")
            if self.socket is not None:
                self.socket.close()
            self.hub.stop(True)
            self.hub.shutdown()

    def socket_connect(self, is_send):
        with QMutexLocker(self.mutex):
            if is_send:
                """
                we have to check is TCP or UDP
                """
                if self.is_tcp:
                    # # ===== initial zmq protocol =====
                    context = zmq.Context()
                    if self.is_req_mode:
                        self.socket = context.socket(zmq.REP)
                    else:
                        self.socket = context.socket(zmq.PUB)
                    # # tcp address use tcp://127.0.0.1:5555
                    self.socket.bind("tcp://" + self.socket_address)

                    self.msg_signal.emit('tcp server start')
                    print("TCP server start at {}".format(self.socket_address))

                else:
                    udp_ip, udo_port = self.socket_address.split(':')
                    self.udp_address = (udp_ip, int(udo_port))
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.msg_signal.emit('udp server start')
                    print("UDP server start at {}".format(self.socket_address))

                self.send_signal = True

            else:
                self.send_signal = False
                self.msg_signal.emit('stop tcp server')
                print('stop tcp server')
                self.socket.close()
                self.socket = None

    def _socket_send(self, send_msg):
        if self.is_tcp:
            self.socket.send_string(send_msg)

        else:
            send_msg = bytes(json.dumps(self.device_data['arm_angle'] + self.device_data['estimate_angle']), 'utf-8')
            self.socket.sendto(send_msg, self.udp_address)

    def record(self, file_name, is_record):
        with QMutexLocker(self.mutex):
            self.record_signal = is_record
            if is_record:
                self.record_file = open(file_name, 'a')

            else:
                self.record_file.close()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stop_signal = True

        if self.is_req_mode:
            self.socket.close()

    def is_stop(self):
        with QMutexLocker(self.mutex):
            return self.stop_signal

    def _req_mode(self):
        self.msg_signal.emit('request mode on')
        while self.hub.running:
            cmd = self.socket.recv()

            if cmd == b'require':
                self._get_devices_data()
                send_data = json.dumps(self.device_data)
                self.socket.send(bytearray(str(send_data), 'utf-8'))

            elif cmd == b'quit':
                self.socket.send(b'Myo disconnect')
                break

            else:
                print("No such command")

    def _pub_mode(self):
        self.msg_signal.emit('publish mode on')
        while self.hub.running:
            t_start = time.time()
            self._get_devices_data()

            if self.send_signal:
                send_data = json.dumps(self.device_data)
                self._socket_send(send_data)

            if self.stop_signal:
                print("stop myo")
                break

            t_interval = time.time() - t_start
            pause_time = (1 / self.send_fs) - t_interval
            if pause_time < 0:
                print(pause_time)
            time.sleep(pause_time if pause_time > 0 else 0)
            # pdb.set_trace()

    def _get_devices_data(self):
        self.device_data['emg'] = self.listener.get_emg_data
        self.device_data['rpy'] = self.listener.get_rpy
        self.device_data['orientation'] = self.listener.get_orientation
        self.device_data['acceleration'] = self.listener.get_acceleration
        self.device_data['gyroscope'] = self.listener.get_gyroscope
        emg_features = self.listener.get_emg_features
        # self.device_data['myo_status'] = self.listener.get_device_state
        # print('\r                 ', end='')
        # print('\r{:.2f}, {:.2f}, {:.2f}'.format(*self.device_data['rpy'][0]), end='')

        if self.get_arm_angle_signal:
            # self.device_data['arm_angle'] = self.arm_angle.cal_arm_angle(
            #     self.device_data['orientation'], self.device_data['acceleration'], self.device_data['gyroscope'])
            self.device_data['arm_angle'] = list(self.arm_angle.cal_arm_angle(
                self.device_data['rpy'], gyr=self.device_data['gyroscope']))
            print('\r                       ', end='')
            print('\r{}, {}, {}, {}'.format(*self.device_data['arm_angle']), end='')

        else:
            self.device_data['arm_angle'] = [0., 0., 0., 0.]

        if self.estimate_signal:
            # # it must be two myo, so directly select two data
            # arm_angle = self.device_data['arm_angle']         # # use for get degrees arm angle
            arm_angle = list(map(math.radians, self.device_data['arm_angle']))
            gyr = self.device_data['gyroscope'][0] + self.device_data['gyroscope'][1]
            gyr = list(map(math.radians, gyr))
            acc = self.device_data['acceleration'][0] + self.device_data['acceleration'][1]

            kinematic_list = arm_angle + gyr + acc

            # # ========= linear model =========
            # emg_list = [channel / 60 for single_emg in self.device_data['emg'] for channel in single_emg]
            # lin_input = np.asarray(kinematic_list + emg_list)
            #
            # select_col = list(range(3, lin_input.shape[-1]))
            # for i in range(4):
            #     select_col[0] = i
            #     self.lin_result[i] = self.estimator[i].predict(lin_input[:, select_col])
            # self.device_data['estimate_angle'] = list(map(lambda x: round(x, 2), self.lin_result))
            # print(' | {}, {}, {}, {}'.format(*self.device_data['estimate_angle']), end='')
            # # ================================
            # # ========== deep mode ===========
            emg_features_mag_3d = np.abs(emg_features)

            self._kinematic_window.append(kinematic_list)
            # self._emg_window.append(self.device_data['emg'][0] + self.device_data['emg'][1])
            self._emg_window.append(emg_features_mag_3d)

            if len(self._kinematic_window) == self._model_window_size:
                input_kinematic = np.asarray(self._kinematic_window)[np.newaxis, ...]
                input_emg = np.asarray(self._emg_window)[np.newaxis, ..., np.newaxis]
                input_emg = input_emg / 60.0        # # normalize EMG signal

                estimate_angle_rad = self.estimator.predict_on_batch([input_kinematic, input_emg])
                estimate_angle_deg = np.degrees(estimate_angle_rad).ravel().tolist()
                self.device_data['estimate_angle'] = list(map(lambda x: round(x, 2), estimate_angle_deg))

                print(' | {}, {}, {}, {}'.format(*self.device_data['estimate_angle']), end='')
        else:
            self.device_data['estimate_angle'] = [0., 0., 0., 0.]

        if self.record_signal:
            save_data = [
                element
                for item in self.send_save if item != 'arm_angle'
                for device_id in range(self.devices_num)
                for element in self.device_data[item][device_id]
            ]

            if 'arm_angle' in self.send_save:
                save_data = list(self.device_data['arm_angle']) + save_data

            emg_features = emg_features.ravel()

            contents = ','.join(map(lambda x: '{:.3f}'.format(x), save_data)) + ',' +\
                       ','.join(map(lambda x: '{:.6f}'.format(x.real), emg_features)) + ',' +\
                       ','.join(map(lambda x: '{:.6f}'.format(x.imag), emg_features)) + '\n'

            self.record_file.write(contents)

    def get_arm_angle(self, is_get):
        with QMutexLocker(self.mutex):
            if self.devices_num != 2:
                print('acquire more myo on arm!')

            elif is_get and 'rpy' in self.device_data:
                # self.arm_angle = ArmAngle(self.device_data['orientation'], dt=0.02)
                arm_compensate_k = (
                    self.ha_compensate_k, self.sf_compensate_k, self.er_compensate_k, self.ef_compensate_k
                )

                self.arm_angle = ArmAngle2(
                    self.device_data['rpy'],
                    compensate_k=arm_compensate_k, use_filter=self.imu_filter, dt=(1/self.send_fs)
                )
                self.get_arm_angle_signal = True

            else:
                self.get_arm_angle_signal = False

    def arm_calibration(self, init_angle):
        # self.arm_angle.calibration(self.device_data['orientation'])
        self.arm_angle.calibration(self.device_data['rpy'], init_angle)

    def get_estimate_angle(self, is_get, model_path=None):
        with QMutexLocker(self.mutex):
            if is_get:
                with open(str(model_path / 'config.yml'), 'r') as model_config_file:
                    model_config = yaml.load(model_config_file)
                    self._model_window_size = model_config['time_length']

                self._kinematic_window = deque(maxlen=self._model_window_size)
                self._emg_window = deque(maxlen=self._model_window_size)

                # # ========== keras model ==========
                self.estimator = load_model(str(model_path / 'rnn_best.h5'))
                self.estimator._make_predict_function()
                # # =================================
                # # ========= linear model ===========
                # self.estimator = joblib.load(str(model_path / 'lin_model.p'))
                # self.lin_result = [0., 0., 0., 0.]
                # # ==================================

                self.estimate_signal = True

            else:
                self.estimate_signal = False
                self.estimator = None
                self._kinematic_window = None
                self._emg_window = None


if __name__ == '__main__':
    pass
