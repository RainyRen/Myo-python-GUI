# coding: utf-8
"""
The Script are writen in python 3
"""
import pdb
import time
import yaml
import json
# import csv

import zmq
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

import myo
from myo_listener import Listener, ArmAngle2


class MyoListen(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        # # ===== load parameter from config file =====
        with open("./config/ini_config.yml", "r") as config_file:
            config = yaml.load(config_file)

        self.is_req_mode = config['req_mode']
        self.tcp_address = config['tcp_address']
        self.emg_filter = config['emg_filter']
        self.moving_ave = config['moving_ave']
        self.filter_order = config['filter_order']
        self.low_cutoff = config['low_cutoff']
        self.window_size = config['window_size']
        self.send_fs = config['send_fs']      # type: int
        self.imu_filter = config['imu_filter']
        self.complementary_a = config['complementary_a']
        self.elbow_compensate_k = config['elbow_compensate_k']
        self.emg_fs = 200       # device EMG sampling frequency
        self.connect_state = False
        self.send_save = config['send_save']
        print(self.send_save)
        # # ===== initial zmq protocol =====
        context = zmq.Context()
        if self.is_req_mode:
            self.socket = context.socket(zmq.REP)
        else:
            self.socket = context.socket(zmq.PUB)
        # # tcp address use tcp://127.0.0.1:5555
        self.socket.bind("tcp://" + self.tcp_address)
        print("Server start at localhost:5555")

        # # ===== try initial myo =====
        if not myo.myo_initialized():
            print("init myo")
            myo.init()

        # # try to open bluetooth protocol =====
        try:
            self.hub = myo.Hub()

        except MemoryError:
            print("Myo Hub could not be created. Make sure Myo Connect is running.")
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
            self.socket.close()
            self.hub.stop(True)
            self.hub.shutdown()

    def send(self, is_send):
        with QMutexLocker(self.mutex):
            self.send_signal = is_send

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
        while self.hub.running:
            cmd = self.socket.recv()

            if cmd == b'require':
                self._get_devices_data()
                send_data = json.dump(self.device_data)
                self.socket.send(bytearray(str(send_data), 'utf-8'))

            elif cmd == b'quit':
                self.socket.send(b'Myo disconnect')
                break

            else:
                print("No such command")

    def _pub_mode(self):
        while self.hub.running:
            t_start = time.time()
            self._get_devices_data()

            if self.send_signal:
                send_data = json.dumps(self.device_data)
                self.socket.send_string(send_data)

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
        # self.device_data['myo_status'] = self.listener.get_device_state
        # print('\r                 ', end='')
        # print('\r{:.2f}, {:.2f}, {:.2f}'.format(*self.device_data['rpy'][0]), end='')

        if self.get_arm_angle_signal:
            # self.device_data['arm_angle'] = self.arm_angle.cal_arm_angle(
            #     self.device_data['orientation'], self.device_data['acceleration'], self.device_data['gyroscope'])
            self.device_data['arm_angle'] = self.arm_angle.cal_arm_angle(
                self.device_data['rpy'], gyr=self.device_data['gyroscope'])
            print('\r                 ', end='')
            print('\r{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*self.device_data['arm_angle']), end='')

        else:
            self.device_data['arm_angle'] = []

        if self.record_signal:
            save_data = [
                element
                for item in self.send_save if item != 'arm_angle'
                for device_id in range(self.devices_num)
                for element in self.device_data[item][device_id]
            ]
            if 'arm_angle' in self.send_save:
                save_data = list(self.device_data['arm_angle']) + save_data

            # with open(self.record_file_name, 'a', newline='') as record_file:
                # writer = csv.writer(record_file, quoting=csv.QUOTE_NONE)
                # writer.writerow(save_data)
                contents = ','.join(map(lambda x: '{:.3f}'.format(x), save_data)) + '\n'
                self.record_file.write(contents)

    def get_arm_angle(self, is_get):
        with QMutexLocker(self.mutex):
            if self.devices_num != 2:
                print('acquire more myo on arm!')

            elif is_get:
                self.get_arm_angle_signal = True
                # self.arm_angle = ArmAngle(self.device_data['orientation'], dt=0.02)
                self.arm_angle = ArmAngle2(
                    self.device_data['rpy'],
                    compensate_k=self.elbow_compensate_k, use_filter=self.imu_filter, dt=(1/self.send_fs)
                )

            else:
                self.get_arm_angle_signal = False

    def arm_calibration(self):
        # self.arm_angle.calibration(self.device_data['orientation'])
        self.arm_angle.calibration(self.device_data['rpy'])


if __name__ == '__main__':
    myo_listen = MyoListen()
    myo_listen.run()
