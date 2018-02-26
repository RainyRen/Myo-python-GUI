import pdb
import time
import yaml
import json
from collections import deque

import numpy as np
import zmq
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

import myo
from myo_preprocess import Filter


class MyoFeed(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        # # ===== load parameter from config file =====
        with open("./config/ini_config.yml", "r") as config_file:
            config = yaml.load(config_file)

        self.is_req_mode = config['req_mode']
        self.tcp_address = config['tcp_address']
        self.do_filter = config['do_filter']
        self.moving_ave = config['moving_ave']
        self.filter_order = config['filter_order']
        self.low_cutoff = config['low_cutoff']
        self.window_size = config['window_size']
        self.send_fs = config['send_fs']
        self.filter_fs = 200  # device EMG sampling frequency
        self.connect_state = False

        if self.do_filter:
            self.high_pass_filter = Filter(self.filter_fs, 'highpass')

        # # ===== initial zmq protocol =====
        context = zmq.Context()
        if self.is_req_mode:
            self.socket = context.socket(zmq.REP)
        else:
            self.socket = context.socket(zmq.PUB)
        # # tcp address use tcp://127.0.0.1:5555
        self.socket.bind("tcp://" + self.tcp_address)
        print("Server start at localhost:5555")

        # *****initial myo*****
        if not myo.myo_initialized():
            print("initial myo")
            myo.init()
        # # ***** try to open bluetooth protocol *****
        try:
            self.hub = myo.Hub()
        except MemoryError:
            print("Myo Hub could not be created.", end=" ")
            print("Make sure Myo Connect is running.")
            return

        # # ***** using feed to monitor event *****
        self.feed = myo.device_listener.Feed()

        self.device_data = {'device_num': 0}
        self.emg_que = deque(maxlen=self.window_size)

        # # ===== thread multi task =====
        self.stop_signal = True
        self.mutex = QMutex()
        self.connect_state = True

    def run(self):
        with QMutexLocker(self.mutex):
            self.stop_signal = False

        try:
            self.hub.run(1000, self.feed)
            print("Waiting for a Myo to connect ...")
            myo_devices = self.feed.get_devices()
            print(myo_devices)
            print("{} myo devices connect".format(len(myo_devices)))
            self.msg_signal.emit("{} myo devices connect".format(len(myo_devices)))

            if not myo_devices:
                print("No Myo connected")
                self.msg_signal.emit("No Myo connected")
                return

            for device in myo_devices:
                self.device_data['device_num'] += 1
                device.set_stream_emg(myo.StreamEmg.enabled)
                device.request_rssi()
                device.request_battery_level()
            # pdb.set_trace()

            print("start dongle")
            if self.is_req_mode:
                print("request mode on")
                self._req_mode(myo_devices)

            else:
                print("publish mode on")
                self._pub_mode(myo_devices)

        except KeyboardInterrupt:
            print("\nQuitting ...")

        finally:
            print("closed hub & socket")
            self.socket.close()
            self.hub.stop(True)
            self.hub.shutdown()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stop_signal = True

        if self.is_req_mode:
            self.socket.close()

        # self.hub.stop(True)
        # self.hub.shutdown()

    def is_stop(self):
        with QMutexLocker(self.mutex):
            return self.stop_signal

    def _req_mode(self, myo_devices):
        while self.hub.running:
            cmd = self.socket.recv()

            if cmd == b'require':
                self._get_devices_data(myo_devices)
                send_data = json.dumps(self.device_data)
                # print(send_data)
                self.socket.send(bytearray(str(send_data), 'utf-8'))

            elif cmd == b'quit':
                self.socket.send(b'Myo disconnect')
                break

            else:
                print("No such command")

    def _pub_mode(self, myo_devices):
        while self.hub.running:
            self._get_devices_data(myo_devices)

            send_data = json.dumps(self.device_data)
            self.socket.send_string(send_data)

            if self.stop_signal:
                print("stop myo")
                break

            time.sleep(1 / self.send_fs)
            # print(send_data)

    def _get_devices_data(self, myo_devices):
        self.device_data['orientation'] = [
            tuple(device.orientation) for device in myo_devices
        ]
        self.device_data['acceleration'] = [
            tuple(device.acceleration) for device in myo_devices
        ]
        self.device_data['gyroscope'] = [
            tuple(device.gyroscope) for device in myo_devices
        ]

        # *****do pre-process on EMG data*****
        emg_list = [device.emg for device in myo_devices]
        if emg_list[0] is not None:
            self.emg_que.append(emg_list)

        else:
            return

        emg_array = np.asarray(self.emg_que)
        filter_emg_list = []
        # print(emg_array.shape)
        # # emg_array shape (window_len, device_num, 8-emg_channel)
        if emg_array.shape[0] == self.window_size:
            # raw_emg = emg_array.reshape(-1, 8, window_size)
            raw_emg = emg_array.swapaxes(0, 1)
            raw_emg = raw_emg.swapaxes(1, 2)

            if self.do_filter:
                filtered_emg = np.asarray([self.high_pass_filter.filtfilt(emg) for emg in raw_emg])

            else:
                filtered_emg = raw_emg

            # # full wave rectification
            filtered_emg = np.fabs(filtered_emg)

            # # if using moving avg
            if self.moving_ave:
                ma_emg = np.asarray(
                    np.ma.average(filtered_emg, axis=-1, weights=np.arange(filtered_emg.shape[-1]))
                )
                filter_emg_list = ma_emg.tolist()

            # # if don't use moving avg
            else:
                filter_emg_list = filtered_emg[:, :, -1].tolist()

        self.device_data['emg'] = filter_emg_list
        # print(device_data['emg'])
        # **********
        self.device_data['status'] = [device.battery_level for device in myo_devices] + \
                                     [device.rssi for device in myo_devices]


if __name__ == '__main__':
    myo_feed = MyoFeed()
    myo_feed.run()
