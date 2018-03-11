# import pdb
import os
import sys
import yaml
import csv
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit, QComboBox, QRadioButton, QCheckBox, QSpinBox, QButtonGroup
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt

from myo_listen import MyoListen
from myo_feed import MyoFeed

ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / "data"
IMAGE_PATH = ROOT_PATH / "images"


class App(QMainWindow):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.title = 'Myo App by RRLAB'
        # self.left = 0
        # self.top = 0
        # self.width = 300
        # self.height = 200
        self.table_widget = TableWidget(self)

        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon("./images/myo.png"))
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.setCentralWidget(self.table_widget)

        self.show()


class TableWidget(QWidget):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.myo_dongle = None
        self.config_dict = dict()

        # # ***** below are gui init *****
        # # ===== define whole layout =====
        self.layout = QVBoxLayout(self)

        # # ===== Initialize tab screen =====
        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_setting = QWidget()
        # self.tabs.resize(300, 200)

        # # ===== Add tabs =====
        self.tabs.addTab(self.tab_main, "Synch to LabVIEW")
        self.tabs.addTab(self.tab_setting, "Advance Setting")

        # # ===== Add main button =====
        self.quit_bnt = QPushButton("QUIT")
        self.quit_bnt.setIcon(QIcon(str(IMAGE_PATH / "quit.png")))
        self.quit_bnt.setStyleSheet("background-color: #EF5350")

        # # ===== Create first tab =====
        self.tcp_connect_bnt = QPushButton("connect")
        self.tcp_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "connect.png")))
        self.tcp_connect_bnt.setCheckable(True)
        self.tcp_send_bnt = QPushButton("send")
        self.tcp_send_bnt.setIcon(QIcon(str(IMAGE_PATH / "send.png")))
        self.tcp_send_bnt.setCheckable(True)
        self.tcp_send_bnt.setDisabled(True)
        self.data_record_bnt = QPushButton("record")
        self.data_record_bnt.setIcon(QIcon(str(IMAGE_PATH / "record.png")))
        self.data_record_bnt.setCheckable(True)
        self.data_record_bnt.setDisabled(True)

        self.arm_cali_bnt = QPushButton('arm calibration')
        self.arm_cali_bnt.setIcon(QIcon(str(IMAGE_PATH / "calibration")))
        self.arm_cali_bnt.setDisabled(True)
        self.arm_angle_bnt = QPushButton('arm angle')
        self.arm_angle_bnt.setIcon(QIcon(str(IMAGE_PATH / "angle.png")))
        self.arm_angle_bnt.setCheckable(True)
        self.arm_angle_bnt.setDisabled(True)

        self.file_delete_bnt = QPushButton("delete")
        self.file_delete_bnt.setIcon(QIcon(str(IMAGE_PATH / "delete.png")))

        self.tcp_address = QLineEdit("127.0.0.1:5555")
        self.file_name = QLineEdit("default")
        self.file_name.setAlignment(Qt.AlignCenter)

        self.tcp_mode_group = QButtonGroup(self.tab_main)
        self.pub_radio_bnt = QRadioButton("Public Mode")
        self.pub_radio_bnt.setChecked(True)
        self.req_radio_bnt = QRadioButton("Request Mode")
        self.tcp_mode_group.addButton(self.pub_radio_bnt)
        self.tcp_mode_group.addButton(self.req_radio_bnt)

        self.send_fs = QSpinBox()
        self.send_fs.setAlignment(Qt.AlignCenter)
        self.send_fs.setValue(10)
        self.send_fs.setRange(0, 200)

        self.myo_msg = QTextEdit()

        # # define different area layout
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("TCP Address:"), 1, 0)
        grid_layout.addWidget(self.tcp_address, 1, 1)
        grid_layout.addWidget(self.pub_radio_bnt, 2, 0)
        grid_layout.addWidget(self.req_radio_bnt, 2, 1)
        grid_layout.addWidget(QLabel("Send Frequency(Hz):"), 4, 0)
        grid_layout.addWidget(self.send_fs, 4, 1)
        grid_layout.addWidget(QLabel("File Name:"), 8, 0)
        grid_layout.addWidget(self.file_name, 8, 1)

        tcp_bnt_layout = QHBoxLayout()
        tcp_bnt_layout.addWidget(self.tcp_connect_bnt)
        tcp_bnt_layout.addWidget(self.tcp_send_bnt)

        arm_angle_layout = QHBoxLayout()
        arm_angle_layout.addWidget(self.arm_cali_bnt)
        arm_angle_layout.addWidget(self.arm_angle_bnt)

        file_bnt_layout = QHBoxLayout()
        file_bnt_layout.addWidget(self.data_record_bnt)
        file_bnt_layout.addWidget(self.file_delete_bnt)

        # # organize layout
        self.tab_main.layout = QVBoxLayout(self)
        self.tab_main.layout.addLayout(grid_layout)
        self.tab_main.layout.addLayout(tcp_bnt_layout)
        self.tab_main.layout.addLayout(arm_angle_layout)
        self.tab_main.layout.addLayout(file_bnt_layout)
        self.tab_main.layout.addWidget(self.myo_msg)
        self.tab_main.setLayout(self.tab_main.layout)

        # # ===================================== Create second tab ====================================================
        self.myo_mode_group = QButtonGroup(self.tab_setting)
        self.myo_listen_radio_bnt = QRadioButton("Listen Mode")
        self.myo_listen_radio_bnt.setChecked(True)
        self.myo_feed_radio_bnt = QRadioButton("Feed Mode")
        self.myo_mode_group.addButton(self.myo_listen_radio_bnt)
        self.myo_mode_group.addButton(self.myo_feed_radio_bnt)

        self.hpf_checkbox = QCheckBox("High Pass Filter")
        self.hpf_checkbox.setChecked(True)
        self.ma_checkbox = QCheckBox("Moving Average")
        self.ma_checkbox.setChecked(True)

        self.hpf_filter_oder = QSpinBox()
        self.hpf_filter_oder.setAlignment(Qt.AlignCenter)
        self.hpf_filter_oder.setValue(4)
        self.hpf_filter_oder.setRange(1, 5)
        self.hpf_cutoff_fs = QSpinBox()
        self.hpf_cutoff_fs.setAlignment(Qt.AlignCenter)
        self.hpf_cutoff_fs.setValue(2)
        self.hpf_cutoff_fs.setRange(0, 1000)
        self.ma_length = QSpinBox()
        self.ma_length.setAlignment(Qt.AlignCenter)
        self.ma_length.setValue(50)
        self.ma_length.setRange(1, 1000)

        self.elbow_compensate_k = QLineEdit("0.182")
        self.elbow_compensate_k.setAlignment(Qt.AlignCenter)
        self.arm_filter_group = QButtonGroup(self.tab_setting)
        self.arm_complementary_bnt = QRadioButton("Complementary")
        self.arm_complementary_bnt.setChecked(True)
        self.arm_kalman_bnt = QRadioButton("Kalman")
        self.arm_kalman_bnt.setCheckable(True)
        self.arm_filter_group.addButton(self.arm_complementary_bnt)
        self.arm_filter_group.addButton(self.arm_kalman_bnt)
        self.arm_filter_group.setExclusive(False)
        self.complementary_a = QLineEdit("0.98")
        self.complementary_a.setAlignment(Qt.AlignCenter)

        # # select record data content
        self.send_emg_checkbox = QCheckBox("EMG")
        self.send_emg_checkbox.setChecked(True)
        self.send_imu_checkbox = QCheckBox("IMU")
        self.send_imu_checkbox.setChecked(True)
        self.send_arm_angle_checkbox = QCheckBox("Arm Angle")
        self.send_arm_angle_checkbox.setChecked(True)
        self.send_status_checkbox = QCheckBox("Status")
        self.send_status_checkbox.setChecked(False)

        # # setting button
        self.setting_connect_bnt = QPushButton("connect")
        self.setting_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "connect.png")))
        self.setting_abort_bnt = QPushButton("abort")
        self.setting_abort_bnt.setIcon(QIcon(str(IMAGE_PATH / "abort.png")))
        self.setting_abort_bnt.setDisabled(True)
        self.apply_setting_bnt = QPushButton("Apply")

        # # setting display content
        self.display_screen = QTextEdit(self)
        self.display_screen.setReadOnly(True)
        self.display_screen.setMaximumHeight(50)

        # # setting sleep mode
        self.sleep_mode = QComboBox()
        self.sleep_mode.addItem("Normal")
        self.sleep_mode.addItem("Never")

        # # ----- define different area layout -----
        self.tab_setting.layout = QVBoxLayout(self)

        high_level_layout = QGridLayout()
        high_level_layout.addWidget(self.myo_listen_radio_bnt, 0, 0)
        high_level_layout.addWidget(self.myo_feed_radio_bnt, 0, 1)

        high_level_layout.addWidget(self.hpf_checkbox, 1, 0)
        high_level_layout.addWidget(self.ma_checkbox, 1, 1)
        high_level_layout.addWidget(QLabel("Filter Oder:"), 2, 0)
        high_level_layout.addWidget(self.hpf_filter_oder, 2, 1)
        high_level_layout.addWidget(QLabel('HPF cutoff fs:'), 3, 0)
        high_level_layout.addWidget(self.hpf_cutoff_fs, 3, 1)
        high_level_layout.addWidget(QLabel("MA Length:"), 4, 0)
        high_level_layout.addWidget(self.ma_length, 4, 1)

        high_level_layout.addWidget(QLabel("Elbow Compensate:"), 5, 0)
        high_level_layout.addWidget(self.elbow_compensate_k, 5, 1)
        high_level_layout.addWidget(self.arm_complementary_bnt, 6, 0)
        high_level_layout.addWidget(self.arm_kalman_bnt, 6, 1)
        high_level_layout.addWidget(QLabel("Complementary a:"), 7, 0)
        high_level_layout.addWidget(self.complementary_a, 7, 1)

        high_level_layout.addWidget(QLabel("Send / Save  Select"), 8, 0, 1, 2)
        high_level_layout.addWidget(self.send_arm_angle_checkbox, 9, 0)
        high_level_layout.addWidget(self.send_imu_checkbox, 9, 1)
        high_level_layout.addWidget(self.send_emg_checkbox, 10, 0)
        high_level_layout.addWidget(self.send_status_checkbox, 10, 1)

        connect_layout = QHBoxLayout(self)
        connect_layout.addWidget(self.setting_connect_bnt)
        connect_layout.addStretch()
        connect_layout.addWidget(self.setting_abort_bnt)

        setting_layout = QFormLayout()
        setting_layout.addRow(QLabel("Name:"), QLineEdit())
        setting_layout.addRow(QLabel("Sleep Mode"), self.sleep_mode)
        setting_layout.addWidget(self.apply_setting_bnt)

        # # organize layout
        self.tab_setting.layout.addLayout(high_level_layout)
        self.tab_setting.layout.addLayout(connect_layout)
        self.tab_setting.layout.addWidget(self.display_screen)
        self.tab_setting.layout.addLayout(setting_layout)
        self.tab_setting.setLayout(self.tab_setting.layout)

        # # ===== Add tabs to widget =====
        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.quit_bnt)
        self.setLayout(self.layout)

        # # Add button functions
        self.quit_bnt.clicked.connect(self.quit)

        self.tcp_connect_bnt.clicked.connect(self.tcp_connect)
        self.tcp_send_bnt.clicked.connect(self.tcp_send)

        self.arm_cali_bnt.clicked.connect(self.arm_calibration)
        self.arm_angle_bnt.clicked.connect(self.arm_angle)

        self.data_record_bnt.clicked.connect(self.data_record)
        self.file_delete_bnt.clicked.connect(self.delete_file)

        self.setting_connect_bnt.clicked.connect(self.setting_connect)
        self.setting_abort_bnt.clicked.connect(self.setting_abort)
        self.apply_setting_bnt.clicked.connect(self.apply_setting)

    def update_msg(self, receive_msg):
        # print("got msg!!!")
        self.myo_msg.append(receive_msg)

    def tcp_connect(self):
        if self.tcp_connect_bnt.isChecked():
            self.myo_msg.clear()
            print("=================")
            print("tcp connect")
            # use_hpf = self.hpf_checkbox.isChecked()
            # print(use_hpf)
            # print(self.hpf_cutoff_fs.value())
            # print(self.ma_length.value())

            # # ===== save config file =====
            self.config_dict['req_mode'] = self.req_radio_bnt.isChecked()       # type: bool
            self.config_dict['emg_filter'] = self.hpf_checkbox.isChecked()      # type: bool
            self.config_dict['moving_ave'] = self.ma_checkbox.isChecked()       # type: bool
            self.config_dict['filter_order'] = self.hpf_filter_oder.value()     # type: int
            self.config_dict['low_cutoff'] = self.hpf_cutoff_fs.value()         # type: int
            self.config_dict['window_size'] = self.ma_length.value()            # type: int
            self.config_dict['tcp_address'] = self.tcp_address.text()           # type: str
            self.config_dict['send_fs'] = self.send_fs.value()                  # type: int
            self.config_dict['elbow_compensate_k'] = float(self.elbow_compensate_k.text())      # type: float
            self.config_dict['imu_filter'] = self.arm_complementary_bnt.isChecked()             # type: bool
            self.config_dict['complementary_a'] = float(self.complementary_a.text())            # type: float
            self.config_dict['send_save'] = []

            if self.send_arm_angle_checkbox.isChecked():
                self.config_dict['send_save'].append('arm_angle')

            if self.send_imu_checkbox.isChecked():
                self.config_dict['send_save'].extend(['rpy', 'gyroscope', 'acceleration'])

            if self.send_emg_checkbox.isChecked():
                self.config_dict['send_save'].append('emg')

            if self.send_status_checkbox.isChecked():
                self.config_dict['send_save'].append('myo_status')

            with open(str(ROOT_PATH / "config" / "ini_config.yml"), "w") as config_file:
                yaml.dump(self.config_dict, config_file, default_flow_style=False)

            # # ===== button state set =====
            self.tcp_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "abort.png")))
            self.tcp_send_bnt.setDisabled(False)
            self.arm_cali_bnt.setDisabled(False)
            self.arm_angle_bnt.setDisabled(False)
            self.data_record_bnt.setDisabled(False)

            # # ===== run myo feed =====
            if self.myo_listen_radio_bnt.isChecked():
                self.myo_dongle = MyoListen()

            elif self.myo_feed_radio_bnt.isChecked():
                self.myo_dongle = MyoFeed()

            self.myo_dongle.msg_signal.connect(self.update_msg)
            if self.myo_dongle.connect_state:
                self.update_msg("connect successful")
                self.myo_dongle.start()

            else:
                self.update_msg("connect fail")
                self.tcp_connect_bnt.setChecked(False)
                self._close_connect_bnt()

        else:
            print("tcp abort")
            self.myo_dongle.stop()
            self.myo_dongle.quit()
            self.update_msg("myo disconnected")

            # # ===== button state set =====
            self._close_connect_bnt()

    def _close_connect_bnt(self):
        self.tcp_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "connect.png")))
        self.tcp_send_bnt.setChecked(False)
        self.tcp_send_bnt.setDisabled(True)
        self.arm_cali_bnt.setDisabled(True)
        self.arm_angle_bnt.setChecked(False)
        self.arm_angle_bnt.setDisabled(True)
        self.data_record_bnt.setChecked(False)
        self.data_record_bnt.setDisabled(True)

    def tcp_send(self):
        if self.tcp_send_bnt.isChecked():
            print("tcp send")
            self.myo_dongle.send(True)

        else:
            self.myo_dongle.send(False)

    def arm_calibration(self):
        self.myo_dongle.arm_calibration()

    def arm_angle(self):
        if self.arm_angle_bnt.isChecked():
            self.update_msg("get arm angle...")
            self.myo_dongle.get_arm_angle(True)
            self.arm_cali_bnt.setDisabled(False)

        else:
            self.update_msg("stop get arm angle...")
            self.myo_dongle.get_arm_angle(False)
            self.arm_cali_bnt.setDisabled(True)

    def data_record(self):
        if self.data_record_bnt.isChecked():
            save_file_name = self._get_save_file_name()
            if not Path(save_file_name).exists():
                print("make new file {}".format(save_file_name))
                self.update_msg("make new file {}".format(save_file_name))

                with open(save_file_name, 'w', newline='') as save_file:
                    writer = csv.writer(save_file)
                    writer.writerow(self.config_dict['send_save'])

            self.update_msg("start recording data...")
            self.myo_dongle.record(save_file_name, True)

        else:
            self.update_msg("stop recording data...")
            self.myo_dongle.record(None, False)

    def finished(self):
        pass

    def delete_file(self):
        save_file_name = Path(self._get_save_file_name())
        if save_file_name.exists():
            os.remove(str(save_file_name))
            self.update_msg("delete {}".format(save_file_name.name))
        else:
            print("No such file")
            self.update_msg("No such file")

    def _get_save_file_name(self):
        file_name = self.file_name.text()
        file_name_parts = file_name.split('.')
        if len(file_name_parts) > 1:
            print("using {} file format".format(file_name_parts[-1]))

        else:
            print("using default(csv) file format")
            file_name += '.csv'

        file_name = './data/' + file_name

        return file_name

    @pyqtSlot()
    def setting_connect(self):
        print("setting connect")
        self.setting_connect_bnt.setDisabled(True)
        self.setting_abort_bnt.setDisabled(False)

    @pyqtSlot()
    def setting_abort(self):
        print("setting disconnect")
        self.setting_connect_bnt.setDisabled(False)
        self.setting_abort_bnt.setDisabled(True)

    @pyqtSlot()
    def apply_setting(self):
        print("apply setting")
        _sleep_mode = self.sleep_mode.currentText()
        self.display_screen.append("set {} sleep".format(_sleep_mode))

    def quit(self):
        # # check myo hub to stop if run
        if self.myo_dongle is not None:
            print("force shutdown myo hub")
            self.myo_dongle.hub.stop(True)
            self.myo_dongle.hub.shutdown()
        QApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
