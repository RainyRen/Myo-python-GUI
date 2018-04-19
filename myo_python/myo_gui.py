# import pdb
import os
import sys
import yaml
import csv
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QTabWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QTextEdit, QLabel, QLineEdit, QRadioButton, QCheckBox, QSpinBox, QButtonGroup
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from myo_listen import MyoListen
from myo_feed import MyoFeed

ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / "data"
IMAGE_PATH = ROOT_PATH / "images"
EXP_PATH = ROOT_PATH / "exp"


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

        with open(str(ROOT_PATH / "config" / "ini_config.yml"), "r") as config_file:
            self.config_dict = yaml.load(config_file)

        # # ***** below are gui init *****
        # # ===== define whole layout =====
        self.layout = QVBoxLayout(self)

        # # ===== Initialize tab screen =====
        self.tabs = QTabWidget()
        self.tab_main = QWidget()
        self.tab_setting = QWidget()
        self.tab_calibrate = QWidget()
        # self.tabs.resize(300, 200)

        # # ===== Add tabs =====
        self.tabs.addTab(self.tab_main, "Myo Control")
        self.tabs.addTab(self.tab_setting, "Advance Setting")
        self.tabs.addTab(self.tab_calibrate, "Calibration")

        # # ===== Add main button =====
        self.quit_bnt = QPushButton("QUIT")
        self.quit_bnt.setIcon(QIcon(str(IMAGE_PATH / "quit.png")))
        self.quit_bnt.setStyleSheet("background-color: #EF5350")

        # # ========================================== Create first tab ================================================
        self.myo_connect_bnt = QPushButton("Connect")
        self.myo_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "connect.png")))
        self.myo_connect_bnt.setCheckable(True)
        self.tcp_send_bnt = QPushButton("Send")
        self.tcp_send_bnt.setIcon(QIcon(str(IMAGE_PATH / "send.png")))
        self.tcp_send_bnt.setCheckable(True)
        self.tcp_send_bnt.setDisabled(True)
        self.data_record_bnt = QPushButton("Record")
        self.data_record_bnt.setIcon(QIcon(str(IMAGE_PATH / "record.png")))
        self.data_record_bnt.setCheckable(True)
        self.data_record_bnt.setDisabled(True)

        self.arm_cali_bnt = QPushButton('Calibration')
        self.arm_cali_bnt.setIcon(QIcon(str(IMAGE_PATH / "calibration")))
        self.arm_cali_bnt.setDisabled(True)
        self.arm_angle_bnt = QPushButton('Arm Angle')
        self.arm_angle_bnt.setIcon(QIcon(str(IMAGE_PATH / "angle.png")))
        self.arm_angle_bnt.setCheckable(True)
        self.arm_angle_bnt.setDisabled(True)

        self.file_delete_bnt = QPushButton("Delete")
        self.file_delete_bnt.setIcon(QIcon(str(IMAGE_PATH / "delete.png")))

        self.estimator_bnt = QPushButton("Estimator")
        self.estimator_bnt.setIcon(QIcon(str(IMAGE_PATH / "estimator.png")))
        self.estimator_bnt.setCheckable(True)
        self.estimator_bnt.setDisabled(True)

        self.socket_address = QLineEdit("localhost:5555")
        self.file_name = QLineEdit("default")
        self.file_name.setAlignment(Qt.AlignCenter)

        self.socket_mode_group = QButtonGroup(self.tab_main)
        self.socket_tcp_mode = QRadioButton('TCP')
        self.socket_udp_mode = QRadioButton('UDP')
        self.socket_mode_group.addButton(self.socket_tcp_mode)
        self.socket_mode_group.addButton(self.socket_udp_mode)

        self.tcp_mode_group = QButtonGroup(self.tab_main)
        self.pub_radio_bnt = QRadioButton("Public Mode")
        self.req_radio_bnt = QRadioButton("Request Mode")
        self.tcp_mode_group.addButton(self.pub_radio_bnt)
        self.tcp_mode_group.addButton(self.req_radio_bnt)

        self.send_fs = QSpinBox()
        self.send_fs.setAlignment(Qt.AlignCenter)
        self.send_fs.setValue(20)
        self.send_fs.setRange(0, 200)

        self.myo_msg = QTextEdit()

        # # define different area layout
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("Socket Address:"), 1, 0)
        grid_layout.addWidget(self.socket_address, 1, 1)
        grid_layout.addWidget(self.socket_tcp_mode, 2, 0)
        grid_layout.addWidget(self.socket_udp_mode, 2, 1)
        grid_layout.addWidget(self.pub_radio_bnt, 3, 0)
        grid_layout.addWidget(self.req_radio_bnt, 3, 1)
        grid_layout.addWidget(QLabel("Send Frequency(Hz):"), 4, 0)
        grid_layout.addWidget(self.send_fs, 4, 1)
        grid_layout.addWidget(QLabel("File Name:"), 5, 0)
        grid_layout.addWidget(self.file_name, 5, 1)

        myo_tcp_layout = QHBoxLayout()
        myo_tcp_layout.addWidget(self.myo_connect_bnt)
        myo_tcp_layout.addWidget(self.tcp_send_bnt)

        arm_angle_layout = QHBoxLayout()
        arm_angle_layout.addWidget(self.arm_cali_bnt)
        arm_angle_layout.addWidget(self.arm_angle_bnt)

        file_bnt_layout = QHBoxLayout()
        file_bnt_layout.addWidget(self.data_record_bnt)
        file_bnt_layout.addWidget(self.file_delete_bnt)

        # # organize layout
        self.tab_main.layout = QVBoxLayout(self)
        self.tab_main.layout.addLayout(grid_layout)
        self.tab_main.layout.addLayout(myo_tcp_layout)
        self.tab_main.layout.addLayout(arm_angle_layout)
        self.tab_main.layout.addLayout(file_bnt_layout)
        self.tab_main.layout.addWidget(self.estimator_bnt)
        self.tab_main.layout.addWidget(self.myo_msg)
        self.tab_main.setLayout(self.tab_main.layout)

        # # ============================================================================================================
        # # ===================================== Create second tab ====================================================
        self.myo_mode_group = QButtonGroup(self.tab_setting)
        self.myo_listen_radio_bnt = QRadioButton("Listen Mode")
        self.myo_listen_radio_bnt.setChecked(True)
        self.myo_feed_radio_bnt = QRadioButton("Feed Mode")
        self.myo_mode_group.addButton(self.myo_listen_radio_bnt)
        self.myo_mode_group.addButton(self.myo_feed_radio_bnt)

        self.hpf_checkbox = QCheckBox("High Pass Filter")
        self.ma_checkbox = QCheckBox("Moving Average")

        self.hpf_filter_oder = QSpinBox()
        self.hpf_filter_oder.setAlignment(Qt.AlignCenter)
        self.hpf_filter_oder.setRange(1, 5)
        self.hpf_cutoff_fs = QSpinBox()
        self.hpf_cutoff_fs.setAlignment(Qt.AlignCenter)
        self.hpf_cutoff_fs.setRange(0, 1000)
        self.ma_length = QSpinBox()
        self.ma_length.setAlignment(Qt.AlignCenter)
        self.ma_length.setRange(1, 1000)

        # # arm muscle deformation compensate content
        self.ha_compensate_k = QLineEdit("0.0")               # # horizontal abduction
        self.ha_compensate_k.setAlignment(Qt.AlignCenter)
        self.sf_compensate_k = QLineEdit("0.0")                 # # shoulder flexion
        self.sf_compensate_k.setAlignment(Qt.AlignCenter)
        self.er_compensate_k = QLineEdit("0.0")                  # # external rotation
        self.er_compensate_k.setAlignment(Qt.AlignCenter)
        self.ef_compensate_k = QLineEdit("0.0")                 # # elbow flexion
        self.ef_compensate_k.setAlignment(Qt.AlignCenter)

        # # imu filter content
        self.arm_filter_group = QButtonGroup(self.tab_setting)
        self.arm_complementary_rbnt = QRadioButton("Complementary")
        self.arm_kalman_rbnt = QRadioButton("Kalman")
        self.arm_kalman_rbnt.setChecked(False)
        self.arm_filter_group.addButton(self.arm_complementary_rbnt)
        self.arm_filter_group.addButton(self.arm_kalman_rbnt)
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

        # # estimator model path
        self.estimator_model_path = QLineEdit('')
        self.estimator_model_path.setAlignment(Qt.AlignCenter)

        # # setting button
        self.reset_bnt = QPushButton('Reset')
        self.reset_bnt.setIcon(QIcon(str(IMAGE_PATH / 'reset.png')))
        self.apply_bnt = QPushButton("Apply")
        self.apply_bnt.setIcon(QIcon(str(IMAGE_PATH / 'apply.png')))

        # # ----- define different area layout -----
        pre_process_layout = QGridLayout()
        # # run myo mode setting layout
        pre_process_layout.addWidget(self.myo_listen_radio_bnt, 0, 0)
        pre_process_layout.addWidget(self.myo_feed_radio_bnt, 0, 1)

        # # EMG signal filter parameter setting layout
        pre_process_layout.addWidget(self.hpf_checkbox, 1, 0)
        pre_process_layout.addWidget(self.ma_checkbox, 1, 1)
        pre_process_layout.addWidget(QLabel("Filter Oder:"), 2, 0)
        pre_process_layout.addWidget(self.hpf_filter_oder, 2, 1)
        pre_process_layout.addWidget(QLabel('HPF cutoff fs:'), 3, 0)
        pre_process_layout.addWidget(self.hpf_cutoff_fs, 3, 1)
        pre_process_layout.addWidget(QLabel("MA Length:"), 4, 0)
        pre_process_layout.addWidget(self.ma_length, 4, 1)

        # # IMU filter parameter setting layout
        pre_process_layout.addWidget(self.arm_complementary_rbnt, 5, 0)
        pre_process_layout.addWidget(self.arm_kalman_rbnt, 5, 1)
        pre_process_layout.addWidget(QLabel("Complementary a:"), 6, 0)
        pre_process_layout.addWidget(self.complementary_a, 6, 1)

        separate_compensate_layout = QHBoxLayout()
        separate_compensate_layout.addStretch()
        separate_compensate_layout.addWidget(QLabel("Muscle Deformation Compensate"))
        separate_compensate_layout.addStretch()

        # # arm muscle deformation compensate setting layout
        muscle_compensate_layout = QGridLayout()
        muscle_compensate_layout.addWidget(QLabel("HA Compensate:"), 0, 0)
        muscle_compensate_layout.addWidget(self.ha_compensate_k, 0, 1)
        muscle_compensate_layout.addWidget(QLabel("SF Compensate:"), 1, 0)
        muscle_compensate_layout.addWidget(self.sf_compensate_k, 1, 1)
        muscle_compensate_layout.addWidget(QLabel("ER Compensate:"), 2, 0)
        muscle_compensate_layout.addWidget(self.er_compensate_k, 2, 1)
        muscle_compensate_layout.addWidget(QLabel("EF Compensate:"), 3, 0)
        muscle_compensate_layout.addWidget(self.ef_compensate_k, 3, 1)

        separate_text_layout = QHBoxLayout()
        separate_text_layout.addStretch()
        separate_text_layout.addWidget(QLabel("Send/Save Item"))
        separate_text_layout.addStretch()

        # # save and send item selecet layout
        choose_item_layout = QGridLayout()
        choose_item_layout.addWidget(self.send_arm_angle_checkbox, 0, 0)
        choose_item_layout.addWidget(self.send_imu_checkbox, 0, 1)
        choose_item_layout.addWidget(self.send_emg_checkbox, 1, 0)
        choose_item_layout.addWidget(self.send_status_checkbox, 1, 1)

        choose_item_layout.addWidget(QLabel("Model Folder"), 2, 0)
        choose_item_layout.addWidget(self.estimator_model_path, 2, 1)

        choose_item_layout.addWidget(self.reset_bnt, 3, 0)
        choose_item_layout.addWidget(self.apply_bnt, 3, 1)

        # # organize layout
        self.tab_setting.layout = QVBoxLayout(self)
        self.tab_setting.layout.addLayout(pre_process_layout)
        self.tab_setting.layout.addLayout(separate_compensate_layout)
        self.tab_setting.layout.addLayout(muscle_compensate_layout)
        self.tab_setting.layout.addLayout(separate_text_layout)
        self.tab_setting.layout.addLayout(choose_item_layout)
        self.tab_setting.setLayout(self.tab_setting.layout)

        # # ===========================================================================================================
        # # ========================================= Create third tab ================================================
        self.cal_init_ha = QLineEdit("0.0")
        self.cal_init_ha.setAlignment(Qt.AlignCenter)
        self.cal_init_sf = QLineEdit("0.0")
        self.cal_init_sf.setAlignment(Qt.AlignCenter)
        self.cal_init_er = QLineEdit("0.0")
        self.cal_init_er.setAlignment(Qt.AlignCenter)
        self.cal_init_ef = QLineEdit("0.0")
        self.cal_init_ef.setAlignment(Qt.AlignCenter)

        # # add layout
        cal_angle_layout = QGridLayout()
        cal_angle_layout.addWidget(QLabel("HA Initial Angle(degrees):"), 0, 0)
        cal_angle_layout.addWidget(self.cal_init_ha, 0, 1)
        cal_angle_layout.addWidget(QLabel("SF Initial Angle(degrees):"), 1, 0)
        cal_angle_layout.addWidget(self.cal_init_sf, 1, 1)
        cal_angle_layout.addWidget(QLabel("ER Initial Angle(degrees):"), 2, 0)
        cal_angle_layout.addWidget(self.cal_init_er, 2, 1)
        cal_angle_layout.addWidget(QLabel("EF Initial Angle(degrees):"), 3, 0)
        cal_angle_layout.addWidget(self.cal_init_ef, 3, 1)

        # # organize layout
        self.tab_calibrate.layout = QVBoxLayout(self)
        self.tab_calibrate.layout.addLayout(cal_angle_layout)
        self.tab_calibrate.setLayout(self.tab_calibrate.layout)

        # # ===========================================================================================================
        # # ===== Add tabs to widget =====
        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.quit_bnt)
        self.setLayout(self.layout)

        # # ===== initial all content in display =====
        self._init_display_content()

        # # Add button functions
        self.quit_bnt.clicked.connect(self.quit)

        self.myo_connect_bnt.clicked.connect(self.connect_myo)
        self.tcp_send_bnt.clicked.connect(self.socket_send)

        self.arm_cali_bnt.clicked.connect(self.arm_calibration)
        self.arm_angle_bnt.clicked.connect(self.arm_angle)

        self.data_record_bnt.clicked.connect(self.data_record)
        self.file_delete_bnt.clicked.connect(self.delete_file)

        self.estimator_bnt.clicked.connect(self.estimator)

        self.reset_bnt.clicked.connect(self.reset_default)
        self.apply_bnt.clicked.connect(self.apply_setting)

    def _init_display_content(self):
        self.socket_tcp_mode.setChecked(self.config_dict['tcp_mode'])
        self.socket_udp_mode.setChecked(not self.config_dict['tcp_mode'])
        self.req_radio_bnt.setChecked(self.config_dict['req_mode'])
        self.pub_radio_bnt.setChecked(not self.config_dict['req_mode'])
        self.hpf_checkbox.setChecked(self.config_dict['emg_filter'])
        self.ma_checkbox.setChecked(self.config_dict['moving_ave'])
        self.hpf_filter_oder.setValue(self.config_dict['filter_order'])
        self.hpf_cutoff_fs.setValue(self.config_dict['low_cutoff'])
        self.ma_length.setValue(self.config_dict['window_size'])
        self.socket_address.setText(self.config_dict['socket_address'])
        self.send_fs.setValue(self.config_dict['send_fs'])
        self.ha_compensate_k.setText(str(self.config_dict['ha_compensate_k']))
        self.sf_compensate_k.setText(str(self.config_dict['sf_compensate_k']))
        self.er_compensate_k.setText(str(self.config_dict['er_compensate_k']))
        self.ef_compensate_k.setText(str(self.config_dict['ef_compensate_k']))
        self.arm_complementary_rbnt.setChecked(self.config_dict['imu_filter'])
        self.complementary_a.setText(str(self.config_dict['complementary_a']))

        self.send_arm_angle_checkbox.setChecked('arm_angle' in self.config_dict['send_save'])
        self.send_imu_checkbox.setChecked('rpy' in self.config_dict['send_save'])
        self.send_emg_checkbox.setChecked('emg' in self.config_dict['send_save'])
        self.send_status_checkbox.setChecked('myo_status' in self.config_dict['send_save'])

        self.estimator_model_path.setText(self.config_dict['model_path'])

    def _reload_display_content(self):
        # # ===== save config file =====
        self.config_dict['tcp_mode'] = self.socket_tcp_mode.isChecked()  # type: bool
        self.config_dict['req_mode'] = self.req_radio_bnt.isChecked()  # type: bool
        self.config_dict['emg_filter'] = self.hpf_checkbox.isChecked()  # type: bool
        self.config_dict['moving_ave'] = self.ma_checkbox.isChecked()  # type: bool
        self.config_dict['filter_order'] = self.hpf_filter_oder.value()  # type: int
        self.config_dict['low_cutoff'] = self.hpf_cutoff_fs.value()  # type: int
        self.config_dict['window_size'] = self.ma_length.value()  # type: int
        self.config_dict['socket_address'] = self.socket_address.text()  # type: str
        self.config_dict['send_fs'] = self.send_fs.value()  # type: int
        self.config_dict['ha_compensate_k'] = float(self.ha_compensate_k.text())  # type: float
        self.config_dict['sf_compensate_k'] = float(self.sf_compensate_k.text())  # type: float
        self.config_dict['er_compensate_k'] = float(self.er_compensate_k.text())  # type: float
        self.config_dict['ef_compensate_k'] = float(self.ef_compensate_k.text())  # type: float
        self.config_dict['imu_filter'] = self.arm_complementary_rbnt.isChecked()  # type: bool
        self.config_dict['complementary_a'] = float(self.complementary_a.text())  # type: float

        self.config_dict['send_save'] = []
        if self.send_arm_angle_checkbox.isChecked():
            self.config_dict['send_save'].append('arm_angle')

        if self.send_imu_checkbox.isChecked():
            self.config_dict['send_save'].extend(['rpy', 'gyroscope', 'acceleration'])

        if self.send_emg_checkbox.isChecked():
            self.config_dict['send_save'].append('emg')

        if self.send_status_checkbox.isChecked():
            self.config_dict['send_save'].append('myo_status')

        self.config_dict['model_path'] = self.estimator_model_path.text()

    def update_msg(self, receive_msg):
        # print("got msg!!!")
        self.myo_msg.append(receive_msg)

    def connect_myo(self):
        """
        start myo and open tcp port for connected
        :return:
        """
        if self.myo_connect_bnt.isChecked():
            self.myo_msg.clear()
            print("=================")
            print("connect to myo hub")

            # # ===== reload setting =====
            self._reload_display_content()

            # # ===== button state set =====
            self.myo_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "abort.png")))
            self.tcp_send_bnt.setDisabled(False)
            self.arm_cali_bnt.setDisabled(False)
            self.arm_angle_bnt.setDisabled(False)
            self.data_record_bnt.setDisabled(False)

            # # ===== run myo feed =====
            if self.myo_listen_radio_bnt.isChecked():
                self.myo_dongle = MyoListen(self.config_dict)

            elif self.myo_feed_radio_bnt.isChecked():
                self.myo_dongle = MyoFeed()

            self.myo_dongle.msg_signal.connect(self.update_msg)
            if self.myo_dongle.connect_state:
                self.update_msg("connect successful")
                self.myo_dongle.start()

            else:
                self.update_msg("connect fail")
                self.myo_connect_bnt.setChecked(False)
                self._close_connect_bnt()

        else:
            self.myo_dongle.stop()
            self.myo_dongle.quit()
            time.sleep(0.5)
            # self.myo_dongle.terminate()
            self.update_msg("myo disconnected")
            self.myo_dongle = None

            # # ===== button state set =====
            self._close_connect_bnt()

    def _close_connect_bnt(self):
        self.myo_connect_bnt.setIcon(QIcon(str(IMAGE_PATH / "connect.png")))
        self.tcp_send_bnt.setChecked(False)
        self.tcp_send_bnt.setDisabled(True)
        self.arm_cali_bnt.setDisabled(True)
        self.arm_angle_bnt.setChecked(False)
        self.arm_angle_bnt.setDisabled(True)
        self.data_record_bnt.setChecked(False)
        self.data_record_bnt.setDisabled(True)
        self.estimator_bnt.setDisabled(True)

    def socket_send(self):
        if self.tcp_send_bnt.isChecked():
            print("tcp send")
            self.myo_dongle.socket_connect(True)

            # # ===== set send button icon =====
            self.tcp_send_bnt.setIcon(QIcon(str(IMAGE_PATH / "stop.png")))

        else:
            self.myo_dongle.socket_connect(False)

            # # ===== set send button icon =====
            self.tcp_send_bnt.setIcon(QIcon(str(IMAGE_PATH / "send.png")))

    def arm_calibration(self):
        init_ha = float(self.cal_init_ha.text())
        init_sf = float(self.cal_init_sf.text())
        init_er = float(self.cal_init_er.text())
        init_ef = float(self.cal_init_ef.text())
        init_angle = (init_ha, init_sf, init_er, init_ef)
        self.myo_dongle.arm_calibration(init_angle)

    def arm_angle(self):
        if self.arm_angle_bnt.isChecked():
            self.update_msg("get arm angle...")
            self.myo_dongle.get_arm_angle(True)

            # # open button can use
            self.arm_cali_bnt.setDisabled(False)
            self.estimator_bnt.setDisabled(False)

        else:
            self.update_msg("stop get arm angle...")
            self.myo_dongle.get_arm_angle(False)
            self.arm_cali_bnt.setDisabled(True)

            if self.myo_dongle.estimate_signal:
                self.myo_dongle.get_estimate_angle(False)
            self.estimator_bnt.setDisabled(True)

    def estimator(self):
        if self.estimator_bnt.isChecked():
            print('estimator turn on')
            self.update_msg('estimator turn on')
            self.myo_dongle.get_estimate_angle(True, model_path=EXP_PATH / self.estimator_model_path.text())

        else:
            print('estimator turn off')
            self.update_msg('estimator turn off')
            self.myo_dongle.get_estimate_angle(False)

    def data_record(self):
        """
        record data when clicked. Automatic check file exists, if file exists, append to it's bottom
        :return: None
        """
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
        """
        delete file shown in the screen, if file doesn't exists will do nothing
        :return:
        """
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

    # # ===== second tabs button functions =====
    """
    !!! below are only display result, not really work
    Haven't finished!!!
    """
    def reset_default(self):
        print("reset to default")
        with open(str(ROOT_PATH / "config" / "ini_default.yml"), 'r') as default_file:
            self.config_dict = yaml.load(default_file)

        self._init_display_content()
        self._save_setting()

    def apply_setting(self):
        print("apply setting")
        self._reload_display_content()
        self._save_setting()

    def _save_setting(self):
        with open(str(ROOT_PATH / "config" / "ini_config.yml"), "w") as config_file:
            yaml.dump(self.config_dict, config_file, default_flow_style=False)

    def quit(self):
        # # check myo hub to stop if run
        if self.myo_dongle is not None:
            if self.myo_dongle.hub is not None:
                print("force shutdown myo hub")
                self.myo_dongle.hub.stop(True)
                self.myo_dongle.hub.shutdown()

        QApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
