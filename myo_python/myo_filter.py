import numpy as np
from scipy import signal


# # ===== high pass filter =====
def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=4, two_way=True):
    b, a = butter_highpass(cutoff, fs, order=order)
    if two_way:
        y = signal.filtfilt(b, a, data, method="gust")
    else:
        y = signal.lfilter(b, a, data)
    return y


# # ===== band pass filter =====
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')

    return b, a


def buterr_bandpass_filter(data, lowcut, highcut, fs, order=4, two_way=True):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if two_way:
        y = signal.filtfilt(b, a, data)
    else:
        y = signal.lfilter(b, a, data)
    return y


# # ===== notch filter =====
def butter_notch(f0, fs, Q=30):
    """
    get notch filter parameters
    :param f0: frequency to be removed from signal (Hz)
    :param fs: sample frequency (Hz)
    :param Q: quality factor
    :return:
    """
    # # normalized frequency
    w0 = f0 / (fs/2)
    b, a = signal.iirnotch(w0, Q)

    return b, a


def butter_notch_filter(data, f0, fs, Q=30, two_way=True):
    b, a = butter_notch(f0, fs, Q)
    if two_way:
        y = signal.filtfilt(b, a, data, method="gust")
    else:
        y = signal.lfilter(b, a, data)

    return y


# # ===== def moving average =====
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


class Filter(object):
    def __init__(self, fs, filter_type, low_cut=2, high_cut=200, notch_cut=60, Q=30, order=4):
        """
        :param int fs: signal frequency
        :param str filter_type: 'lowpass', 'highpass', 'bandpass', 'notch
        :param int low_cut: low frequency cut off
        :param int high_cut: high frequency cut off
        :param int notch_cut: notch filter frequency cut off
        :param int Q: Quality factor.
            Dimensionless parameter that characterizes notch filter -3 dB bandwidth
            bw relative to its center frequency, Q = w0/bw.
        :param int order: filter order
        :param bool two-way: filter through forward and backward if true
        """
        self.fs = fs
        self.filter_type = filter_type
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.notch_cut = notch_cut
        self.Q = Q

        self.order = order

        self.b, self.a = self._obtain_filter()

    def _obtain_filter(self):
        nyq = 0.5 * self.fs
        norm_low_cut = self.low_cut / nyq
        nom_high_cut = self.high_cut / nyq

        if self.filter_type in {"lowpass", "low"}:
            print("get low pass filter")
            b, a = signal.butter(self.order, nom_high_cut, btype='lowpass', analog=False)

        elif self.filter_type in {"highpass", "high"}:
            print("get high pass filter")
            b, a = signal.butter(self.order, norm_low_cut, btype='highpass', analog=False)

        elif self.filter_type == "bandpass":
            print('Get band pass filter')
            b, a = signal.butter(self.order, [norm_low_cut, norm_low_cut], btype='highpass', analog=False)

        elif self.filter_type == "notch":
            w0 = self.notch_cut / (self.fs / 2)
            b, a = signal.iirnotch(w0, self.Q)

        else:
            raise NameError("No such filter type")

        return b, a

    def filtfilt(self, data):
        return signal.filtfilt(self.b, self.a, data, method="gust")

    def lfilter(self, data):
        return signal.lfilter(self.b, self.a, data)


class Complementary(object):
    def __init__(self, dt, angle=0, a=0.98):
        """
        complementary filter in arm angle, do not use accelerometor
        :param float dt:
        :param float angle:
        :param float a:
        """
        self.dt = dt
        self.angle = angle
        self._a = a
        self._a_cpl = 1 - a

    def get_angle(self, new_angle, gyr):
        self.angle = self._a * (self.angle + gyr * self.dt) + self._a_cpl * new_angle
        return self.angle


class Kalman(object):
    def __init__(self):
        # # We will set the variables like so, these can also be tuned by the user
        self._Q_angle = 0.001
        self._Q_bias = 0.003
        self._R_measure = 0.03

        self._angle = 0.     # # reset the angle
        self._bias = 0.      # # reset the bias
        self._rate = 0.

        self._P = [[0., 0.]] * 2

    def get_angle(self, new_angle, new_rate, dt):
        # # step 1
        self._rate = new_rate - self._bias
        self._angle += self._rate * dt

        # # Update estimation error covariance - Project the error covariance ahead
        # # step 2
        self._P[0][0] += dt * (dt * self._P[1][1] - self._P[0][1] - self._P[1][0] + self._Q_angle)
        self._P[0][1] -= dt * self._P[1][1]
        self._P[1][0] -= dt * self._P[1][1]
        self._P[1][1] += self._Q_bias * dt
        # # Discrete Kalman filter measurement update equations - Measurement Update ("Correct")
        # # Calculate Kalman gain - Compute the Kalman gain

        # # step 4
        S = self._P[0][0] + self._R_measure

        K = list()
        K.append(self._P[0][0] / S)
        K.append(self._P[1][0] / S)

        # # Calculate angle and bias - Update estimate with measurement zk (newAngle)
        # # step 3
        y = new_angle - self._angle        # # angle difference
        # # step 6
        self._angle += K[0] * y
        self._bias += K[1] * y

        # # Calculate estimation error covariance - Update the error covariance
        # # step 7
        p00_temp = self._P[0][0]
        p01_temp = self._P[0][1]

        self._P[0][0] -= K[0] * p00_temp
        self._P[0][1] -= K[0] * p01_temp
        self._P[1][0] -= K[1] * p00_temp
        self._P[1][1] -= K[1] * p01_temp

        return self._angle
