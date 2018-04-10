import tensorflow as tf


class Multi2One(object):
    def __init__(self, model_config):
        """
        model define for tensorflow
        """
        self.model_config = model_config
        self.dropout_rate = 0.2
        self.window_size = 20

        # # ====================== build graph ====================
        self.kinematic_2_in = tf.placeholder(tf.float32, [None, 2])
        self.kinematic_5_in = tf.placeholder(tf.float32, [None, 2])
        self.kinematic_7_in = tf.placeholder(tf.float32, [None, 2])
        self.emg_in = tf.placeholder(tf.float32, [None, self.model_config['time_length'], 160])

        self.emg_rnn_cell = tf.contrib.rnn.GRUCell(self.model_config['rnn_neurons'])
        self.emg_rnn_cell = tf.contrib.rnn.DropoutWrapper(
            self.kinematic_rnn_cell,
            input_keep_prob=1.0 - self.dropout_rate,
            output_keep_prob=1.0 - self.dropout_rate,
            state_keep_prob=1.0 - self.dropout_rate,
            variational_recurrent=False
        )

        rnn_state = (tf.zeros([self.emg_in.shape[0], self.emg_rnn_cell.state_size[0]]),
                     tf.zeros([self.emg_in.shape[0], self.emg_rnn_cell.state_size[1]]))

        # # ------ encodeer stage ------
        for time_step in range(self.window_size):
            pass

        # # ------ decoder stage -------
        for axis_out in range(3):
            pass


if __name__ == "__main__":
    pass
