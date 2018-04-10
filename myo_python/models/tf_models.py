import tensorflow as tf


class Multi2One(object):
    def __init__(self, model_config):
        """
        model define for tensorflow
        """
        self.model_config = model_config
        self.dropout_rate = 0.2

        # # ===== build graph =====
        self.kinematic_in = tf.placeholder(tf.float32, [None, self.model_config['time_length'], 15])
        self.emg_in = tf.placeholder(tf.float32, [None, self.model_config['time_length'], 16])

        self.kinematic_rnn_cell = tf.contrib.rnn.GRUCell(self.model_config['rnn_neurons'])
        self.kinematic_rnn_cell = tf.contrib.rnn.DropoutWrapper(
            self.kinematic_rnn_cell,
            input_keep_prob=1.0 - self.dropout_rate,
            output_keep_prob=1.0 - self.dropout_rate,
            state_keep_prob=1.0 - self.dropout_rate,
            variational_recurrent=False
        )

        self.emg_rnn_cell = tf.contrib.rnn.BasicGRUCellLSTMCell()


if __name__ == "__main__":
    pass
