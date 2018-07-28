import tensorflow as tf
import numpy as np


class NTMOneShotLearningModel:
    def __init__(self, args, inference=False):
        input_dim = 32 if args.emg_raw else 176

        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, args.seq_length, input_dim]
        )
        self.x_label = tf.placeholder(
            dtype=tf.float32,
            shape=[None, args.seq_length, args.output_dim]
        )

        if args.one_target:
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.output_dim])
        else:
            self.y = tf.placeholder(
                dtype=tf.float32,
                shape=[None, args.seq_length, args.output_dim])

        if args.model == 'LSTM':
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            from .ntm import ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    read_head_num=args.read_head_num,
                                    write_head_num=args.write_head_num,
                                    addressing_mode='content_and_location',
                                    output_dim=args.output_dim,
                                    batch_size=args.batch_size)
        elif args.model == 'MANN':
            from .ntm import mann_cell as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num, batch_size=args.batch_size)
        elif args.model == 'MANN2':
            from .ntm import mann_cell_2 as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num, batch_size=args.batch_size)

        else:
            raise ImportError("No such model")

        # state = cell.zero_state(args.batch_size, tf.float32)
        state = cell.state
        self.state_list = [state]   # For debugging
        self.o = []
        for t in range(args.seq_length):
            if inference and t > args.seq_length - args.future_time:
                output, state = cell(tf.concat([self.x[:, t, :], self.o[-1]], axis=1), state)
            else:
                output, state = cell(tf.concat([self.x[:, t, :], self.x_label[:, t, :]], axis=1), state)

            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

            self.o.append(output)
            self.state_list.append(state)
        if args.one_target:
            self.o = self.o[-1]
            self.o = tf.identity(self.o, name='output')
        else:
            self.o = tf.stack(self.o, axis=1, name='output')
        self.state_list.append(state)

        eps = 1e-8
        # cross entropy function
        # self.learning_loss = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2]))
        self.learning_loss = tf.losses.mean_squared_error(self.y, self.o)

        # self.o = tf.reshape(self.o, shape=[args.batch_size, args.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(
            #     learning_rate=args.learning_rate, momentum=0.9, decay=0.95
            # )
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss)


class MPAN:
    def __init__(self, args):
        # # define input Placeholder
        self.x_kin = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 16])
        self.x_emg = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 16])
        self.x_label = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, args.output_dim])

        if args.one_target:
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.output_dim])
        else:
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, args.output_dim])

        # # select controller kernel
        if args.model == 'LSTM':
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            from .ntm import ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    read_head_num=args.read_head_num,
                                    write_head_num=args.write_head_num,
                                    addressing_mode='content_and_location',
                                    output_dim=args.output_dim,
                                    batch_size=args.batch_size)
        elif args.model == 'MANN':
            from .ntm import mann_cell as mann_cell
            cell = mann_cell.MANNCellA(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num, batch_size=args.batch_size)
        elif args.model == 'MANN2':
            from .ntm import mann_cell_2 as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num, batch_size=args.batch_size)

        else:
            raise ImportError("No such model")

        # state = cell.zero_state(args.batch_size, tf.float32)
        state = cell.state
        self.state_list = [state]   # For debugging
        self.o = []
        for t in range(args.seq_length):
            output, state = cell(self.x_kin[:, t, :], self.x_emg[:, t, ...], self.x_label[:, t, :], state)

            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

            self.o.append(output)
            self.state_list.append(state)
        if args.one_target:
            self.o = self.o[-1]
            self.o = tf.identity(self.o, name='output')
        else:
            self.o = tf.stack(self.o, axis=1, name='output')
        self.state_list.append(state)

        eps = 1e-8
        # cross entropy function
        # self.learning_loss = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2]))
        self.learning_loss = tf.losses.mean_squared_error(self.y, self.o)

        # self.o = tf.reshape(self.o, shape=[args.batch_size, args.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(
            #     learning_rate=args.learning_rate, momentum=0.9, decay=0.95
            # )
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss)


if __name__ == "__main__":
    pass
