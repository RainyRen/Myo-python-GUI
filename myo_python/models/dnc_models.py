import tensorflow as tf
# import sonnet as snt

from .dnc import dnc


class DNCRegression:
    def __init__(self, args):
        input_dim = 32 if args.emg_raw else 176

        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, args.seq_length, input_dim]
        )
        self.x_label = tf.placeholder(
            dtype=tf.float32,
            shape=[None, args.seq_length, args.output_dim]
        )
        self.y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, args.seq_length, args.output_dim])

        access_config = {
            "memory_size": args.memory_size,
            "word_size": args.word_size,
            "num_reads": args.num_read_heads,
            "num_writes": args.num_write_heads,
        }
        controller_config = {
            "hidden_size": args.hidden_size,
        }
        clip_value = args.clip_value

        dnc_core = dnc.DNC(access_config, controller_config, args.output_dim, clip_value)
        initial_state = dnc_core.initial_state(args.batch_size)

        x_with_label = tf.concat([self.x, self.x_label], axis=-1)

        self.output_sequence, _ = tf.nn.dynamic_rnn(
            cell=dnc_core,
            inputs=x_with_label,
            time_major=False,
            initial_state=initial_state)

        self.o = tf.identity(self.output_sequence, name='output')
        self.learning_loss = tf.losses.mean_squared_error(self.y, self.o)

        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        # Set up optimizer with global norm clipping.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.learning_loss, trainable_variables), args.max_grad_norm)

        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        optimizer = tf.train.RMSPropOptimizer(
            args.learning_rate, epsilon=args.optimizer_epsilon)
        self.train_step = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)


if __name__ == '__main__':
    pass
