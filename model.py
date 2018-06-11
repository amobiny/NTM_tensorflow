import numpy as np
import tensorflow as tf


class NTMOneShotLearningModel:
    def __init__(self, args):
        if args.label_type == 'one_hot':
            args.output_dim = args.n_classes
        elif args.label_type == 'five_hot':
            args.output_dim = 25

        self.x_image = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,
                                                               args.seq_length,
                                                               args.image_width * args.image_height])
        self.x_label = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,
                                                               args.seq_length,
                                                               args.output_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[args.batch_size,
                                                         args.seq_length,
                                                         args.output_dim])

        if args.model == 'LSTM':
            def rnn_cell(rnn_size):
                return tf.contrib.rnn.BasicLSTMCell(rnn_size)

            cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'NTM':
            import ntm.ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                    read_head_num=args.read_head_num,
                                    write_head_num=args.write_head_num,
                                    addressing_mode='content_and_location',
                                    output_dim=args.output_dim)
        elif args.model == 'MANN':
            import ntm.mann_cell as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num)
        elif args.model == 'MANN2':
            import ntm.mann_cell_2 as mann_cell
            cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
                                      head_num=args.read_head_num)

        state = cell.zero_state(args.batch_size, tf.float32)
        self.state_list = [state]  # For debugging
        self.o = []
        for t in range(args.seq_length):
            output, state = cell(tf.concat([self.x_image[:, t, :], self.x_label[:, t, :]], axis=1), state)
            # output, state = cell(self.y[:, t, :], state)
            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [args.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)
            if args.label_type == 'one_hot':
                output = tf.nn.softmax(output, dim=1)
            elif args.label_type == 'five_hot':
                output = tf.stack([tf.nn.softmax(o) for o in tf.split(output, 5, axis=1)], axis=1)
            self.o.append(output)
            self.state_list.append(state)
        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)

        eps = 1e-8
        if args.label_type == 'one_hot':
            self.loss = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2]))
        elif args.label_type == 'five_hot':
            self.loss = -tf.reduce_mean(
                tf.reduce_sum(tf.stack(tf.split(self.y, 5, axis=2), axis=2) * tf.log(self.o + eps)
                              , axis=[1, 2, 3]))
        self.o = tf.reshape(self.o, shape=[args.batch_size, args.seq_length, -1])
        self.loss_summary = tf.summary.scalar('Loss', self.loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
