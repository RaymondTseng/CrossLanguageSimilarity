import tensorflow as tf
import numpy as np
class CNN:
    def __init__(self, seq_length, class_num, filter_sizes, filters_num,
                 embedding_size, learning_rate, l2_reg_lambda=0.0):
        self.name = 'cnn'
        self.seq_length = seq_length
        self.class_num = class_num
        self.filter_sizes = filter_sizes
        self.filters_num = filters_num
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda

        self.build_network()

    def build_network(self):
        self.sources = tf.placeholder(tf.int32, [None, self.seq_length], name='sources')
        self.targets = tf.placeholder(tf.int32, [None, self.seq_length], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.scores = tf.placeholder(tf.float32, [None], name='scores')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        inputs = tf.concat([sources, targets], axis=1)
        inputs = tf.expand_dims(inputs, -1)

        all_num_filters = self.filters_num * len(self.filter_sizes)
        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    self.l2_loss += tf.nn.l2_loss(W)
                    self.l2_loss += tf.nn.l2_loss(b)
                    conv = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.avg_pool(h, ksize=[1, self.seq_length * 2 - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, all_num_filters])


        with tf.name_scope('drop_out'):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)


        with tf.name_scope('output'):
            W = self.weight_variable([all_num_filters, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(h_drop, W) + b
            # tradition
            self.logits = tf.reshape(self.logits, [-1])


        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(self.logits * self.scores) - \
                   tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)


        with tf.name_scope('loss'):
            losses = tf.square(self.logits - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)


        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = 1.0

        return [self.pearson, self.loss], feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


class BiLSTM:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'bilstm'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda

        self.build_network()

    def build_network(self):
        self.sources = tf.placeholder(tf.int32, [None, self.seq_length], name='sources')
        self.targets = tf.placeholder(tf.int32, [None, self.seq_length], name='targets')
        self.scores = tf.placeholder(tf.float32, [None], name='scores')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)


        # Bi-LSTM layer
        with tf.name_scope('bidirectional_lstm'):

            def creat_lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
                return lstm_cell

            with tf.variable_scope('source_bilstm'):

                fw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                      state_is_tuple=True)
                bw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                      state_is_tuple=True)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sources, dtype=tf.float32)
                source_output = tf.concat([tf.slice(outputs[0], [0, self.seq_length - 1, 0], [-1, -1, -1]),
                                           tf.slice(outputs[1], [0, self.seq_length - 1, 0], [-1, -1, -1])],
                                          axis=2)

            with tf.variable_scope('target_bilstm'):
                fw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                      state_is_tuple=True)
                bw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                      state_is_tuple=True)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, targets, dtype=tf.float32)
                target_output = tf.concat([tf.slice(outputs[0], [0, self.seq_length - 1, 0], [-1, -1, -1]),
                                           tf.slice(outputs[1], [0, self.seq_length - 1, 0], [-1, -1, -1])],
                                          axis=2)
            source_output = tf.reshape(source_output, [-1, 2 * self.hidden_size])
            target_output = tf.reshape(target_output, [-1, 2 * self.hidden_size])
            output = tf.concat([source_output, target_output], axis=1)

        with tf.name_scope('output'):
            W = self.weight_variable([self.hidden_size * 4, self.class_num])
            b = self.bias_variable([self.class_num])
            self.logits = tf.matmul(output, W) + b
            self.logits = tf.reshape(self.logits, [-1])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(self.logits * self.scores) - \
                   tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)


        with tf.name_scope('loss'):
            losses = tf.square(self.logits - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)


        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = 1.0

        return [self.pearson, self.loss], feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

