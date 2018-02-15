# -*- coding:utf-8 -*-


import tensorflow as tf

class BiLSTM:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.0):
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
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)

        # Bi-LSTM layer
        with tf.name_scope('bidirectional_lstm'):

            def creat_lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
                return lstm_cell

            fw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            source_outputs, source_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sources, dtype=tf.float32)
            target_outputs, target_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, targets, dtype=tf.float32)

        with tf.name_scope('concat_output'):
            source_output = tf.concat([tf.slice(source_outputs[0], [0, self.seq_length - 1, 0], [-1, -1, -1]),
                                      tf.slice(source_outputs[1], [0, self.seq_length - 1, 0], [-1, -1, -1])], axis=2)
            target_output = tf.concat([tf.slice(target_outputs[0], [0, self.seq_length - 1, 0], [-1, -1, -1]),
                                       tf.slice(target_outputs[1], [0, self.seq_length - 1, 0], [-1, -1, -1])], axis=2)

            source_output = tf.reshape(source_output, [-1, 2 * self.hidden_size])
            target_output = tf.reshape(target_output, [-1, 2 * self.hidden_size])

            W = self.weight_variable([2 * self.hidden_size, self.hidden_size])
            b = self.bias_variable([self.hidden_size])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            source_output = tf.matmul(source_output, W) + b
            target_output = tf.matmul(target_output, W) + b

        with tf.name_scope('output'):
            # output ==> [batch_size, 4 * hidden_size]
            output = tf.concat([source_output, target_output], axis=1)

            softmax_W = self.weight_variable([2 * self.hidden_size, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(softmax_W)
            self.l2_loss += tf.nn.l2_loss(softmax_b)
            self.logits = tf.matmul(output, softmax_W) + softmax_b
            self.logits = tf.reshape(self.logits, [-1])

        with tf.name_scope('loss'):
            losses = tf.square(self.logits - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(self.logits * self.scores) - \
                   tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.loss)


        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.pearson, self.loss], feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

