# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.):
        self.name = 'lstm'
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
        self.score_probs = tf.placeholder(tf.float32, [None, self.class_num], name='scores')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)


        _, source_states = self.lstm(sources)
        _, target_states = self.lstm(targets)

        L0 = tf.abs(source_states[0][1] - target_states[0][1])
        R0 = source_states[0][1] * target_states[0][1]



        with tf.name_scope('output'):
            w1 = self.weight_variable([self.hidden_size, self.hidden_size])
            w2 = self.weight_variable([self.hidden_size, self.hidden_size])
            b = self.bias_variable([self.hidden_size])
            S0 = tf.tanh(tf.matmul(L0, w1) + tf.matmul(R0, w2) + b)

            w3 = self.weight_variable([self.hidden_size, self.class_num])
            b3 = self.bias_variable([self.class_num])
            self.logits = tf.nn.softmax(tf.matmul(S0, w3) + b3)
            self.logits = tf.clip_by_value(self.logits, 1e-6, 1.)

        with tf.name_scope('loss'):
            losses = tf.reduce_sum(self.score_probs * tf.log(self.score_probs / self.logits), axis=1)

            regularization_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * regularization_loss
            tf.summary.scalar('loss', self.loss)

            scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            self.scores = tf.reshape(tf.matmul(self.logits, scores), [-1])


        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def creat_lstm_cell(self):
        with tf.variable_scope('lstm-cell', reuse=True):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def lstm(self, x):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.score_probs] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.scores, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.score_probs] = scores_batch
        feed_dict[self.keep_prob] = 1.0

        return [self.scores, self.loss], feed_dict




    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        scale = np.sqrt(6. / len(shape))
        initial = tf.random_uniform(shape, minval=(-1 * scale), maxval=scale)

        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        scale = np.sqrt(6. / len(shape))
        initial = tf.random_uniform(shape, minval=(-1 * scale), maxval=scale)

        return tf.Variable(initial)


