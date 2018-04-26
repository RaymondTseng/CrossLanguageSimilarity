import tensorflow as tf
import numpy as np
class LSTM:
    def __init__(self, seq_length, embedding_size, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.):
        self.name = 'lstm'
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda

        self.build_network()

    def build_network(self):
        self.sources = tf.placeholder(tf.int32, [None, self.seq_length], name='sources')
        self.targets = tf.placeholder(tf.int32, [None, self.seq_length], name='targets')
        self.sources_length = tf.placeholder(tf.float32, [None], name='sources_length')
        self.targets_length = tf.placeholder(tf.float32, [None], name='targets_length')
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

        with tf.name_scope('lstm'):
            source_outputs, source_states = self.lstm(sources, self.sources_length)
            target_outputs, target_states = self.lstm(targets, self.targets_length, reuse=True)

        with tf.name_scope('operation'):
            source_sdv = source_states[0].h
            target_sdv = target_states[0].h

            sdv = tf.concat([source_sdv, target_sdv], axis=1)

        with tf.name_scope('output'):
            w = self.weight_varible([self.hidden_size * 2, 1])
            b = self.weight_varible([1])

            self.l2_loss += tf.nn.l2_loss(w)
            self.l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.reshape(tf.sigmoid(tf.matmul(sdv, w) + b), [-1])

            with tf.name_scope('pearson'):
                self.pearson = self.make_pearson(self.logits, self.scores)

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.scores - self.logits)) + self.l2_reg_lambda * self.l2_loss

            with tf.name_scope('train'):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def make_pearson(self, v1, v2):
        mid1 = tf.reduce_mean(v1 * v2, axis=-1) - \
               tf.reduce_mean(v1, axis=-1) * tf.reduce_mean(v2, axis=-1)

        mid2 = tf.sqrt(tf.reduce_mean(tf.square(v1), axis=-1) - tf.square(tf.reduce_mean(v1, axis=-1))) * \
               tf.sqrt(tf.reduce_mean(tf.square(v2), axis=-1) - tf.square(tf.reduce_mean(v2, axis=-1)))

        return mid1 / mid2

    def lstm(self, x, length, reuse=None):
        with tf.variable_scope('lstm', reuse=reuse):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell])

            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, sequence_length=length, dtype=tf.float32)
        return outputs, states

    def weight_varible(self, shape):
        f_in = shape[0]
        f_out = 0 if len(shape) == 1 else shape[-1]
        limit = np.sqrt(6. / (f_in + f_out))
        init = tf.random_uniform(shape, minval=(-1 * limit), maxval=limit)
        return tf.Variable(init)


    def train_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        return [self.optimizer, self.loss, self.pearson], feed_dict

    def dev_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        return [self.loss, self.pearson], feed_dict

    def test_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        return [self.loss, self.pearson], feed_dict