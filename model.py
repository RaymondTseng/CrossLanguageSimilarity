# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, args):
        self.args = args

        self.build_network()



    def build_network(self):
        self.sources = tf.placeholder(tf.int32, [self.args['time_step'], None], name='input')
        self.targets = tf.placeholder(tf.int32, [self.args['time_step'], None], name='target')
        self.scores = tf.placeholder(tf.float32, [None], name='score')

        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding', shape=(self.args['vocabulary_size'], self.args['embedding_size']), initializer=tf.truncated_normal_initializer)
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)


        def creat_lstm_cell():
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.args['hidden_size'], forget_bias=1.0, state_is_tuple=True)
            if self.args['is_training']:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.args['keep_prob'])
            return lstm_cell

        fw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.args['layer_num'])], state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([creat_lstm_cell() for _ in range(self.args['layer_num'])], state_is_tuple=True)



        # Bi-LSTM layer
        with tf.variable_scope('bidirectional_lstm'):
            source_outputs, source_states =tf.nn.bidirectional_dynamic_rnn\
                    (fw_cell, bw_cell, sources, dtype=tf.float32, time_major=True)
            target_outputs, target_states = tf.nn.bidirectional_dynamic_rnn \
                (fw_cell, bw_cell, targets, dtype=tf.float32, time_major=True)


            # output ==> [batch_size, 2 * hidden_size]
            source_output = tf.concat([source_outputs[0][-1], source_outputs[1][-1]], axis=1)
            source_w = self.weight_variable([2 * self.args['hidden_size'], self.args['hidden_size']])
            source_b = self.bias_variable([self.args['hidden_size']])
            # output ==> [batch_size, hidden_size]
            source_output = tf.matmul(source_output, source_w) + source_b


            # output ==> [batch_size, 2 * hidden_size]
            target_output = tf.concat([target_outputs[0][-1], target_outputs[1][-1]], axis=1)
            target_w = self.weight_variable([2 * self.args['hidden_size'], self.args['hidden_size']])
            target_b = self.bias_variable([self.args['hidden_size']])
            # output ==> [batch_size, hidden_size]
            target_output = tf.matmul(target_output, target_w) + target_b



        # output ==> [batch_size, 2 * hidden_size]
        output = tf.concat([source_output, target_output], axis=1)

        softmax_w = self.weight_variable([2 * self.args['hidden_size'], self.args['class_num']])
        softmax_b = self.bias_variable([self.args['class_num']])

        # logits ==> [batch_size, class_num]
        self.logits = tf.reshape(tf.matmul(output, softmax_w) + softmax_b, [-1])


        self.loss = tf.reduce_mean(tf.square(self.logits - self.scores))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args['learning_rate']).minimize(self.loss)

        mid1 = tf.reduce_mean(self.logits * self.scores) - \
               tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

        mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
               tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

        self.pearson = mid1 / mid2







    def train_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        sources_batch = np.transpose(sources_batch)
        targets_batch = np.transpose(targets_batch)
        scores_batch = np.transpose(scores_batch)
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch


        return (self.optimizer, self.pearson, self.loss, ), feed_dict


    def test_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        sources_batch = np.transpose(sources_batch)
        targets_batch = np.transpose(targets_batch)
        scores_batch = np.transpose(scores_batch)
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch

        return (self.pearson, self.loss,), feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

