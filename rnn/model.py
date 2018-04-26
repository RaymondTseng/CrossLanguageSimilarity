
import tensorflow as tf
import numpy as np

class LSTM_PROB:
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
        self.sources_length = tf.placeholder(tf.int32, [None], name='sources_length')
        self.targets_length = tf.placeholder(tf.int32, [None], name='targets_length')
        self.scores = tf.placeholder(tf.float32, [None, self.class_num], name='scores')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)

        with tf.name_scope('lstm'):
            _, source_states = self.lstm(sources, self.sources_length)
            _, target_states = self.lstm(targets, self.targets_length)

        with tf.name_scope('operation'):
            h_sub = tf.abs(source_states[0][1] - target_states[0][1])
            h_mul = source_states[0][1] * target_states[0][1]

            h_sub = tf.contrib.layers.fully_connected(h_sub, self.hidden_size, activation_fn=tf.nn.tanh,
                                                       weights_initializer=tf.variance_scaling_initializer(
                                                           mode="fan_avg", distribution="uniform"),
                                                       biases_initializer=tf.zeros_initializer)

            h_mul = tf.contrib.layers.fully_connected(h_mul, self.hidden_size, activation_fn=tf.nn.tanh,
                                                      weights_initializer=tf.variance_scaling_initializer(
                                                          mode="fan_avg", distribution="uniform"),
                                                      biases_initializer=tf.zeros_initializer)
            sdv = h_mul + h_sub

            output = tf.contrib.layers.fully_connected(sdv, self.hidden_size, activation_fn=tf.nn.tanh,
                                                       weights_initializer=tf.variance_scaling_initializer(
                                                           mode="fan_avg", distribution="uniform"),
                                                       biases_initializer=tf.zeros_initializer)
            output = tf.nn.dropout(output, self.keep_prob)

            self.logits = tf.contrib.layers.fully_connected(output, self.class_num, activation_fn=tf.nn.softmax,
                                                            weights_initializer=tf.variance_scaling_initializer(
                                                                mode="fan_avg", distribution="uniform"),
                                                            biases_initializer=tf.zeros_initializer)


        with tf.name_scope('pearson'):
            norm_scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            true_scores = tf.reshape(tf.matmul(self.scores, norm_scores), [-1])
            pred_scores = tf.reshape(tf.matmul(self.logits, norm_scores), [-1])

            mid1 = tf.reduce_mean(true_scores * pred_scores) - \
                   tf.reduce_mean(true_scores) * tf.reduce_mean(pred_scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(true_scores)) - tf.square(tf.reduce_mean(true_scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(pred_scores)) - tf.square(tf.reduce_mean(pred_scores)))

            self.pearson = mid1 / mid2


        with tf.name_scope('loss'):
            scores = tf.clip_by_value(self.scores, 1e-10, 1.)
            logits = tf.clip_by_value(self.logits, 1e-10, 1.)
            kl_loss = (tf.reduce_sum(scores * tf.log(scores / logits), axis=1) +
                            tf.reduce_sum(logits * tf.log(logits / scores), axis=1)) / 2.0
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(kl_loss) + self.l2_reg_lambda * l2_loss



        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def creat_lstm_cell(self):
        with tf.variable_scope('lstm-cell', reuse=True):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, 6)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def lstm(self, x, seq_length):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, seq_length, dtype=tf.float32)
        return outputs, states

    def train_step(self, sources_batch, targets_batch, scores_batch, sources_length, targets_length, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.sources_length] = sources_length
        feed_dict[self.targets_length] = targets_length
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.loss, self.pearson], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch, sources_length, targets_length):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.sources_length] = sources_length
        feed_dict[self.targets_length] = targets_length
        feed_dict[self.keep_prob] = 1.0

        return [self.loss, self.pearson], feed_dict




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





