
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
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 1.0

        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, tf.transpose(self.sources))
            targets = tf.nn.embedding_lookup(embedding, tf.transpose(self.targets))

        inputs = tf.concat([sources, targets], axis=1)
        inputs = tf.expand_dims(inputs, -1)

        with tf.variable_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.args['filter_sizes']):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.args['embedding_size'], 1, self.args['num_filters']]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.args['num_filters']])
                    conv = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.args['time_step'] * 2 - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            all_num_filters = self.args['num_filters'] * len(self.args['filter_sizes'])
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, all_num_filters])

        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope('output'):
            W = self.weight_variable([all_num_filters, self.args['class_num']])
            b = self.bias_variable([self.args['class_num']])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(h_drop, W) + b
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
            tf.summary.scalar('pearson', self.pearson)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.args['learning_rate']).minimize(self.loss)

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        sources_batch = np.transpose(sources_batch)
        targets_batch = np.transpose(targets_batch)
        scores_batch = np.transpose(scores_batch)
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def test_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        sources_batch = np.transpose(sources_batch)
        targets_batch = np.transpose(targets_batch)
        scores_batch = np.transpose(scores_batch)
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