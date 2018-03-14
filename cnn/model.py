
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
        self.scores = tf.placeholder(tf.float32, [None], name='scores')
        self.source_features = tf.placeholder(tf.float32, [None, self.seq_length, 39], name='source_features')
        self.target_features = tf.placeholder(tf.float32, [None, self.seq_length, 39], name='target_features')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)

        sources = tf.concat([sources, self.source_features], axis=2)
        targets = tf.concat([targets, self.target_features], axis=2)

        sources = tf.expand_dims(sources, -1)
        targets = tf.expand_dims(targets, -1)
        self.all_num_filters = self.filters_num * len(self.filter_sizes)

        sources = self.convolution(sources)
        targets = self.convolution(targets)

        sdv = tf.concat([tf.abs(sources - targets), sources * targets], axis=1)


        with tf.name_scope('fcnn'):
            output1 = tf.contrib.layers.fully_connected(sdv, self.filters_num, activation_fn=tf.nn.tanh,
                                                        # weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                        # biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                                                        )
            output2 = tf.contrib.layers.fully_connected(output1, self.class_num, activation_fn=tf.nn.softmax,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                        biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
                                                        )

            scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            self.logits = tf.reshape(tf.matmul(output2, scores), [-1])


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

        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)


    def convolution(self, x):
        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size + 39, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.all_num_filters])
        return h_pool_flat


    def train_step(self, sources_batch, targets_batch, scores_batch, source_features, target_features, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.source_features] = source_features
        feed_dict[self.target_features] = target_features
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch, source_features, target_features, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.source_features] = source_features
        feed_dict[self.target_features] = target_features
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


