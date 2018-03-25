
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
        self.score_probs = tf.placeholder(tf.float32, [None, self.class_num], name='scores')
        # self.source_features = tf.placeholder(tf.float32, [None, self.seq_length, 39], name='source_features')
        # self.target_features = tf.placeholder(tf.float32, [None, self.seq_length, 39], name='target_features')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)

        # sources = tf.concat([sources, self.source_features], axis=2)
        # targets = tf.concat([targets, self.target_features], axis=2)

        sources = tf.expand_dims(sources, -1)
        targets = tf.expand_dims(targets, -1)
        self.all_num_filters = self.filters_num * len(self.filter_sizes)

        sources = self.convolution(sources)
        targets = self.convolution(targets, True)

        sdv = tf.concat([tf.abs(sources - targets), sources * targets], axis=1)


        with tf.name_scope('fcnn'):
            w1 = self.weight_variable([2 * self.all_num_filters, self.all_num_filters])
            b1 = self.weight_variable([self.all_num_filters])
            output1 = tf.nn.tanh(tf.matmul(sdv, w1) + b1)

            w2 = self.weight_variable([self.all_num_filters, self.class_num])
            b2 = self.weight_variable([self.class_num])
            self.logits = tf.nn.softmax(tf.matmul(output1, w2) + b2)
            self.logits = tf.clip_by_value(self.logits, 1e-6, 1.)

            # scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            # self.logits = tf.reshape(tf.matmul(output2, scores), [-1])


        with tf.name_scope('loss'):
            losses = tf.reduce_sum(self.score_probs * tf.log(self.score_probs / self.logits), axis=1)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', self.loss)

            scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            self.scores = tf.reshape(tf.matmul(self.logits, scores), [-1])


        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def convolution(self, x, reuse=None):
        with tf.variable_scope('cnn', reuse=reuse):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                W = tf.get_variable('conv-maxpool-W-%s' % filter_size, filter_shape,
                                    initializer=self.he_uniform_initializer(filter_shape))
                b = tf.get_variable('conv-maxpool-b-%s' % filter_size, [self.filters_num],
                                    initializer=self.he_uniform_initializer(filter_shape))
                conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.all_num_filters])
        return h_pool_flat


    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.score_probs] = scores_batch
        # feed_dict[self.source_features] = source_features
        # feed_dict[self.target_features] = target_features
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.scores, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.score_probs] = scores_batch
        # feed_dict[self.source_features] = source_features
        # feed_dict[self.target_features] = target_features
        feed_dict[self.keep_prob] = keep_prob

        return [self.scores, self.loss], feed_dict


    def he_uniform_initializer(self, shape):
        """Create a weight variable with appropriate initialization."""
        scale = np.sqrt(6. / len(shape))
        return tf.random_uniform_initializer(minval=(-1 * scale), maxval=scale)

    def weight_variable(self, shape):
        scale = np.sqrt(6. / len(shape))
        initial = tf.random_uniform(shape, minval=(-1 * scale), maxval=scale)

        return tf.Variable(initial)





