import tensorflow as tf
import numpy as np


def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

class CNN_PROB:
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
        self.scores_prob = tf.placeholder(tf.float32, [None, self.class_num], name='scores')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)
        with tf.name_scope('convolution'):
            source_outputs = self.convolution(sources)
            target_outputs = self.convolution(targets, True)
            self.all_filter_num = len(self.filter_sizes) * self.filters_num

        with tf.name_scope('operation'):
            source_conc = tf.concat(source_outputs, axis=-1)
            target_conc = tf.concat(target_outputs, axis=-1)

            h_sub = tf.abs(source_conc - target_conc)
            h_add = source_conc + target_conc
            h_mul = source_conc * target_conc

            w1 = self.weight_varible([self.all_filter_num, self.all_filter_num])
            b1 = self.weight_varible([self.all_filter_num])

            w2 = self.weight_varible([self.all_filter_num, self.all_filter_num])
            b2 = self.weight_varible([self.all_filter_num])

            w3 = self.weight_varible([self.all_filter_num, self.all_filter_num])
            b3 = self.weight_varible([self.all_filter_num])

            sdv = tf.matmul(h_sub, w1) + b1 + tf.matmul(h_add, w2) + b2 - (tf.matmul(h_mul, w3) + b3)

        with tf.name_scope('drop_out'):
            sdv = tf.nn.dropout(sdv, self.keep_prob)

        with tf.name_scope('output'):
            w = self.weight_varible([self.all_filter_num, self.class_num])
            b = self.weight_varible([self.class_num])

            self.logits = tf.nn.softmax(tf.matmul(sdv, w) + b)

        with tf.name_scope('loss'):
            scores = tf.clip_by_value(self.scores_prob, 1e-7, 1.)
            logits = tf.clip_by_value(self.logits, 1e-7, 1.)
            avg_distance = (tf.reduce_sum(scores * tf.log(scores / logits), axis=1) +
                            tf.reduce_sum(logits * tf.log(logits / scores), axis=1)) / 2.
            self.loss = tf.reduce_mean(avg_distance)

        with tf.name_scope('pearson'):
            norm_scores = tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, 1])
            true_scores = tf.reshape(tf.matmul(self.scores_prob, norm_scores), [-1])
            pred_scores = tf.reshape(tf.matmul(self.logits, norm_scores), [-1])

            mid1 = tf.reduce_mean(true_scores * pred_scores) - \
                   tf.reduce_mean(true_scores) * tf.reduce_mean(pred_scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(true_scores)) - tf.square(tf.reduce_mean(true_scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(pred_scores)) - tf.square(tf.reduce_mean(pred_scores)))

            self.pearson = mid1 / mid2


        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)




    def convolution(self, x, reuse=None):
        pool_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            limit = np.sqrt(6. / (filter_size + self.embedding_size))
            conv = tf.layers.conv1d(x, self.filters_num, filter_size,
                                    reuse=reuse, name='conv-%s' % filter_size,
                                    kernel_initializer=tf.random_uniform_initializer(-1 * limit, limit))
            pool = tf.layers.max_pooling1d(conv, self.seq_length - filter_size + 1, 1)
            activate = tf.nn.relu(pool)

            pool_outputs.append(tf.squeeze(activate, axis=1))
        return pool_outputs

    def weight_varible(self, shape):
        f_in = shape[0]
        f_out = 0 if len(shape) == 1 else shape[-1]
        limit = np.sqrt(6. / (f_in + f_out))
        init = tf.random_uniform(shape, minval=(-1 * limit), maxval=limit)
        return tf.Variable(init)


    def train_step(self, batch_sources, batch_targets, batch_scores_prob, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores_prob] = batch_scores_prob
        feed_dict[self.keep_prob] = keep_prob

        return [self.optimizer, self.loss, self.pearson], feed_dict

    def dev_step(self, batch_sources, batch_targets, batch_scores_prob):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores_prob] = batch_scores_prob
        feed_dict[self.keep_prob] = 1.0

        return [self.loss, self.pearson], feed_dict



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

        sources = tf.nn.dropout(sources, self.keep_prob)
        targets = tf.nn.dropout(targets, self.keep_prob)


        with tf.name_scope('convolution'):
            source_outputs = []
            target_outputs = []
            for filter_size in self.filter_sizes:
                if filter_size == self.seq_length:
                    source_outputs.append(self.horizon_convolution(sources, filter_size, 'max'))
                    target_outputs.append(self.horizon_convolution(targets, filter_size, 'max', True))
                else:
                    source_outputs.append(self.horizon_convolution(sources, filter_size, 'max'))
                    target_outputs.append(self.horizon_convolution(targets, filter_size, 'max', True))
                    source_outputs.append(self.horizon_convolution(sources, filter_size, 'avg'))
                    target_outputs.append(self.horizon_convolution(targets, filter_size, 'avg', True))
                    source_outputs.append(self.horizon_convolution(sources, filter_size, 'min'))
                    target_outputs.append(self.horizon_convolution(targets, filter_size, 'min', True))
            self.all_filter_num = len(self.filter_sizes) * self.filters_num

        with tf.name_scope('operation'):

            sims = [tf.expand_dims(self.make_pearson(source_outputs[i], target_outputs[i]), axis=-1)
                        for i in range(len(source_outputs))]

            # source_sdv = tf.reduce_sum(sources, axis=1) / tf.expand_dims(self.sources_length, 1)
            # target_sdv = tf.reduce_sum(targets, axis=1) / tf.expand_dims(self.targets_length, 1)
            #
            # sims.append(tf.expand_dims(self.make_pearson(source_sdv, target_sdv), axis=-1))

            softmax_w = tf.nn.softmax(self.weight_varible([1, len(sims)]))
            self.l2_loss += tf.nn.l2_loss(softmax_w)

            sims = tf.concat(sims, axis=-1)

            self.temp = softmax_w
            self.logits = tf.reduce_sum(sims * softmax_w, axis=-1)

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

    def horizon_convolution(self, x, filter_size, type, reuse=None):
        limit = np.sqrt(6. / (filter_size + self.embedding_size))
        conv = tf.layers.conv1d(x, self.filters_num, filter_size,
                                reuse=reuse, name='h-conv-' + type + '-%s' % filter_size,
                                kernel_initializer=tf.random_uniform_initializer(-1 * limit, limit),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda))
        if type == 'max':
            pool = tf.layers.max_pooling1d(conv, self.seq_length - filter_size + 1, 1)
        elif type == 'avg':
            pool = tf.layers.average_pooling1d(conv, self.seq_length - filter_size + 1, 1)
        else:
            pool = -1 * tf.layers.max_pooling1d(-1 * conv, self.seq_length - filter_size + 1, 1)

        activate = LeakyRelu(pool)

        output = tf.squeeze(activate, axis=1)

        output = tf.nn.dropout(output, self.keep_prob)
        return output

    def vertical_convolution(self, x, filter_size, type, reuse=None):
        x = tf.expand_dims(x, -1)
        limit = np.sqrt(6. / (filter_size + self.seq_length))
        conv = tf.layers.conv2d(x, self.filters_num, [self.seq_length, filter_size],
                                reuse=reuse, name='v-conv-' + type + '-%s' % filter_size,
                                kernel_initializer=tf.random_uniform_initializer(-1 * limit, limit),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda))
        if type == 'max':
            pool = tf.layers.max_pooling2d(conv, [1, self.embedding_size - filter_size + 1], 1)
        elif type == 'avg':
            pool = tf.layers.average_pooling2d(conv, [1, self.embedding_size - filter_size + 1], 1)
        else:
            pool = -1 * tf.layers.max_pooling2d(-1 * conv, [1, self.embedding_size - filter_size + 1], 1)

        activate = LeakyRelu(pool)

        output = tf.squeeze(activate, axis=[1, 2])

        output = tf.nn.dropout(output, self.keep_prob)
        return output
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

        return [self.loss, self.pearson, self.temp], feed_dict