
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
        self.sources_length = tf.placeholder(tf.float32, [None], name='sources_length')
        self.targets_length = tf.placeholder(tf.float32, [None], name='targets_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.scores = tf.placeholder(tf.float32, [None, self.class_num], name='scores')

        self.if_train = tf.placeholder(tf.bool, name='if_train')



        # embedding layer
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)



        with tf.name_scope('convolution'):
            source_outputs = []
            target_outputs = []
            for filter_size in self.filter_sizes:
                source_outputs.append(self.horizon_convolution(sources, filter_size, 'max'))
                target_outputs.append(self.horizon_convolution(targets, filter_size, 'max', True))
            self.all_filter_num = len(self.filter_sizes) * self.filters_num

        with tf.name_scope('operation'):
            # source_outputs.append(tf.reduce_mean(sources, axis=1))
            # target_outputs.append(tf.reduce_mean(targets, axis=1))

            source_output = tf.concat(source_outputs, axis=-1)
            target_output = tf.concat(target_outputs, axis=-1)

            h_sub = tf.abs(source_output - target_output)
            h_mul = source_output * target_output

            h_sub = tf.contrib.layers.fully_connected(h_sub, self.all_filter_num, activation_fn=tf.nn.tanh,
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                      biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            h_mul = tf.contrib.layers.fully_connected(h_mul, self.all_filter_num, activation_fn=tf.nn.tanh,
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                      biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            sdv = h_sub + h_mul

            output = tf.contrib.layers.fully_connected(sdv, self.all_filter_num, activation_fn=tf.nn.tanh,
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                    biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

            output = tf.nn.dropout(output, self.keep_prob)

            self.logits = tf.contrib.layers.fully_connected(output, self.class_num, activation_fn=tf.nn.softmax,
                                                            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                            biases_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

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
            self.loss = tf.reduce_mean(kl_loss)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def make_pearson(self, v1, v2):
        mid1 = tf.reduce_mean(v1 * v2, axis=-1) - \
               tf.reduce_mean(v1, axis=-1) * tf.reduce_mean(v2, axis=-1)

        mid2 = tf.sqrt(tf.reduce_mean(tf.square(v1), axis=-1) - tf.square(tf.reduce_mean(v1, axis=-1))) * \
               tf.sqrt(tf.reduce_mean(tf.square(v2), axis=-1) - tf.square(tf.reduce_mean(v2, axis=-1)))

        return mid1 / mid2

    def horizon_convolution(self, x, filter_size, type, reuse=None):
        limit = tf.sqrt(2. / filter_size)
        conv = tf.layers.conv1d(x, self.filters_num, filter_size,
                                reuse=reuse, name='h-conv-' + type + '-%s' % filter_size,
                                kernel_initializer=tf.initializers.random_uniform(minval=-1 * limit, maxval=limit),
                                bias_initializer=tf.initializers.random_uniform(minval=-1 * limit, maxval=limit))
        activate = tf.nn.relu(conv)
        if type == 'max':
            pool = tf.layers.max_pooling1d(activate, self.seq_length - filter_size + 1, 1)
        elif type == 'avg':
            pool = tf.layers.average_pooling1d(activate, self.seq_length - filter_size + 1, 1)
        else:
            pool = -1 * tf.layers.max_pooling1d(-1 * activate, self.seq_length - filter_size + 1, 1)

        output = tf.squeeze(pool, axis=1)

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

        activate = tf.nn.relu(pool)

        output = tf.squeeze(activate, axis=[1, 2])

        return output


    def batch_norm_layer(self, x, train_phase, scope_bn):
        with tf.variable_scope(scope_bn):
            beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
            axises = list(np.arange(len(x.shape) - 1))
            batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(train_phase, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def glorot_uniform_varible(self, shape):
        f_in = shape[0]
        f_out = 0 if len(shape) == 1 else shape[-1]
        limit = np.sqrt(6. / (f_in + f_out))
        init = tf.random_uniform(shape, minval=(-1 * limit), maxval=limit)
        return tf.Variable(init)

    def zeros_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)


    def train_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        feed_dict[self.if_train] = True

        return [self.optimizer, self.loss, self.pearson], feed_dict

    def dev_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        feed_dict[self.if_train] = False

        return [self.loss, self.pearson], feed_dict

    def test_step(self, batch_sources, batch_targets, batch_scores_prob, source_length_batch, target_length_batch):
        feed_dict = {}
        feed_dict[self.sources] = batch_sources
        feed_dict[self.targets] = batch_targets
        feed_dict[self.scores] = batch_scores_prob
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.sources_length] = source_length_batch
        feed_dict[self.targets_length] = target_length_batch

        feed_dict[self.if_train] = False

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
        with tf.variable_scope('embedding', reuse=True):
            embedding = tf.get_variable('embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
            targets = tf.nn.embedding_lookup(embedding, self.targets)



        with tf.name_scope('convolution'):
            source_outputs = []
            target_outputs = []
            for filter_size in self.filter_sizes:
                source_outputs.append(self.horizon_convolution(sources, filter_size, 'max'))
                target_outputs.append(self.horizon_convolution(targets, filter_size, 'max', True))
            self.all_filter_num = len(self.filter_sizes) * self.filters_num

        with tf.name_scope('operation'):


            source_output = tf.concat(source_outputs, axis=-1)
            target_output = tf.concat(target_outputs, axis=-1)

            h_sub = tf.abs(source_output - target_output)
            h_mul = source_output * target_output

            sdv = tf.concat([h_sub, h_mul], -1)

            w1 = self.weight_varible([2 * self.all_filter_num, self.all_filter_num])
            b1 = self.weight_varible([self.all_filter_num])
            self.l2_loss += tf.nn.l2_loss(w1)
            self.l2_loss += tf.nn.l2_loss(b1)

            sdv = tf.tanh(tf.matmul(sdv, w1) + b1)
            sdv = tf.nn.dropout(sdv, self.keep_prob)

            w2 = self.weight_varible([self.all_filter_num, 1])
            b2 = self.weight_varible([1])
            self.l2_loss += tf.nn.l2_loss(w2)
            self.l2_loss += tf.nn.l2_loss(b2)



            self.logits = tf.reshape(tf.nn.sigmoid(tf.matmul(sdv, w2) + b2), [-1])

        with tf.name_scope('pearson'):
            self.pearson = self.make_pearson(self.logits, self.scores)


        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.logits - self.scores)) + self.l2_reg_lambda * self.l2_loss

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

        return [self.loss, self.pearson], feed_dict







