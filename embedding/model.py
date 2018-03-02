import tensorflow as tf
import numpy as np

class GAN:

    def __init__(self, seq_length, class_num, filter_sizes, filters_num, embedding_size, learning_rate, l2_reg_lambda=0.0):
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
        # self.scores = tf.placeholder(tf.float32, [None], name='scores')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
        with tf.variable_scope('target_embedding', reuse=True):
            embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(embedding, self.targets)

        sources = tf.expand_dims(sources, -1)
        targets = tf.expand_dims(targets, -1)

        self.generator_sample = self.generator_conv(sources)
        self.discriminator_real = self.discriminator_conv(targets)
        self.discriminator_fake = self.discriminator_conv(tf.expand_dims(self.generator_sample, -1))

        self.discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                 (logits=self.discriminator_real,
                                                  labels=tf.ones_like(self.discriminator_real)) +
                                                 tf.nn.sigmoid_cross_entropy_with_logits
                                                 (logits=self.discriminator_fake,
                                                  labels=tf.zeros_like(self.discriminator_fake)))
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                             (logits=self.discriminator_fake,
                                              labels=tf.ones_like(self.discriminator_fake)))

        self.pearson = self.pearson_step(tf.reshape(self.generator_sample, [-1, self.seq_length * self.embedding_size]),
                                         tf.reshape(targets, [-1, self.seq_length * self.embedding_size]))

        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.discriminator_loss,
                                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                   scope='discriminator-cnn'))
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.generator_loss,
                                                        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                   scope='generator-cnn'))



    def generator_conv(self, inputs):
        with tf.variable_scope('generator-cnn'):
            all_num_filters = self.filters_num * len(self.filter_sizes)
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("generator-conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, all_num_filters])

            with tf.name_scope('output'):
                W = self.weight_variable([all_num_filters, self.seq_length * self.embedding_size])
                b = self.bias_variable([self.seq_length * self.embedding_size])
                outputs = tf.matmul(h_pool_flat, W) + b
                outputs = tf.reshape(outputs, [-1, self.seq_length, self.embedding_size])


        return outputs

    def discriminator_conv(self, inputs, reuse=False):
        with tf.variable_scope('discriminator-cnn', reuse=reuse):
            all_num_filters = self.filters_num * len(self.filter_sizes)
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("discriminator-conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, all_num_filters])

            with tf.name_scope('output'):
                W = self.weight_variable([all_num_filters, self.class_num])
                b = self.bias_variable([self.class_num])
                outputs = tf.matmul(h_pool_flat, W) + b

        return outputs

    def generator_step(self, sources_batch, targets_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        # feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.generator_optimizer, self.generator_loss], feed_dict


    def discriminator_step(self, sources_batch, targets_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.discriminator_optimizer, self.discriminator_loss, self.pearson], feed_dict

    def pearson_step(self, x1, x2):
        mid1 = tf.reduce_mean(x1 * x2) - \
               tf.reduce_mean(x1) * tf.reduce_mean(x2)

        mid2 = tf.sqrt(tf.reduce_mean(tf.square(x1)) - tf.square(tf.reduce_mean(x1))) * \
               tf.sqrt(tf.reduce_mean(tf.square(x2)) - tf.square(tf.reduce_mean(x2)))

        pearson = mid1 / mid2
        return pearson


    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


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
        with tf.variable_scope('source_embedding', reuse=True):
            embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(embedding, self.sources)
        with tf.variable_scope('target_embedding', reuse=True):
            embedding = tf.get_variable('target_embedding')
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

            source_W = self.weight_variable([2 * self.hidden_size, self.hidden_size])
            source_b = self.bias_variable([self.hidden_size])
            self.l2_loss += tf.nn.l2_loss(source_W)
            self.l2_loss += tf.nn.l2_loss(source_b)
            source_output = tf.matmul(source_output, source_W) + source_b

            target_W = self.weight_variable([2 * self.hidden_size, self.hidden_size])
            target_b = self.bias_variable([self.hidden_size])
            self.l2_loss += tf.nn.l2_loss(target_W)
            self.l2_loss += tf.nn.l2_loss(target_b)
            target_output = tf.matmul(target_output, target_W) + target_b


        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(source_output * target_output) - \
                   tf.reduce_mean(source_output) * tf.reduce_mean(target_output)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(source_output)) - tf.square(tf.reduce_mean(source_output))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(target_output)) - tf.square(tf.reduce_mean(target_output)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)


        with tf.name_scope('loss'):
            losses = tf.square(self.pearson - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)


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

class CNN:
    def __init__(self, seq_length, class_num, filter_sizes, filters_num,
                 embedding_size, learning_rate, l2_reg_lambda=0.0):
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
        self.scores = tf.placeholder(tf.float32, [None], name='scores')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        sources = tf.expand_dims(sources, -1)
        targets = tf.expand_dims(targets, -1)

        all_num_filters = self.filters_num * len(self.filter_sizes)
        with tf.name_scope('source-cnn'):
            source_pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("source-conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(sources, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    source_pooled_outputs.append(pooled)

            source_h_pool = tf.concat(source_pooled_outputs, 3)
            source_h_pool_flat = tf.reshape(source_h_pool, [-1, all_num_filters])

        with tf.name_scope('target-cnn'):
            target_pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("target-conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    self.l2_loss += tf.nn.l2_loss(W)
                    self.l2_loss += tf.nn.l2_loss(b)
                    conv = tf.nn.conv2d(targets, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.max_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    target_pooled_outputs.append(pooled)

            target_h_pool = tf.concat(target_pooled_outputs, 3)
            target_h_pool_flat = tf.reshape(target_h_pool, [-1, all_num_filters])

        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(source_h_pool_flat * target_h_pool_flat) - \
                   tf.reduce_mean(source_h_pool_flat) * tf.reduce_mean(target_h_pool_flat)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(source_h_pool_flat)) - tf.square(tf.reduce_mean(source_h_pool_flat))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(target_h_pool_flat)) - tf.square(tf.reduce_mean(target_h_pool_flat)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)


        with tf.name_scope('loss'):
            losses = tf.square(self.pearson - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)


        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
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

class CNN_Attention:
    def __init__(self, seq_length, class_num, filter_sizes, filters_num,
                 embedding_size, learning_rate, l2_reg_lambda=0.0):
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
        self.scores = tf.placeholder(tf.float32, [None], name='scores')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        # [batch, seq_length, dimension, 1]
        sources = tf.expand_dims(sources, -1)
        targets = tf.expand_dims(targets, -1)

        # [batch, dimension, seq_length, 1]
        sources = tf.transpose(sources, [0, 2, 1, 3])
        targets = tf.transpose(targets, [0, 2, 1, 3])

        S0 = self.all_pool('input-source', sources, self.seq_length, self.embedding_size)
        T0 = self.all_pool('input-source', targets, self.seq_length, self.embedding_size)

        with tf.name_scope('input_attention'):
            att_W = self.weight_variable([self.seq_length, self.embedding_size])
            self.l2_loss += tf.nn.l2_loss(att_W)

            # [batch, seq_length, seq_length]
            att_mat = self.attention_mat(sources, targets)

            # [batch, seq_length, seq_length] * [seq_length, dimension] => [batch, seq_length, dimension]
            # matrix transpose => [batch, dimension, seq_length]
            # expand dims => [batch, dimension, seq_length, 1]
            sources_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, att_W)), -1)
            targets_a = tf.expand_dims(tf.matrix_transpose(
                tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), att_W)), -1)

            # [batch, dimension, seq_length, 2]
            sources = tf.concat([sources, sources_a], axis=3)
            targets = tf.concat([targets, targets_a], axis=3)


        with tf.name_scope('cnn'):
            sources_conv_outputs = self.convolution('sources', sources)
            targets_conv_outputs = self.convolution('targets', targets)

            source_pool_outputs = []
            target_pool_outputs = []

            for i, source_conv in enumerate(sources_conv_outputs):
                target_conv = targets_conv_outputs[i]

                att_mat = self.attention_mat(source_conv, target_conv)

                source_att, target_att = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

                source_weight_pool = self.weight_pool('source', source_conv, source_att, self.filter_sizes[i])
                source_all_pool = self.all_pool('source', source_conv,
                                                self.seq_length + self.filter_sizes[i] - 1, self.filters_num)
                target_weight_pool = self.weight_pool('target', target_conv, target_att, self.filter_sizes[i])
                target_all_pool = self.all_pool('target', target_conv,
                                                self.seq_length + self.filter_sizes[i] - 1, self.filters_num)

                source_pool_outputs.append(source_all_pool)
                target_pool_outputs.append(target_all_pool)

            # self.shape1 = tf.shape(sources_conv_outputs[2])
            # self.shape2 = tf.shape(targets_conv_outputs[2])
            all_filters_num = len(self.filter_sizes) * self.filters_num
            S1 = tf.concat(source_pool_outputs, 1)
            T1 = tf.concat(target_pool_outputs, 1)



        with tf.name_scope('dropout'):
            S1 = tf.nn.dropout(S1, self.keep_prob)
            T1 = tf.nn.dropout(T1, self.keep_prob)

        with tf.name_scope('output'):
            S = tf.concat([S0, S1], axis=1)
            T = tf.concat([T0, T1], axis=1)
            mid1 = tf.reduce_mean(S * T) - tf.reduce_mean(S) * tf.reduce_mean(T)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(S)) - tf.square(tf.reduce_mean(S))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(T)) - tf.square(tf.reduce_mean(T)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)

        with tf.name_scope('loss'):
            losses = tf.square(self.pearson - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)


        with tf.name_scope('training'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob


        return [self.optimizer, self.pearson, self.loss], feed_dict


    def dev_step(self, sources_batch, targets_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.pearson, self.loss], feed_dict

    def all_pool(self, name_scope, x, pool_width, output_num):
        with tf.name_scope(name_scope + '-all_pool'):
            all_pool = tf.nn.avg_pool(x, [1, 1, pool_width, 1], [1, 1, 1, 1], 'VALID')
            all_pool = tf.reshape(all_pool, [-1, output_num])
        return all_pool

    def weight_pool(self, name_scope, x, attention, filter_size):
        with tf.name_scope(name_scope + '-weight_pool'):
            pools = []
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

            for i in range(self.seq_length):
                # [batch, filters_num, filter_size, 1], [batch, 1, filter_size, 1] => [batch, filters_num, 1, 1]
                pools.append(tf.reduce_sum(x[:, :, i:i + filter_size, :] * attention[:, :, i:i + filter_size, :],
                                           axis=2,
                                           keep_dims=True))

            # [batch, filters_num, seq_length, 1]
            weight_pool = tf.concat(pools, axis=2, name="w_ap")
        return weight_pool

    def convolution(self, name_scope, x):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope(name_scope + "-conv-%s" % filter_size):
                filter_shape = [self.embedding_size, filter_size, 2, self.filters_num]
                W = self.weight_variable(filter_shape)
                b = self.bias_variable([self.filters_num])
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                conv = tf.nn.conv2d(self.pad_for_wide_conv(x, filter_size), W, [1, 1, 1, 1], padding='VALID')
                conv = tf.nn.bias_add(conv, b)
                conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                pooled_outputs.append(conv_trans)
        return pooled_outputs

        # zero padding to inputs for wide convolution
    def pad_for_wide_conv(self, x, w):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

        return dot_products / (norm1 * norm2)

    def attention_mat(self, x1, x2):
        # x1, x2 = [batch, dimension, seq_length, 1]
        # x2 => [batch, dimension, 1, seq_length]
        # x1 - x2 = [batch, height, seq_length, seq_length]
        # [batch, seq_length, seq_length]
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
        return 1 / (1 + euclidean)


    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

