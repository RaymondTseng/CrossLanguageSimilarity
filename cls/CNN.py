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


        self.all_num_filters = self.filters_num * len(self.filter_sizes)
        sources = self.convolution(sources)
        targets = self.convolution(targets)

        sdv = tf.concat([tf.abs(sources - targets), sources * targets], axis=1)
        # sdv = tf.concat([sources, targets], axis=1)

        with tf.name_scope('drop_out'):
            h_drop = tf.nn.dropout(sdv, self.keep_prob)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.all_num_filters, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # self.logits = tf.sigmoid(tf.matmul(h_drop, W) + b)
            self.logits = tf.matmul(h_drop, W) + b

            # self.logits = tf.matmul(tf.nn.softmax(self.logits), tf.reshape(tf.range(self.class_num, dtype=tf.float32), [self.class_num, -1]))

            self.logits = tf.reshape(self.logits, [-1])

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.scores, logits=self.logits)
            # losses = tf.square(self.logits - self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)

        self.logits = tf.sigmoid(self.logits)

        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(self.logits * self.scores) - \
                   tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)

        with tf.name_scope('accuracy'):
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.logits - self.scores))
            tf.summary.scalar('accuracy', self.accuracy)



        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)


    def convolution(self, x):
        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.nn.avg_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.all_num_filters])
        return h_pool_flat

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.optimizer, self.pearson, self.loss, self.accuracy], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = 1.0

        return [self.pearson, self.loss, self.accuracy], feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

class CNNAttention:
    def __init__(self, seq_length, class_num, filter_sizes, filters_num,
                 embedding_size, attention_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'cnnattention'
        self.seq_length = seq_length
        self.class_num = class_num
        self.filter_sizes = filter_sizes
        self.filters_num = filters_num
        self.embedding_size = embedding_size
        self.attention_num = attention_num
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


        self.all_num_filters = self.filters_num * len(self.filter_sizes)
        # [batch, 3, filters_num]
        sources = self.convolution(sources)
        targets = self.convolution(targets)

        # sources = self.attention(sources)
        # targets = self.attention(targets)

        sdv = tf.concat([tf.abs(sources - targets), sources * targets], axis=1)

        with tf.name_scope('drop_out'):
            h_drop = tf.nn.dropout(sdv, self.keep_prob)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.all_num_filters, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(h_drop, W) + b
            self.logits = tf.reshape(self.logits, [-1])

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.scores, logits=self.logits)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self.loss)

        self.logits = tf.sigmoid(self.logits)

        with tf.name_scope('pearson'):
            mid1 = tf.reduce_mean(self.logits * self.scores) - \
                   tf.reduce_mean(self.logits) * tf.reduce_mean(self.scores)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.logits)) - tf.square(tf.reduce_mean(self.logits))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores)))

            self.pearson = mid1 / mid2
            tf.summary.scalar('pearson', self.pearson)

        with tf.name_scope('accuracy'):
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.logits - self.scores))
            tf.summary.scalar('accuracy', self.accuracy)



        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)

    def attention(self, x):
        # x ==> [batch, conv_num, filters_num]
        with tf.name_scope('attention'):
            attn_w = self.weight_variable([self.filters_num, self.attention_num])
            attn_b = self.bias_variable([self.attention_num])
            attn_v = self.weight_variable([self.attention_num, 1])
            attn_activate = tf.tanh(tf.einsum('ijk,kl->ijl', x, attn_w) + attn_b)
            # [batch, 3]
            attn_prob = tf.nn.softmax(tf.squeeze(tf.einsum('ijk,kl->ijl', attn_activate, attn_v), 2))
            attn_out = tf.squeeze(tf.matmul(tf.expand_dims(attn_prob, 1), x), 1)
        return attn_out


    def convolution(self, x):
        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, self.filters_num]
                    W = self.weight_variable(filter_shape)
                    b = self.bias_variable([self.filters_num])
                    conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = self.attention(tf.squeeze(h, 2))
                    # pooled = tf.nn.avg_pool(h, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                    #                         strides=[1, 1, 1, 1], padding='VALID')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 1)
        return h_pool

    def train_step(self, sources_batch, targets_batch, scores_batch, keep_prob):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = keep_prob

        return [self.optimizer, self.pearson, self.loss, self.accuracy], feed_dict


    def dev_step(self, sources_batch, targets_batch, scores_batch):
        feed_dict = {}
        feed_dict[self.sources] = sources_batch
        feed_dict[self.targets] = targets_batch
        feed_dict[self.scores] = scores_batch
        feed_dict[self.keep_prob] = 1.0

        return [self.pearson, self.loss, self.accuracy], feed_dict

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)




