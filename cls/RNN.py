# -*- coding:utf-8 -*-


import tensorflow as tf

class RNN:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'rnn'
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.rnn(sources, name='source')
        _, target_states = self.rnn(targets, name='target')

        outputs = tf.concat([source_states[0] - target_states[0], source_states[0] * target_states[0]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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

    def creat_rnn_cell(self):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell

    def rnn(self, x, name):
        with tf.variable_scope(name + '-rnn', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_rnn_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class GRU:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'gru'
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.rnn(sources, name='source')
        _, target_states = self.rnn(targets, name='target')

        outputs = tf.concat([source_states[0] - target_states[0], source_states[0] * target_states[0]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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

    def creat_gru_cell(self):
        rnn_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell

    def rnn(self, x, name):
        with tf.variable_scope(name + '-gru', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_gru_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class LSTM:
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
        self.scores = tf.placeholder(tf.float32, [None], name='scores')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.variable_scope('source_embedding', reuse=True):
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.lstm(sources, name='source')
        _, target_states = self.lstm(targets, name='target')

        outputs = tf.concat([source_states[0][1] - target_states[0][1], source_states[0][1] * target_states[0][1]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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

    def creat_lstm_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def lstm(self, x, name):
        with tf.variable_scope(name + '-lstm', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class BiLSTM:
    def __init__(self, seq_length, hidden_size, layer_num, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'bilstm'
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.bilstm(sources, 'source')
        _, target_states = self.bilstm(targets, 'target')

        source_outputs = tf.concat([source_states[0][0][1], source_states[1][0][1]], axis=1)
        target_outputs = tf.concat([target_states[0][0][1], target_states[1][0][1]], axis=1)

        outputs = tf.concat([source_outputs - target_outputs, source_outputs * target_outputs], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([4 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def creat_lstm_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def bilstm(self, x, name):
        with tf.variable_scope(name + '-bilstm'):

            fw_cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs, states



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

class RNNAttention:
    def __init__(self, seq_length, hidden_size, layer_num, attention_length, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'rnnattention'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.attention_length = attention_length
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.rnn(sources, name='source')
        _, target_states = self.rnn(targets, name='target')

        outputs = tf.concat([source_states[0][1] - target_states[0][1], source_states[0][1] * target_states[0][1]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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


    def creat_rnn_cell(self):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
        rnn_cell = tf.contrib.rnn.AttentionCellWrapper(rnn_cell, self.attention_length)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_cell

    def rnn(self, x, name):
        with tf.variable_scope(name + '-rnn', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_rnn_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class GRUAttention:
    def __init__(self, seq_length, hidden_size, layer_num, attention_length, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'gruattention'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.attention_length = attention_length
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.rnn(sources, name='source')
        _, target_states = self.rnn(targets, name='target')

        outputs = tf.concat([source_states[0][1] - target_states[0][1], source_states[0][1] * target_states[0][1]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # gvs = optimizer.compute_gradients(self.loss)
            # capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
            # self.optimizer = optimizer.apply_gradients(capped_gvs)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def creat_gru_cell(self):
        gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        gru_cell = tf.contrib.rnn.AttentionCellWrapper(gru_cell, self.attention_length)
        gru_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_cell, output_keep_prob=self.keep_prob)
        return gru_cell

    def rnn(self, x, name):
        with tf.variable_scope(name + '-gru', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_gru_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class LSTMAttention:
    def __init__(self, seq_length, hidden_size, layer_num, attention_length, class_num, learning_rate, l2_reg_lambda=0.):
        self.name = 'lstmattention'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.attention_length = attention_length
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.lstm(sources, name='source')
        _, target_states = self.lstm(targets, name='target')

        outputs = tf.concat([source_states[0][1] - target_states[0][1], source_states[0][1] * target_states[0][1]], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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

    def creat_lstm_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, self.attention_length)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def lstm(self, x, name):
        with tf.variable_scope(name + '-lstm', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return outputs, states

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

class BiLSTMAttention:
    def __init__(self, seq_length, hidden_size, layer_num, attention_length, class_num, learning_rate, l2_reg_lambda=0.0):
        self.name = 'bilstmattention'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.attention_length = attention_length
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
            source_embedding = tf.get_variable('source_embedding')
            sources = tf.nn.embedding_lookup(source_embedding, self.sources)

        with tf.variable_scope('target_embedding', reuse=True):
            target_embedding = tf.get_variable('target_embedding')
            targets = tf.nn.embedding_lookup(target_embedding, self.targets)

        _, source_states = self.bilstm(sources, 'source')
        _, target_states = self.bilstm(targets, 'target')

        source_outputs = tf.concat([source_states[0][0][1], source_states[1][0][1]], axis=1)
        target_outputs = tf.concat([target_states[0][0][1], target_states[1][0][1]], axis=1)

        outputs = tf.concat([source_outputs - target_outputs, source_outputs * target_outputs], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([4 * self.hidden_size, self.class_num])
            b = self.bias_variable([self.class_num])
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.matmul(outputs, W) + b
            # tradition
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
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def creat_lstm_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, self.attention_length)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def bilstm(self, x, name):
        with tf.variable_scope(name + '-bilstm'):

            fw_cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)
            bw_cell = tf.contrib.rnn.MultiRNNCell([self.creat_lstm_cell() for _ in range(self.layer_num)],
                                                  state_is_tuple=True)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs, states



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


