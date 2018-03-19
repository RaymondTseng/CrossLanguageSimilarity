import tensorflow as tf
class ConvLSTM:
    def __init__(self, seq_length, hidden_size, layer_num, filter_size, embedding_size, class_num, learning_rate, l2_reg_lambda=0.):
        self.name = 'convlstm'
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.filter_size = filter_size
        self.embedding_size = embedding_size
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

        _, source_states = self.conv_lstm(tf.expand_dims(sources, -1), name='source')
        _, target_states = self.conv_lstm(tf.expand_dims(targets, -1), name='target')

        last_source_state = self.average_pool(tf.expand_dims(source_states[0][1], -1))
        last_target_state = self.average_pool(tf.expand_dims(target_states[0][1], -1))
        outputs = tf.concat([last_source_state - last_target_state, last_source_state * last_target_state], axis=1)


        with tf.name_scope('output'):
            W = self.weight_variable([2 * self.embedding_size, self.class_num])
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

    def average_pool(self, x):
        avg_pool = tf.nn.avg_pool(x, [1, 1, self.hidden_size, 1], [1, 1, 1, 1], padding='VALID')
        avg_pool = tf.squeeze(avg_pool, [2, 3])
        return avg_pool

    def creat_conv_lstm_cell(self):
        conv_lstm_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1, input_shape=[50, 1], output_channels=self.hidden_size,
                                                     kernel_shape=[self.filter_size])
        conv_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=conv_lstm_cell, output_keep_prob=self.keep_prob)
        return conv_lstm_cell

    def conv_lstm(self, x, name):
        with tf.variable_scope(name + '-conv_lstm', reuse=False):
            cell = tf.contrib.rnn.MultiRNNCell([self.creat_conv_lstm_cell() for _ in range(self.layer_num)],
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