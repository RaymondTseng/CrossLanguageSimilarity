import tensorflow as tf
from utils import utils, data_helper
from utils.RandomBatchData import RandomBatchData
from model import *
from operator import mul
import numpy as np
import time



# Model Hyper Parameters
tf.flags.DEFINE_integer('hidden_size', 300, 'hidden size in LSTM layer (default: 128)')
tf.flags.DEFINE_integer('seq_length', 30, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 6, 'classes number (default 1)')
tf.flags.DEFINE_integer('layer_num', 1, 'Number of BiLSTM layer (default: 1)')
tf.flags.DEFINE_integer('embedding_size', 300, 'embedding size')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.004, "L2 regularization lambda (default: 0.0)")


# Data Parameters
tf.flags.DEFINE_string('train_path', '/home/raymond/Downloads/data/sts-train.csv', 'train data')
tf.flags.DEFINE_string('dev_path', '/home/raymond/Downloads/data/sts-dev.csv', 'dev data')
tf.flags.DEFINE_string('test_path', '/home/raymond/Downloads/data/sts-test.csv', 'test data')
tf.flags.DEFINE_string('embedding_path', '/home/raymond/Downloads/data/glove.6B.300d.txt', 'word embedding')
tf.flags.DEFINE_string('save_path', '../save/cnn.model', 'save model')
tf.flags.DEFINE_string('log_path', '../log/', 'log training data')
tf.flags.DEFINE_boolean('if_train', True, 'training or load')


# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 64, "Number of training epochs (default: 64)")
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print ""


# Data Preparation
# ==================================================

# Load data
print "Loading data..."
train_sources, train_targets, train_scores = data_helper.load_sts_data(FLAGS.train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_sts_data(FLAGS.dev_path)
test_sources, test_targets, test_scores = data_helper.load_sts_data(FLAGS.test_path)

print('train: %s dev: %s test: %s' % (len(train_sources), len(dev_sources), len(test_sources)))

print 'load embedding...'
word2idx, word_embedding = data_helper.load_embedding(FLAGS.embedding_path, True)

train_sources, train_sources_length = utils.word2id(train_sources, word2idx, FLAGS.seq_length)
train_targets, train_targets_length = utils.word2id(train_targets, word2idx, FLAGS.seq_length)


dev_sources, dev_sources_length = utils.word2id(dev_sources, word2idx, FLAGS.seq_length)
dev_targets, dev_targets_length = utils.word2id(dev_targets, word2idx, FLAGS.seq_length)

test_sources, test_sources_length = utils.word2id(test_sources, word2idx, FLAGS.seq_length)
test_targets, test_targets_length = utils.word2id(test_targets, word2idx, FLAGS.seq_length)

train_scores_prob = utils.build_porbs(train_scores, FLAGS.class_num)
dev_scores_prob = utils.build_porbs(dev_scores, FLAGS.class_num)
test_scores_prob = utils.build_porbs(test_scores, FLAGS.class_num)

# train_scores = utils.normalize_probs(train_scores)
# dev_scores = utils.normalize_probs(dev_scores)
# test_scores = utils.normalize_probs(test_scores)

time_stamp = str(int(time.time()))


# Training
# ==================================================


with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(word_embedding), trainable=False)


        model = LSTM_PROB(FLAGS.seq_length, FLAGS.hidden_size, FLAGS.layer_num, FLAGS.class_num, FLAGS.learning_rate,
                          FLAGS.l2_reg_lambda)

        saver = tf.train.Saver()

        if FLAGS.if_train:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
        else:
            saver.restore(session, FLAGS.save_path)

        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            print(variable.name)
            print(shape)
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print('total params num: %s' % num_params)

        # training loop, for each batch

        random_data = RandomBatchData(train_sources, train_targets, train_scores_prob, train_sources_length,
                                      train_targets_length, FLAGS.batch_size)
        epoch = 1
        max_dev_pearson = 0.
        max_test_pearson = 0.
        while (epoch <= FLAGS.epochs_num):
            print('----------------------------------------------------------------------------------')
            while (True):
                source_batch, target_batch, score_prob_batch, source_length_batch, target_length_batch, finish = \
                    random_data.next_batch()

                ops, feed_dict = model.train_step(source_batch, target_batch, score_prob_batch, source_length_batch,
                                                  target_length_batch, FLAGS.keep_prob)
                _, loss, pearson = session.run(ops, feed_dict=feed_dict)

                print(' --- epoch %s --- train step --- loss: %.4f --- pearson: %.4f --- ' % (epoch, loss, pearson))
                # print(w)
                if (finish):
                    ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_scores_prob,
                                                    dev_sources_length, dev_targets_length)
                    loss, pearson = session.run(ops, feed_dict=feed_dict)

                    print(' --- epoch %s --- validate step --- loss: %.4f --- pearson: %.4f --- ' % (epoch, loss, pearson))
                    if pearson > max_dev_pearson:
                        max_dev_pearson = pearson
                    break
            print('----------------------------------------------------------------------------------')

            if (epoch % 1 == 0):
                ops, feed_dict = model.dev_step(test_sources, test_targets, test_scores_prob,
                                                test_sources_length, test_targets_length)
                loss, pearson = session.run(ops, feed_dict=feed_dict)

                print(' --- test step --- loss: %.4f --- pearson: %.4f --- ' % (loss, pearson))

                if pearson > max_test_pearson:
                    max_test_pearson = pearson
            epoch += 1
            print('')
        print(' --- max dev pearson: %.4f --- ' % max_dev_pearson)
        print(' --- max test pearson: %.4f --- ' % max_test_pearson)



