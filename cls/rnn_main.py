import tensorflow as tf
from utils import utils, data_helper
from utils.RandomBatchData import RandomBatchData
from RNN import *
from operator import mul
import numpy as np
import time



# Model Hyper Parameters
tf.flags.DEFINE_integer('hidden_size', 300, 'hidden size in LSTM layer (default: 128)')
tf.flags.DEFINE_integer('seq_length', 30, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 6, 'classes number (default 1)')
tf.flags.DEFINE_integer('layer_num', 1, 'the number of layers (default 1)')
tf.flags.DEFINE_integer('embedding_size', 300, 'embedding size')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.004, "L2 regularization lambda (default: 0.0)")


# Data Parameters
tf.flags.DEFINE_string('train_path', '/home/raymond/Downloads/all_cross-lingual_data/STS.train.es-en', 'train data')
tf.flags.DEFINE_string('dev_path', '/home/raymond/Downloads/all_cross-lingual_data/STS.dev.es-en', 'dev data')
tf.flags.DEFINE_string('test_path', '/home/raymond/Downloads/all_cross-lingual_data/STS.test.a.es-en', 'test data')
tf.flags.DEFINE_string('source_embedding_path', '/home/raymond/Downloads/es-en/esvec.300d.txt', 'source word embedding')
tf.flags.DEFINE_string('target_embedding_path', '/home/raymond/Downloads/es-en/envec.300d.txt', 'target word embedding')
tf.flags.DEFINE_string('save_path', '../save/cnn.model', 'save model')
tf.flags.DEFINE_string('log_path', '../log/', 'log training data')
tf.flags.DEFINE_boolean('if_train', True, 'training or load')


# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 300, "Number of training epochs (default: 64)")
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
train_sources, train_targets, train_scores = data_helper.load_cross_lang_sentence_data(FLAGS.train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_cross_lang_sentence_data(FLAGS.dev_path)
test_sources, test_targets, test_scores = data_helper.load_cross_lang_sentence_data(FLAGS.test_path)

print 'load source embedding...'
source_word2idx, source_word_embedding = data_helper.load_embedding(FLAGS.source_embedding_path, True)
print 'load target embedding...'
target_word2idx, target_word_embedding = data_helper.load_embedding(FLAGS.target_embedding_path, True)

train_sources, train_sources_length = utils.word2id(train_sources, source_word2idx, FLAGS.seq_length)
train_targets, train_targets_length = utils.word2id(train_targets, target_word2idx, FLAGS.seq_length)

dev_sources, dev_sources_length = utils.word2id(dev_sources, source_word2idx, FLAGS.seq_length)
dev_targets, dev_targets_length = utils.word2id(dev_targets, target_word2idx, FLAGS.seq_length)

test_sources, test_sources_length = utils.word2id(test_sources, source_word2idx, FLAGS.seq_length)
test_targets, test_targets_length = utils.word2id(test_targets, target_word2idx, FLAGS.seq_length)

# train_scores_prob = utils.build_porbs(train_scores, FLAGS.class_num)
# dev_scores_prob = utils.build_porbs(dev_scores, FLAGS.class_num)
# test_scores_prob = utils.build_porbs(test_scores, FLAGS.class_num)

train_scores = utils.normalize_probs(train_scores)
dev_scores = utils.normalize_probs(dev_scores)
test_scores = utils.normalize_probs(test_scores)

time_stamp = str(int(time.time()))


# Training
# ==================================================


with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('source_embedding'):
            source_embedding = tf.get_variable('source_embedding', shape=source_word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(source_word_embedding), trainable=False)

        with tf.variable_scope('target_embedding'):
            target_embedding = tf.get_variable('target_embedding', shape=target_word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(target_word_embedding), trainable=False)

        model = LSTM(FLAGS.seq_length, FLAGS.embedding_size, FLAGS.hidden_size, FLAGS.layer_num, FLAGS.class_num,
                        FLAGS.learning_rate, FLAGS.l2_reg_lambda)

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

        random_data = RandomBatchData(train_sources, train_targets, train_scores, train_sources_length,
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
                    ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_scores,
                                                    dev_sources_length, dev_targets_length)
                    loss, pearson = session.run(ops, feed_dict=feed_dict)

                    print(' --- epoch %s --- validate step --- loss: %.4f --- pearson: %.4f --- ' % (epoch, loss, pearson))
                    if pearson > max_dev_pearson:
                        max_dev_pearson = pearson
                    break
            print('----------------------------------------------------------------------------------')

            if (epoch % 5 == 0):
                ops, feed_dict = model.test_step(test_sources, test_targets, test_scores,
                                                test_sources_length, test_targets_length)
                loss, pearson = session.run(ops, feed_dict=feed_dict)

                print(' --- test step --- loss: %.4f --- pearson: %.4f --- ' % (loss, pearson))

                if pearson > max_test_pearson:
                    max_test_pearson = pearson
            epoch += 1
            print('')
        print(' --- max dev pearson: %.4f --- ' % max_dev_pearson)
        print(' --- max test pearson: %.4f --- ' % max_test_pearson)
        ops, feed_dict = model.test_step(test_sources, test_targets, test_scores,
                                        test_sources_length, test_targets_length)
        loss, pearson,  = session.run(ops, feed_dict=feed_dict)
        print(' --- test step --- loss: %.4f --- pearson: %.4f --- ' % (loss, pearson))

        # saver.save(session, FLAGS.save_path)

