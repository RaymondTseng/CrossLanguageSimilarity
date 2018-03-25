# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import utils, data_helper
from model import LSTM
import numpy as np
import time
from functools import reduce
from operator import mul


# Model Hyper Parameters
tf.flags.DEFINE_integer('hidden_size', 300, 'hidden size in LSTM layer (default: 128)')
tf.flags.DEFINE_integer('seq_length', 20, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 6, 'classes number (default 1)')
tf.flags.DEFINE_integer('layer_num', 1, 'Number of BiLSTM layer (default: 1)')
tf.flags.DEFINE_integer('embedding_size', 300, 'embedding size')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-3, "L2 regularization lambda (default: 0.0)")

# Data Parameters
tf.flags.DEFINE_string('train_path', '/home/raymond/Downloads/data/sts-train.csv', 'train set')
tf.flags.DEFINE_string('dev_path', '/home/raymond/Downloads/data/sts-dev.csv', 'dev set')
tf.flags.DEFINE_string('test_path', '/home/raymond/Downloads/data/sts-test.csv', 'test set')
tf.flags.DEFINE_string('embedding_path', '/home/raymond/Downloads/data/glove.6B.300d.txt', 'word embedding source')
tf.flags.DEFINE_string('save_path', '../save', 'save model')
tf.flags.DEFINE_string('log_path', '../log', 'log training data')
tf.flags.DEFINE_float("train_sample_percentage", .8, "Percentage of the training data to use for validation")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 20000, "Number of training epochs (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_every", 5000, "Save model after this many steps (default: 5000)")
tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print ""


# Data Preparation
# ==================================================

# Load data
print ("Loading data...")
train_sources, train_targets, train_scores = data_helper.load_sts_data(FLAGS.train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_sts_data(FLAGS.dev_path)
test_sources, test_targets, test_scores = data_helper.load_sts_data(FLAGS.test_path)

# train_source_features, train_target_features = utils.get_all_handcraft_features(train_sources, train_targets, FLAGS.seq_length)
# dev_source_features, dev_target_features = utils.get_all_handcraft_features(dev_sources, dev_targets, FLAGS.seq_length)
# test_source_features, test_target_features = utils.get_all_handcraft_features(test_sources, test_targets, FLAGS.seq_length)

word2idx, word_embeddings = data_helper.load_embedding(FLAGS.embedding_path, True)
train_sources = utils.word2id(train_sources, word2idx, FLAGS.seq_length)
train_targets = utils.word2id(train_targets, word2idx, FLAGS.seq_length)
dev_sources = utils.word2id(dev_sources, word2idx, FLAGS.seq_length)
dev_targets = utils.word2id(dev_targets, word2idx, FLAGS.seq_length)
test_sources = utils.word2id(test_sources, word2idx, FLAGS.seq_length)
test_targets = utils.word2id(test_targets, word2idx, FLAGS.seq_length)

dev_score_probs = utils.build_porbs(dev_scores, FLAGS.class_num)
test_score_probs = utils.build_porbs(test_scores, FLAGS.class_num)




print("Train/Dev split: {:d}/{:d}".format(len(train_scores), len(dev_scores)))

time_stamp = str(int(time.time()))
# Training
# ==================================================

with tf.Graph().as_default():
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=word_embeddings.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(word_embeddings), trainable=False)

        model = LSTM(FLAGS.seq_length, FLAGS.hidden_size, FLAGS.layer_num, FLAGS.class_num,
                        FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            print(variable.name, shape)
            num_params += reduce(mul, [dim.value for dim in shape], 1)

        print('total parameters: ' + str(num_params))

        # train_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/train', session.graph)
        # dev_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/dev', session.graph)
        # merged = tf.summary.merge_all()
        # saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # training loop, for each batch

        for step in range(FLAGS.epochs_num):
            # sources_batch, targets_batch, scores_batch, source_features_batch, target_features_batch = \
            #     utils.random_batch_with_handcraft(train_sources, train_targets, train_scores, train_source_features,
            #                             train_target_features, FLAGS.batch_size)

            sources_batch, targets_batch, scores_batch = utils.random_batch(train_sources, train_targets, train_scores, FLAGS.batch_size)
            score_probs_batch = utils.build_porbs(scores_batch, FLAGS.class_num)

            ops, feed_dict = model.train_step(sources_batch, targets_batch, score_probs_batch, FLAGS.keep_prob)
            _, scores, loss = session.run(ops, feed_dict=feed_dict)
            pearson = utils.pearson(scores, scores_batch)


            print ('--- training step %s --- pearson: %.3f --- loss: %.3f ---' % (step + 1, pearson, loss))

            if (step + 1) % FLAGS.evaluate_every == 0:
                ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_score_probs)
                scores, loss = session.run(ops, feed_dict=feed_dict)
                pearson = utils.pearson(scores, dev_scores)


                print ('--- evaluation --- pearson: %.3f --- loss: %.3f ---' % (pearson, loss))


        ops, feed_dict = model.dev_step(test_sources, test_targets, test_score_probs)
        scores, loss = session.run(ops, feed_dict=feed_dict)
        pearson = utils.pearson(scores, test_scores)
        print('--- test --- pearson: %.3f --- loss: %.3f ---' % (pearson, loss))