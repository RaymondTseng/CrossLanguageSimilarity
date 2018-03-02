# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import utils, data_helper
from model import CNN_Attention, CNN, BiLSTM, GAN
import numpy as np



# Model Hyper Parameters
tf.flags.DEFINE_integer('filters_num', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_integer('seq_length', 72, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 2, 'classes number (default 1)')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding size')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.004, "L2 regularization lambda (default: 0.0)")

#BiLSTM
tf.flags.DEFINE_integer('hidden_num', 128, 'Number of units in hidden layer (default: 128)')
tf.flags.DEFINE_integer('layer_num', 2, 'Number of hidden layer (default: 1)')

# Data Parameters
tf.flags.DEFINE_string('data_path', '/home/raymond/Downloads/data/cls.txt', 'source data')
tf.flags.DEFINE_string('source_embedding_path', '/home/raymond/Downloads/data/glove.6B.50d.txt', 'source word embedding')
tf.flags.DEFINE_string('target_embedding_path', '/home/raymond/Downloads/data/spanish.news.50d.txt', 'target word embedding')
tf.flags.DEFINE_string('save_path', '../save', 'save model')
tf.flags.DEFINE_string('log_path', '../log', 'log training data')
tf.flags.DEFINE_float("train_sample_percentage", .8, "Percentage of the training data to use for validation")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 10000, "Number of training epochs (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_every", 5000, "Save model after this many steps (default: 5000)")
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
sources, targets, scores = data_helper.load_cross_lang_sentence_data(FLAGS.data_path)
source_word2idx, source_word_embedding = data_helper.load_embedding(FLAGS.source_embedding_path)
target_word2idx, target_word_embedding = data_helper.load_embedding(FLAGS.target_embedding_path)
sources = utils.word2id(sources, source_word2idx, FLAGS.seq_length)
targets = utils.word2id(targets, target_word2idx, FLAGS.seq_length)




# Training
# ==================================================

with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('source_embedding'):
            source_embedding = tf.get_variable('source_embedding', shape=source_word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(source_word_embedding), trainable=True)

        with tf.variable_scope('target_embedding'):
            target_embedding = tf.get_variable('target_embedding', shape=source_word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(target_word_embedding), trainable=True)

        # model = BiLSTM(FLAGS.seq_length, FLAGS.hidden_num, FLAGS.layer_num, FLAGS.class_num,
        #                FLAGS.learning_rate, FLAGS.l2_reg_lambda)
        # seq_length, class_num, filter_sizes, filters_num, embedding_size, learning_rate, l2_reg_lambda=0.0
        model = GAN(FLAGS.seq_length, FLAGS.class_num, map(int, FLAGS.filter_sizes.split(',')), FLAGS.filters_num,
                       FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        train_writer = tf.summary.FileWriter(FLAGS.log_path + '/train', session.graph)
        merged = tf.summary.merge_all()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # training loop, for each batch

        for step in range(FLAGS.epochs_num):
            sources_batch, targets_batch, scores_batch = utils.random_batch(sources, targets, scores, FLAGS.batch_size)

            ops, feed_dict = model.discriminator_step(sources_batch, targets_batch, FLAGS.keep_prob)

            _, discriminator_loss, pearson = session.run(ops, feed_dict=feed_dict)

            k = 5
            generator_loss = 0
            for i in range(k):
                ops, feed_dict = model.generator_step(sources_batch, targets_batch, FLAGS.keep_prob)
                _, temp_loss = session.run(ops, feed_dict=feed_dict)
                generator_loss += temp_loss
            print '--- training step %s --- discriminator_loss: %.3f --- generator_loss: %.3f --- pearson: %.3f' % \
                  (step + 1, discriminator_loss, generator_loss / k, pearson)



