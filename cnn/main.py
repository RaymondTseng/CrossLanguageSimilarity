# -*- coding:utf-8 -*-
#! /usr/bin/env python
import tensorflow as tf
from utils import load, sick_helper
from model import CNN
import os



# Model Hyper Parameters
tf.flags.DEFINE_integer('filters_num', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_integer('seq_length', 36, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 1, 'classes number (default 1)')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding size')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")

# Data Parameters
tf.flags.DEFINE_string('data_path', '/home/raymond/Downloads/data/SICK-train.txt', 'data source')
tf.flags.DEFINE_string('embedding_path', '/home/raymond/Downloads/data/glove.6B.50d.txt', 'word embedding source')
tf.flags.DEFINE_string('save_path', '../save', 'save model')
tf.flags.DEFINE_string('log_path', '../log', 'log training data')
tf.flags.DEFINE_float("train_sample_percentage", .9, "Percentage of the training data to use for validation")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 40000, "Number of training epochs (default: 200)")
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
sources, targets, scores = sick_helper.load_csv(FLAGS.data_path)
word2idx, word_embedding = load.load_embedding(FLAGS.embedding_path)
sources = load.word2id(sources, word2idx, FLAGS.seq_length)
targets = load.word2id(targets, word2idx, FLAGS.seq_length)
sample_num = len(scores)
train_num = int(sample_num * FLAGS.train_sample_percentage)

# Split train/test set
# TODO: This is very crude, should use cross-validation
train_sources, dev_sources = sources[:train_num], sources[train_num:]
train_targets, dev_targets = targets[:train_num], targets[train_num:]
train_scores, dev_scores = scores[:train_num], scores[train_num:]
print "Train/Dev split: {:d}/{:d}".format(len(train_scores), len(dev_scores))

# Training
# ==================================================

with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=word_embedding.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(word_embedding), trainable=True)

        model = CNN(FLAGS.seq_length, FLAGS.class_num, list(map(int, FLAGS.filter_sizes.split(','))),
                    FLAGS.filters_num, FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        train_writer = tf.summary.FileWriter(FLAGS.log_path + '/train', session.graph)
        dev_writer = tf.summary.FileWriter(FLAGS.log_path + '/dev', session.graph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # training loop, for each batch

        for step in range(FLAGS.epochs_num):
            sources_batch, targets_batch, scores_batch = load.random_batch(sources, targets, scores, FLAGS.batch_size)

            ops, feed_dict = model.train_step(sources_batch, targets_batch, scores_batch, FLAGS.keep_prob)
            summaries, _, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
            train_writer.add_summary(summaries, global_step=step + 1)

            print '--- training step %s --- loss: %.3f --- pearson: %.3f ---' % (step + 1, loss, pearson)

            if (step + 1) % FLAGS.evaluate_every == 0:
                ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_scores, 1.0)
                summaries, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
                dev_writer.add_summary(summaries, global_step=step + 1)

                print '--- evaluation --- loss: %.3f --- pearson: %.3f ---' % (loss, pearson)

            if (step + 1) % FLAGS.save_every == 0:
                path = os.path.join(FLAGS.save_path, 'model-' + str(step + 1))
                saver.save(session, path)

                print '--- save model --- model path: ' + path

