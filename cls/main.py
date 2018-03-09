import tensorflow as tf
from utils import utils, data_helper
from model import CNN, BiLSTM
import numpy as np
import time



# Model Hyper Parameters
tf.flags.DEFINE_integer('filters_num', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_integer('seq_length', 36, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 1, 'classes number (default 1)')
tf.flags.DEFINE_integer('embedding_size', 50, 'embedding size')
tf.flags.DEFINE_string('filter_sizes', '3,4,5,6,7', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.004, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer('hidden_size', 128, 'the number of hidden units (default 128)')
tf.flags.DEFINE_integer('layer_num', 1, 'the number of hidden layer (default 1)')


# Data Parameters
tf.flags.DEFINE_string('train_path', '/home/raymond/Downloads/all_cross-lingual_data/STS.train.es-en', 'train data')
tf.flags.DEFINE_string('dev_path', '/home/raymond/Downloads/all_cross-lingual_data/STS.dev.a.es-en', 'dev data')
tf.flags.DEFINE_string('source_embedding_path', '/home/raymond/Downloads/data/model.es.50.txt', 'source word embedding')
tf.flags.DEFINE_string('target_embedding_path', '/home/raymond/Downloads/data/model.en.50.txt', 'target word embedding')
tf.flags.DEFINE_string('save_path', '../save/', 'save model')
tf.flags.DEFINE_string('log_path', '../log/', 'log training data')


# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 60000, "Number of training epochs (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_every", 5000, "Save model after this many steps (default: 5000)")
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')


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

source_word2idx, source_word_embedding = data_helper.load_embedding(FLAGS.source_embedding_path, False)
target_word2idx, target_word_embedding = data_helper.load_embedding(FLAGS.target_embedding_path, False)

train_sources = utils.word2id(train_sources, source_word2idx, FLAGS.seq_length)
train_targets = utils.word2id(train_targets, target_word2idx, FLAGS.seq_length)

dev_sources = utils.word2id(dev_sources, source_word2idx, FLAGS.seq_length)
dev_targets = utils.word2id(dev_targets, target_word2idx, FLAGS.seq_length)

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

        model = CNN(FLAGS.seq_length, FLAGS.class_num, map(int, FLAGS.filter_sizes.split(',')), FLAGS.filters_num,
                       FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        # model = BiLSTM(FLAGS.seq_length, FLAGS.hidden_size, FLAGS.layer_num, FLAGS.class_num,
        #                FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        train_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/train', session.graph)
        dev_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/dev', session.graph)
        merged = tf.summary.merge_all()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # training loop, for each batch

        for step in range(FLAGS.epochs_num):
            sources_batch, targets_batch, scores_batch = utils.random_batch(train_sources, train_targets, train_scores, FLAGS.batch_size)

            ops, feed_dict = model.train_step(sources_batch, targets_batch, scores_batch, FLAGS.keep_prob)

            summary, _, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
            train_writer.add_summary(summary, global_step=step + 1)

            print '--- training step %s --- loss: %.3f --- pearson: %.3f' % (step + 1, loss, pearson)

            if (step + 1) % FLAGS.evaluate_every == 0:
                ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_scores)
                summary, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
                dev_writer.add_summary(summary, global_step=step + 1)
                print '--- evaluate step %s --- loss: %.3f --- pearson: %.3f' % (step + 1, loss, pearson)
