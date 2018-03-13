# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import utils, data_helper
from model import CNN
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



# Model Hyper Parameters
tf.flags.DEFINE_integer('filters_num', 300, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_integer('seq_length', 30, 'sequence length (default 36)')
tf.flags.DEFINE_integer('class_num', 6, 'classes number (default 1)')
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding size')
tf.flags.DEFINE_string('filter_sizes', '1', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.004, "L2 regularization lambda (default: 0.0)")

# Data Parameters
tf.flags.DEFINE_string('train_path', 'G:\data\stsbenchmark\sts-train.csv', 'train set')
tf.flags.DEFINE_string('dev_path', 'G:\data\stsbenchmark\sts-dev.csv', 'dev set')
tf.flags.DEFINE_string('test_path', 'G:\data\stsbenchmark\sts-test.csv', 'test set')
tf.flags.DEFINE_string('embedding_path', 'G:\data\/glove.6B.100d.txt', 'word embedding source')
tf.flags.DEFINE_string('save_path', '../save/', 'save model')
tf.flags.DEFINE_string('log_path', '../log/', 'log training data')
tf.flags.DEFINE_float("train_sample_percentage", .8, "Percentage of the training data to use for validation")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs_num", 10000, "Number of training epochs (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("save_every", 5000, "Save model after this many steps (default: 5000)")
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')


FLAGS = tf.flags.FLAGS
# FLAGS = FLAGS._parse_args()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


# Data Preparation
# ==================================================

# Load data
print ("Loading data...")
train_sources, train_targets, train_scores = data_helper.load_sts_data(FLAGS.train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_sts_data(FLAGS.dev_path)
test_sources, test_targets, test_scores = data_helper.load_sts_data(FLAGS.test_path)
word2idx, word_embeddings = data_helper.load_embedding(FLAGS.embedding_path, True)
train_sources = utils.word2id(train_sources, word2idx, FLAGS.seq_length)
train_targets = utils.word2id(train_targets, word2idx, FLAGS.seq_length)
dev_sources = utils.word2id(dev_sources, word2idx, FLAGS.seq_length)
dev_targets = utils.word2id(dev_targets, word2idx, FLAGS.seq_length)
test_sources = utils.word2id(test_sources, word2idx, FLAGS.seq_length)
test_targets = utils.word2id(test_targets, word2idx, FLAGS.seq_length)

# sources, targets, scores = data_helper.load_sick_data(FLAGS.data_path)
# word2idx, word_embedding = data_helper.load_embedding(FLAGS.embedding_path)
# sources = utils.word2id(sources, word2idx, FLAGS.seq_length)
# targets = utils.word2id(targets, word2idx, FLAGS.seq_length)


# Split train/test set
# 5 fold cross-validation
# sample_num = len(scores)
# fold_num = int(sample_num / 5)
# # 1 - 2, 3, 4, 5
# train_sources, dev_sources = sources[fold_num:], sources[:fold_num]
# train_targets, dev_targets = targets[fold_num:], targets[:fold_num]
# train_scores, dev_scores = scores[fold_num:], scores[:fold_num]

# 2 - 1, 3, 4, 5
# train_sources, dev_sources = sources[:fold_num] + sources[2 * fold_num:], sources[fold_num: 2 * fold_num]
# train_targets, dev_targets = targets[:fold_num] + targets[2 * fold_num:], targets[fold_num: 2 * fold_num]
# train_scores, dev_scores = np.append(scores[:fold_num], scores[2 * fold_num:]), scores[fold_num: 2 * fold_num]

# 3 - 1, 2, 4, 5
# train_sources, dev_sources = sources[: 2 * fold_num] + sources[3 * fold_num:], sources[2 * fold_num: 3 * fold_num]
# train_targets, dev_targets = targets[: 2 * fold_num] + targets[3 * fold_num:], targets[2 * fold_num: 3 * fold_num]
# train_scores, dev_scores = np.append(scores[: 2 * fold_num], scores[3 * fold_num:]), scores[2 * fold_num: 3 * fold_num]

# 4 - 1, 2, 3, 5
# train_sources, dev_sources = sources[: 3 * fold_num] + sources[4 * fold_num:], sources[3 * fold_num: 4 * fold_num]
# train_targets, dev_targets = targets[: 3 * fold_num] + targets[4 * fold_num:], targets[3 * fold_num: 4 * fold_num]
# train_scores, dev_scores = np.append(scores[: 3 * fold_num], scores[4 * fold_num:]), scores[3 * fold_num: 4 * fold_num]

# 5 - 1, 2, 3, 4
# train_sources, dev_sources = sources[: 4 * fold_num], sources[4 * fold_num:]
# train_targets, dev_targets = targets[: 4 * fold_num], targets[4 * fold_num:]
# train_scores, dev_scores = scores[: 4 * fold_num], scores[4 * fold_num:]
print("Train/Dev split: {:d}/{:d}".format(len(train_scores), len(dev_scores)))

time_stamp = str(int(time.time()))
# Training
# ==================================================

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    session = tf.Session(config=config)
    with session.as_default():
        # Define training procedure

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=word_embeddings.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(word_embeddings), trainable=True)

        model = CNN(FLAGS.seq_length, FLAGS.class_num, list(map(int, FLAGS.filter_sizes.split(','))),
                    FLAGS.filters_num, FLAGS.embedding_size, FLAGS.learning_rate, FLAGS.l2_reg_lambda)

        train_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/train', session.graph)
        dev_writer = tf.summary.FileWriter(FLAGS.log_path + time_stamp + model.name + '/dev', session.graph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # training loop, for each batch

        for step in range(FLAGS.epochs_num):
            sources_batch, targets_batch, scores_batch = utils.random_batch(train_sources, train_targets, train_scores,
                                                                            FLAGS.batch_size)

            ops, feed_dict = model.train_step(sources_batch, targets_batch, scores_batch, FLAGS.keep_prob)
            summaries, _, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
            train_writer.add_summary(summaries, global_step=step + 1)

            print ('--- training step %s --- loss: %.3f --- pearson: %.3f ---' % (step + 1, loss, pearson))

            if (step + 1) % FLAGS.evaluate_every == 0:
                ops, feed_dict = model.dev_step(dev_sources, dev_targets, dev_scores, 1.0)
                summaries, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)
                dev_writer.add_summary(summaries, global_step=step + 1)

                print ('--- evaluation --- loss: %.3f --- pearson: %.3f ---' % (loss, pearson))

        ops, feed_dict = model.dev_step(test_sources, test_targets, test_scores, 1.0)
        pearson, loss = session.run(ops, feed_dict=feed_dict)

        print('--- test --- loss: %.3f --- pearson: %.3f ---' % (loss, pearson))
            # if (step + 1) % FLAGS.save_every == 0:
            #     path = os.path.join(FLAGS.save_path, 'model-' + str(step + 1))
            #     saver.save(session, path)
            #
            #     print '--- save model --- model path: ' + path

