# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import load, sick_helper
from model import Model
import os
import numpy as np

train_path = '/home/raymond/Downloads/data/SICK-train.txt'
test_path = '/home/raymond/Downloads/data/sts-test.csv'
dev_path = '/home/raymond/Downloads/data/SICK-dev.txt'
# train_path = 'train23.csv'
# test_path = 'train1.csv'
embedding_path = '/home/raymond/Downloads/data/glove.6B.50d.txt'
word2idx_path = '../word2idx.txt'
save_path = '../save'
log_path = '../log'


args = {}

args['time_step'] = 36
args['embedding_size'] = 50
args['filter_sizes'] = [3, 4, 5]
args['num_filters'] = 128
args['keep_prob'] = 0.5
args['batch_size'] = 64
args['class_num'] = 1
args['learning_rate'] = 0.001
args['epoch_num'] = 20000
args['save_time'] = 5000
args['is_training'] = True

if __name__ == '__main__':
    # load data
    if args['is_training']:
        scores, sources, targets = sick_helper.load_csv(train_path)
        dev_scores, dev_sources, dev_targets = sick_helper.load_csv(dev_path)
    else:
        scores, sources, targets = load.load_csv(test_path)

    word2idx = load.load_word2idx(word2idx_path)
    sources = load.word2id(sources, word2idx, args['time_step'])
    targets = load.word2id(targets, word2idx, args['time_step'])
    if args['is_training']:
        dev_sources = load.word2id(dev_sources, word2idx, args['time_step'])
        dev_targets = load.word2id(dev_targets, word2idx, args['time_step'])
    word_embedding = load.load_embedding(embedding_path)

    print 'load embedding...'
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', shape=[len(word2idx), args['embedding_size']],
                                    initializer=tf.constant_initializer(word_embedding), trainable=True)

    args['vocabulary_size'] = len(word2idx)
    model = Model(args)

    # saver, session and summaries
    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_path + '/train', session.graph)
    dev_writer = tf.summary.FileWriter(log_path + '/dev', session.graph)
    session.run(tf.global_variables_initializer())


    if args['is_training']:
        # counting variable
        global_step = 0
        print 'training...'
        for e in range(args['epoch_num']):

            sources_batch, targets_batch, scores_batch = load.random_batch(sources, targets, scores, args['batch_size'])

            ops, feed_dict = model.train_step(sources_batch, targets_batch, scores_batch, args['keep_prob'])

            summary, _, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)

            train_writer.add_summary(summary, global_step)

            if (global_step + 1) % 1 == 0:
                print '--- training step %s --- loss: %.3f --- pearson: %.3f ---' \
                          % (global_step + 1, loss, pearson)

            if (global_step + 1) % 100 == 0:
                ops, feed_dict = model.test_step(dev_sources, dev_targets, dev_scores, 1.0)
                summary, pearson, loss = session.run([merged] + ops, feed_dict=feed_dict)

                dev_writer.add_summary(summary, global_step)

                print '--- dev test --- loss: %.3f --- pearson: %.3f ---' \
                      % (loss, pearson)

            # if (global_step + 1) % args['save_time'] == 0:
            #     path = os.path.join(save_path, 'model-' + str(global_step + 1))
            #     print '--- save model --- model path: ' + path
            #     saver.save(session, path)

            global_step += 1

        train_writer.close()
        dev_writer.close()
