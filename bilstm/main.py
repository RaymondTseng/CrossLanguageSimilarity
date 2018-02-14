# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import load
from model import Model
import os
import numpy as np

train_path = '/home/raymond/Downloads/data/sts-train.csv'
test_path = '/home/raymond/Downloads/data/sts-test.csv'
dev_path = '/home/raymond/Downloads/data/sts-dev.csv'
# train_path = 'train23.csv'
# test_path = 'train1.csv'
embedding_path = '/home/raymond/Downloads/data/glove.6B.50d.txt'
word2idx_path = 'word2idx.txt'
save_path = 'save'


args = {}

args['time_step'] = 15
args['embedding_size'] = 50
args['hidden_size'] = 128
args['keep_prob'] = 0.5
args['layer_num'] = 1
args['batch_size'] = 32
args['class_num'] = 1
args['max_grad_norm'] = 5
args['learning_rate'] = 0.001
args['epoch_num'] = 20000
args['save_time'] = 5000
args['is_training'] = True




if __name__ == '__main__':
    # load data
    if args['is_training']:
        scores, sources, targets = load.load_csv(train_path)
        dev_scores, dev_sources, dev_targets = load.load_csv(dev_path)
    else:
        scores, sources, targets = load.load_csv(test_path)

    word2idx = load.load_word2idx(word2idx_path)
    sources = load.word2id(sources, word2idx, args['time_step'])
    targets = load.word2id(targets, word2idx, args['time_step'])
    if args['is_training']:
        dev_sources = load.word2id(dev_sources, word2idx, args['time_step'])
        dev_targets = load.word2id(dev_targets, word2idx, args['time_step'])
    embedding = load.load_embedding(embedding_path)

    print 'load embedding...'
    with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', shape=[len(word2idx), args['embedding_size']])
        embedding.assign(embedding)

    args['vocabulary_size'] = len(word2idx)
    model = Model(args)

    # saver and session
    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    if args['is_training']:
        # counting variable
        global_step = 0
        print 'training...'
        for e in range(args['epoch_num']):

            sources_batch, targets_batch, scores_batch = load.random_batch(sources, targets, scores, args['batch_size'])

            ops, feed_dict = model.train_step(sources_batch, targets_batch, scores_batch)

            _, pearson, loss = session.run(ops, feed_dict=feed_dict)

            if (global_step + 1) % 100 == 0:
                print '--- training step %s --- loss: %.3f --- pearson: %.3f ---' \
                          % (global_step + 1, loss, pearson)

            if (global_step + 1) % 1000 == 0:
                dev_sources_batch, dev_targets_batch, dev_scores_batch = load.random_batch(dev_sources, dev_targets, dev_scores,
                                                                                           args['batch_size'])
                ops, feed_dict = model.test_step(dev_sources_batch, dev_targets_batch, dev_scores_batch)
                pearson, loss = session.run(ops, feed_dict=feed_dict)

                print '--- dev test --- loss: %.3f --- pearson: %.3f ---' \
                      % (loss, pearson)

            if (global_step + 1) % args['save_time'] == 0:
                path = os.path.join(save_path, 'model-' + str(global_step + 1))
                print '--- save model --- model path: ' + path
                saver.save(session, path)

            global_step += 1

    else:
        print 'testing...'
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print 'restore success!!'
        else:
            print 'restore failed!!'
            os._exit(0)

        start = 0
        end = start + args['batch_size']
        end = end if end <= len(sources) else len(sources)

        step = 0
        # predict_labels = []
        # true_labels = []
        all_pearson = []

        while True:

            sources_batch = sources[start:end]
            targets_batch = targets[start:end]
            scores_batch = scores[start:end]


            ops, feed_dict = model.test_step(sources_batch, targets_batch, scores_batch)
            pearson, loss = session.run(ops, feed_dict)
            # predict_labels.extend(prediction)
            # true_labels.extend(np.reshape(feed_dict[model.scores], [-1]))
            print '---------------'
            print pearson
            print loss
            print '---------------'
            all_pearson.append(pearson)

            step += 1

            start = end
            end = start + args['batch_size']
            if end > len(sources):
                break

        print 'mean pearson:'
        print np.mean(all_pearson, axis=0)
        # print classification_report(true_labels, predict_labels, digits=4)



