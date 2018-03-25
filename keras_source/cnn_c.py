import data_helper
from utils import utils
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda, Add
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Conv2D, MaxPooling2D, AveragePooling1D, BatchNormalization
from keras.models import Model
import keras.backend as kb
from keras.optimizers import *
import numpy as np

# 1016
train_path = '/home/raymond/Downloads/all_cross-lingual_data/STS.train.a.es-en'
dev_path = '/home/raymond/Downloads/all_cross-lingual_data/STS.dev.a.es-en'
test_path = '/home/raymond/Downloads/all_cross-lingual_data/STS.dev.b.es-en'
source_embedding_path = '/home/raymond/Downloads/es-en/esvec.300d.txt'
target_embedding_path = '/home/raymond/Downloads/es-en/envec.300d.txt'

seq_length = 20
class_num = 6
embedding_size = 300
filter_sizes = [1, 2, 3]
filter_num = 300
batch_size = 64
epochs_num = 300
# regularizer_rate = 0.04
drop_out_rate = 0.5
version = 'v2'

print('loading data...')
train_sources, train_targets, train_scores = data_helper.load_cross_lang_sentence_data(train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_cross_lang_sentence_data(dev_path)
test_sources, test_targets, test_scores = data_helper.load_cross_lang_sentence_data(test_path)

source_word2idx, source_word_embeddings = data_helper.load_embedding(source_embedding_path, True)
target_word2idx, target_word_embeddings = data_helper.load_embedding(target_embedding_path, True)

train_sources = utils.word2id(train_sources, source_word2idx, seq_length)
train_targets = utils.word2id(train_targets, target_word2idx, seq_length)
dev_sources = utils.word2id(dev_sources, source_word2idx, seq_length)
dev_targets = utils.word2id(dev_targets, target_word2idx, seq_length)
test_sources = utils.word2id(test_sources, source_word2idx, seq_length)
test_targets = utils.word2id(test_targets, target_word2idx, seq_length)

train_score_probs = utils.build_porbs(train_scores, class_num)
dev_score_probs = utils.build_porbs(dev_scores, class_num)
test_score_probs = utils.build_porbs(test_scores, class_num)

print('build model...')

source_input = Input(batch_shape=(None, seq_length))
target_input = Input(batch_shape=(None, seq_length))

source_embedding = Embedding(len(source_word2idx),
                            embedding_size,
                            input_length=seq_length,
                            weights=[source_word_embeddings],
                            trainable=False)

target_embedding = Embedding(len(target_word2idx),
                            embedding_size,
                            input_length=seq_length,
                            weights=[target_word_embeddings],
                            trainable=False)

source = source_embedding(source_input)
target = target_embedding(target_input)

if version == 'v1':
    source_pool_output = []
    target_pool_output = []
    all_filter_num = len(filter_sizes) * filter_num
    for filter_size in filter_sizes:
        conv = Conv1D(filter_num, filter_size, activation='relu')

        max_pool = MaxPooling1D(seq_length - filter_size + 1)
        reshape = Reshape([filter_num])

        source_pool_output.append(reshape(max_pool(conv(source))))
        target_pool_output.append(reshape(max_pool(conv(target))))

    source_conc = Concatenate()(source_pool_output)
    target_conc = Concatenate()(target_pool_output)

    abs = Lambda(lambda x: kb.abs(x))
    h_sub = abs(Subtract()([source_conc, target_conc]))
    h_mul = Multiply()([source_conc, target_conc])

    sdv = Concatenate()([h_sub, h_mul])
    drop_out = Dropout(drop_out_rate)
    sdv = drop_out(sdv)
    w1 = Dense(all_filter_num, activation='tanh')
    output1 = w1(sdv)

    w2 = Dense(class_num, activation='softmax')
    logits = w2(output1)
elif version == 'v2':
    source_pool_output = []
    target_pool_output = []
    all_filter_num = len(filter_sizes) * filter_num
    for filter_size in filter_sizes:
        conv = Conv1D(filter_num, filter_size, activation='relu')

        max_pool = MaxPooling1D(seq_length - filter_size + 1)
        reshape = Reshape([filter_num])

        source_pool_output.append(reshape(max_pool(conv(source))))
        target_pool_output.append(reshape(max_pool(conv(target))))

    source_conc = Concatenate()(source_pool_output)
    target_conc = Concatenate()(target_pool_output)

    abs = Lambda(lambda x: kb.abs(x))
    h_sub = abs(Subtract()([source_conc, target_conc]))
    h_add = Add()([source_conc, target_conc])
    h_mul = Multiply()([source_conc, target_conc])

    drop_out = Dropout(drop_out_rate)
    h_sub = drop_out(h_sub)
    h_add = drop_out(h_add)
    h_mul = drop_out(h_mul)
    w1 = Dense(all_filter_num, activation='tanh')
    w2 = Dense(all_filter_num, activation='tanh')
    w3 = Dense(all_filter_num, activation='tanh')
    output1 = Add()([w1(h_add), w2(h_sub)])
    output1 = Subtract()([output1, w3(h_mul)])

    w3 = Dense(class_num, activation='softmax')
    logits = w3(output1)
elif version == 'v3':
    source_pool_output = []
    target_pool_output = []
    all_filter_num = len(filter_sizes) * filter_num
    for filter_size in filter_sizes:
        conv = Conv1D(filter_num, filter_size, activation='relu')
        bn = BatchNormalization()
        max_pool = MaxPooling1D(seq_length - filter_size + 1)
        reshape = Reshape([filter_num])

        source_pool_output.append(reshape(max_pool(bn(conv(source)))))
        target_pool_output.append(reshape(max_pool(bn(conv(target)))))

    source_conc = Concatenate()(source_pool_output)
    target_conc = Concatenate()(target_pool_output)

    def make_cos_sim(x):
        norm1 = kb.sqrt(kb.sum(kb.square(x[0]), axis=1))
        norm2 = tf.sqrt(kb.sum(kb.square(x[1]), axis=1))
        dot_products = kb.sum(x[0] * x[1], axis=1)

        return dot_products / (norm1 * norm2)
    avg_pool = AveragePooling1D(seq_length)
    reshape = Reshape([filter_num])
    source_avg = reshape(avg_pool(source))
    target_avg = reshape(avg_pool(target))


    abs = Lambda(lambda x: kb.abs(x))
    h1_sub = abs(Subtract()([source_conc, target_conc]))
    h1_add = Add()([source_conc, target_conc])
    h1_mul = Multiply()([source_conc, target_conc])

    h2_sub = abs(Subtract()([source_avg, target_avg]))
    h2_add = Add()([source_avg, target_avg])
    h2_mul = Multiply()([source_avg, target_avg])

    drop_out = Dropout(drop_out_rate)
    h1_sub = drop_out(h1_sub)
    h1_add = drop_out(h1_add)
    h1_mul = drop_out(h1_mul)

    h2_sub = drop_out(h2_sub)
    h2_add = drop_out(h2_add)
    h2_mul = drop_out(h2_mul)

    w1 = Dense(filter_num, activation='tanh')
    w2 = Dense(filter_num, activation='tanh')
    w3 = Dense(filter_num, activation='tanh')
    output1 = Add()([w1(h1_add), w2(h1_sub)])
    output1 = Subtract()([output1, w3(h1_mul)])

    output2 = Add()([h2_add, h2_sub])
    output2 = Subtract()([output1, h2_mul])

    output = Concatenate()([output1, output2])

    w3 = Dense(class_num, activation='softmax')
    logits = w3(output)

model = Model(inputs=[source_input, target_input], outputs=logits)

def kl_distance(y_true, y_pred):
    y_true = kb.clip(y_true, 1e-6, 1.)
    y_pred = kb.clip(y_pred, 1e-6, 1.)
    avg_distance = (kb.sum(y_true * kb.log(y_true / y_pred), axis=1) +
                    kb.sum(y_pred * kb.log(y_pred / y_true), axis=1)) / 2.0
    return kb.mean(avg_distance)

def js_distance(y_true, y_pred):
    y_true = kb.clip(y_true, 1e-6, 1.)
    y_pred = kb.clip(y_pred, 1e-6, 1.)
    M = (y_true + y_pred) / 2.0
    avg_distance = 0.5 * kb.sum(y_true * kb.log(y_true / M), axis=1) + 0.5 * kb.sum(y_pred * kb.log(y_pred / M), axis=1)
    return kb.mean(avg_distance)

def pearson(y_true, y_pred):
    scores = kb.reshape(K.arange(0, 6, dtype='float32'), [class_num, 1])
    true_scores = kb.reshape(kb.dot(y_true, scores), [-1])
    pred_scores = kb.reshape(kb.dot(y_pred, scores), [-1])

    mid1 = kb.mean(true_scores * pred_scores) - \
           kb.mean(true_scores) * kb.mean(pred_scores)

    mid2 = kb.sqrt(kb.mean(kb.square(true_scores)) - kb.square(kb.mean(true_scores))) * \
           kb.sqrt(kb.mean(kb.square(pred_scores)) - kb.square(kb.mean(pred_scores)))

    pearson = mid1 / mid2
    return pearson


model.compile(optimizer='Adam', loss=kl_distance, metrics=[pearson])

model.fit([train_sources, train_targets], train_score_probs, epochs=epochs_num, batch_size=batch_size,
          validation_data=([dev_sources, dev_targets], dev_score_probs))

results = model.evaluate([dev_sources, dev_targets], dev_score_probs, batch_size=len(dev_score_probs))
print('dev loss: ' + str(results[0]) + ' - dev pearson: ' + str(results[1]))

results = model.evaluate([test_sources, test_targets], test_score_probs, batch_size=len(test_score_probs))
print('test loss: ' + str(results[0]) + ' - test pearson: ' + str(results[1]))




