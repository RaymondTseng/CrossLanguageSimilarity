
import data_helper
from utils import utils
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, regularizers
from keras.models import Model
import keras.backend as kb
from keras.optimizers import *
import numpy as np



train_path = '/home/raymond/Downloads/data/sts-train.csv'
dev_path = '/home/raymond/Downloads/data/sts-dev.csv'
test_path = '/home/raymond/Downloads/data/sts-test.csv'
embedding_path = '/home/raymond/Downloads/data/glove.6B.300d.txt'

seq_length = 30
class_num = 6
embedding_size = 300
filter_sizes = [1, 2, 3]
filter_num = 300
batch_size = 64
epochs_num = 32
# drop_out_rate = 0.5
regularizer_rate = 0.

print ("loading data...")
train_sources, train_targets, train_scores = data_helper.load_sts_data(train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_sts_data(dev_path)
test_sources, test_targets, test_scores = data_helper.load_sts_data(test_path)

word2idx, word_embeddings = data_helper.load_embedding(embedding_path, True)
train_sources = utils.word2id(train_sources, word2idx, seq_length)
train_targets = utils.word2id(train_targets, word2idx, seq_length)
dev_sources = utils.word2id(dev_sources, word2idx, seq_length)
dev_targets = utils.word2id(dev_targets, word2idx, seq_length)
test_sources = utils.word2id(test_sources, word2idx, seq_length)
test_targets = utils.word2id(test_targets, word2idx, seq_length)

train_score_probs = utils.build_porbs(train_scores, class_num)
dev_score_probs = utils.build_porbs(dev_scores, class_num)
test_score_probs = utils.build_porbs(test_scores, class_num)



print('build model...')
source_input = Input(batch_shape=(None, seq_length))
target_input = Input(batch_shape=(None, seq_length))

embedding_layer = Embedding(len(word2idx),
                            embedding_size,
                            input_length=seq_length,
                            weights=[word_embeddings],
                            trainable=False)

source = embedding_layer(source_input)
target = embedding_layer(target_input)

def he_uniform(shape, dtype=None):
    scale = np.sqrt(6. / len(shape))
    return kb.random_uniform(shape, -1 * scale, scale, dtype=dtype)

source_pool_output = []
target_pool_output = []
all_filter_num = len(filter_sizes) * filter_num
for filter_size in filter_sizes:
    conv = Conv1D(filter_num, filter_size, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))
    max_pool = MaxPooling1D(seq_length - filter_size + 1)
    reshape = Reshape([filter_num])

    source_conv = conv(source)
    source_pool = max_pool(source_conv)

    target_conv = conv(target)
    target_pool = max_pool(target_conv)

    source_pool_output.append(reshape(source_pool))
    target_pool_output.append(reshape(target_pool))

if len(filter_sizes) != 1:
    source_conc = Concatenate()(source_pool_output)
    target_conc = Concatenate()(target_pool_output)
else:
    source_conc = source_pool_output[0]
    target_conc = target_pool_output[0]

abs = Lambda(lambda x: kb.abs(x))
sdv = Concatenate()([abs(Subtract()([source_conc, target_conc])), Multiply()([source_conc, target_conc])])

output1 = Dense(all_filter_num, activation='tanh', kernel_regularizer=regularizers.l2(regularizer_rate))(sdv)
# output1 = Dropout(drop_out_rate)(output1)
logits = Dense(class_num, activation='softmax', kernel_regularizer=regularizers.l2(regularizer_rate))(output1)


model = Model(inputs=[source_input, target_input], outputs=logits)

def kl_distance(y_true, y_pred):
    y_true = kb.clip(y_true, 1e-6, 1.)
    y_pred = kb.clip(y_pred, 1e-6, 1.)
    avg_distance = (kb.sum(y_true * kb.log(y_true / y_pred), axis=1) +
                    kb.sum(y_pred * kb.log(y_pred / y_true), axis=1)) / 2.0
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

results = model.evaluate([test_sources, test_targets], test_score_probs, batch_size=len(test_score_probs))
print('test loss: ' + str(results[0]) + ' - test pearson: ' + str(results[1]))
