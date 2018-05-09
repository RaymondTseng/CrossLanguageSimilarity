import data_helper
from utils import utils
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda, Add, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Masking
from keras.models import Model
import keras.backend as kb
from keras.optimizers import *

# train_path = '/home/raymond/Downloads/semeval_en/semeval.train.txt'
# dev_path = '/home/raymond/Downloads/semeval_en/semeval.dev.txt'
# test_path = '/home/raymond/Downloads/semeval_en/semeval.test.txt'
train_path = '/home/raymond/Downloads/data/sts-train.csv'
dev_path = '/home/raymond/Downloads/data/sts-dev.csv'
test_path = '/home/raymond/Downloads/data/sts-test.csv'
# embedding_path = '/home/raymond/Downloads/data/glove.6B.300d.txt'
embedding_path = '/media/raymond/CE687D43687D2B7B/data/paragram_300_sl999.txt'

seq_length = 30
class_num = 6
embedding_size = 300
filter_sizes = [1, 2, 3]
filter_num = 300
batch_size = 64
epochs_num = 32
drop_out_rate = 0.5
regularizer_rate = 0.004




tracks = ['AR-AR', 'AR-EN', 'SP-SP', 'SP-EN', 'SP-EN-WMT', 'EN-EN', 'EN-TR']


print ("loading data...")
# train_sources, train_targets, train_scores = data_helper.load_cross_lang_sentence_data(train_path, True)
# dev_sources, dev_targets, dev_scores = data_helper.load_cross_lang_sentence_data(dev_path, True)
# test_sources, test_targets, test_scores = data_helper.load_cross_lang_sentence_data(test_path, False)
train_sources, train_targets, train_scores = data_helper.load_sts_data(train_path)
dev_sources, dev_targets, dev_scores = data_helper.load_sts_data(dev_path)
test_sources, test_targets, test_scores = data_helper.load_sts_data(test_path)

word2idx, word_embeddings = data_helper.load_embedding2(embedding_path, True)


# word to id
train_sources, train_sources_length = utils.word2id(train_sources, word2idx, seq_length)
train_targets, train_targets_length = utils.word2id(train_targets, word2idx, seq_length)


dev_sources, dev_sources_length = utils.word2id(dev_sources, word2idx, seq_length)
dev_targets, dev_targets_length = utils.word2id(dev_targets, word2idx, seq_length)

test_sources, test_sources_length = utils.word2id(test_sources, word2idx, seq_length)
test_targets, test_targets_length = utils.word2id(test_targets, word2idx, seq_length)

train_score_probs = utils.build_porbs(train_scores, class_num)
dev_score_probs = utils.build_porbs(dev_scores, class_num)
test_score_probs = utils.build_porbs(test_scores, class_num)

def kl_distance(y_true, y_pred):
    y_true = kb.clip(y_true, 1e-10, 1.)
    y_pred = kb.clip(y_pred, 1e-10, 1.)
    avg_distance = (kb.sum(y_true * kb.log(y_true / y_pred), axis=1) +
                    kb.sum(y_pred * kb.log(y_pred / y_true), axis=1)) / 2.0
    return kb.mean(avg_distance)

def pearson(y_true, y_pred):
    scores = kb.reshape(K.arange(0, 6, dtype='float32'), [class_num, 1])
    y_true = kb.reshape(kb.dot(y_true, scores), [-1])
    y_pred = kb.reshape(kb.dot(y_pred, scores), [-1])

    mid1 = kb.mean(y_true * y_pred) - \
           kb.mean(y_true) * kb.mean(y_pred)

    mid2 = kb.sqrt(kb.mean(kb.square(y_true)) - kb.square(kb.mean(y_true))) * \
           kb.sqrt(kb.mean(kb.square(y_pred)) - kb.square(kb.mean(y_pred)))

    pearson = mid1 / mid2
    return pearson

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

mask = Masking()
lstm = LSTM(filter_num)
source_lstm = lstm(mask(source))
target_lstm = lstm(mask(target))

abs = Lambda(lambda x: kb.abs(x))
h_sub = abs(Subtract()([source_lstm, target_lstm]))
h_mul = Multiply()([source_lstm, target_lstm])

h_conc = Concatenate()([h_sub, h_mul])

logits = Dense(class_num, activation='softmax')(h_conc)


max_dev_pearson = 0.
max_test_pearson = 0.
model = Model(inputs=[source_input, target_input], outputs=logits)
model.compile(optimizer='Adam', loss=kl_distance, metrics=[pearson])
for epoch in range(epochs_num):
    print('epoch num %s ' % epoch)
    model.fit([train_sources, train_targets], train_score_probs, epochs=1, batch_size=batch_size,
              validation_data=([dev_sources, dev_targets], dev_score_probs))

    results = model.evaluate([dev_sources, dev_targets], dev_score_probs, batch_size=len(dev_score_probs))
    print('--- dev loss: %.4f --- dev pearson: %.4f ---' % (results[0], results[1]))
    if results[1] > max_dev_pearson:
        max_dev_pearson = results[1]

    results = model.evaluate([test_sources, test_targets], test_score_probs, batch_size=len(test_score_probs))
    print('--- test loss: %.4f --- test pearson: %.4f ---' % (results[0], results[1]))
    if results[1] > max_test_pearson:
        max_test_pearson = results[1]
    print('')

print('--- max dev pearson: %.4f --- max test pearson: %.4f ---' % (max_dev_pearson, max_test_pearson))
# for epoch in range(epochs_num):
#     print('epoch num %s ' % epoch)
#     model.fit([train_sources, train_targets], train_score_probs, epochs=1, batch_size=batch_size,
#               validation_data=([dev_sources, dev_targets], dev_score_probs))
#
#     results = model.evaluate([dev_sources, dev_targets], dev_score_probs, batch_size=len(dev_score_probs))
#     print('--- dev loss: %.4f --- dev pearson: %.4f ---' % (results[0], results[1]))
#     if results[1] > max_dev_pearson:
#         max_dev_pearson = results[1]
#
#     temp_loss = 0.
#     temp_pearson = 0.
#     for i in range(7):
#         start = i * 250
#         end = start + 250
#         results = model.evaluate([test_sources[start:end], test_targets[start:end]], test_score_probs[start:end],
#                                  batch_size=250)
#         print(tracks[i] + ' --- test loss: %.4f --- test pearson: %.4f ---' % (results[0], results[1]))
#         temp_loss += results[0]
#         temp_pearson += results[1]
#     print('')
#     temp_loss /= 7
#     temp_pearson /= 7
#     print('Primary --- test pearson: %.4f --- test loss: %.4f ---' % (temp_pearson, temp_loss))
#     if temp_pearson > max_test_pearson:
#         max_test_pearson = temp_pearson
#     print('')
#
#
# print('--- max dev pearson: %.4f --- max test pearson: %.4f ---' % (max_dev_pearson, max_test_pearson))
