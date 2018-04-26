
import data_helper
from utils import utils
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda, Add, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, regularizers, AveragePooling1D, Masking, GRU
from keras.models import Model
import keras.backend as kb
from keras.optimizers import *
import numpy as np



graph_path = '/home/raymond/Downloads/semeval_en/semeval.graph.txt'
embedding_path = '/home/raymond/Downloads/data/glove.6B.300d.txt'

seq_length = 30
class_num = 6
embedding_size = 300
filter_sizes = [1, 2, 3]
filter_num = 300
batch_size = 64
tune_epochs_num = 300
drop_out_rate = 0.5
regularizer_rate = 0.004





print ("loading data...")

graph_sources, graph_targets, graph_scores = data_helper.load_cross_lang_sentence_data(graph_path, False)


word2idx, word_embeddings = data_helper.load_embedding(embedding_path, True)


graph_sources, graph_sources_length = utils.word2id(graph_sources, word2idx, seq_length)
graph_targets, graph_targets_length = utils.word2id(graph_targets, word2idx, seq_length)



graph_score_probs = utils.build_porbs(graph_scores, class_num)


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

source_outputs = []
target_outputs = []
all_filter_num = len(filter_sizes) * filter_num
for filter_size in filter_sizes:
    conv = Conv1D(filter_num, filter_size, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform')
    max_pool = MaxPooling1D(seq_length - filter_size + 1)
    reshape = Reshape([filter_num])

    source_conv = conv(source)
    target_conv = conv(target)

    source_sdv = reshape(max_pool(source_conv))
    target_sdv = reshape(max_pool(target_conv))

    source_outputs.append(source_sdv)
    target_outputs.append(target_sdv)

mask = Masking()
gru = GRU(filter_num, dropout=drop_out_rate, recurrent_dropout=0.2)

source_outputs.append(gru(mask(source)))
target_outputs.append(gru(mask(target)))

source_conc = Concatenate()(source_outputs)
target_conc = Concatenate()(target_outputs)


abs = Lambda(lambda x: kb.abs(x))
h_sub = abs(Subtract()([source_conc, target_conc]))
h_mul = Multiply()([source_conc, target_conc])


w1 = Dense(all_filter_num, activation='tanh', kernel_regularizer=regularizers.l2(regularizer_rate),
                  bias_regularizer=regularizers.l2(regularizer_rate))
w2 = Dense(all_filter_num, activation='tanh', kernel_regularizer=regularizers.l2(regularizer_rate),
                  bias_regularizer=regularizers.l2(regularizer_rate))


sdv = Add()([w1(h_sub), w2(h_mul)])


output = Dense(all_filter_num, activation='tanh', kernel_regularizer=regularizers.l2(regularizer_rate),
                  bias_regularizer=regularizers.l2(regularizer_rate))(sdv)
output = Dropout(drop_out_rate)(output)
logits = Dense(class_num, activation='softmax', kernel_regularizer=regularizers.l2(regularizer_rate),
                  bias_regularizer=regularizers.l2(regularizer_rate))(output)


max_dev_pearson = 0.
max_test_pearson = 0.
model = Model(inputs=[source_input, target_input], outputs=logits)
model.load_weights('../save/cnn.semeval.model.weights.0.6879')
model.compile(optimizer='Adam', loss=kl_distance, metrics=[pearson])

predict_model = Model(inputs=model.input,
                                     outputs=[model.get_layer('conv1d_3').get_output_at(1),
                                              model.get_layer('max_pooling1d_3').get_output_at(1)])
pre_output = predict_model.predict([graph_sources, graph_targets], batch_size=1)

def get_info(conv_outputs, max_outputs):
    for i in range(len(graph_scores)):
        conv_output = conv_outputs[i]
        max_output = max_outputs[i][0]
        idx = dict()
        for num_index, num in enumerate(max_output):
            for seq_index in range(28):
                if num == conv_output[seq_index][num_index]:
                    if idx.has_key(seq_index):
                        idx[seq_index] += 1
                    else:
                        idx[seq_index] = 1
        print(idx)
        print('--------------------------------------------')

get_info(pre_output[0], pre_output[1])













