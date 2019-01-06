#/usr/bin/env python
#coding=utf-8
import numpy as np
import jieba
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda, Add, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, regularizers, AveragePooling1D, Masking, GRU
from keras.models import Model
import keras.backend as kb
from keras.utils import to_categorical
import sys
import keras

seq_length = 30
class_num = 2
embedding_size = 50
filter_sizes = [1, 2, 3]
filter_num = 50
drop_out_rate = 0.5
regularizer_rate = 0.004

def build_model(word2idx):

    print('build model...')
    source_input = Input(batch_shape=(None, seq_length))
    target_input = Input(batch_shape=(None, seq_length))

    embedding_layer = Embedding(len(word2idx),
                                embedding_size,
                                input_length=seq_length)

    source = embedding_layer(source_input)
    target = embedding_layer(target_input)

    source_outputs = []
    target_outputs = []
    all_filter_num = len(filter_sizes) * filter_num
    for filter_size in filter_sizes:
        conv = Conv1D(filter_num, filter_size, activation='relu', kernel_initializer='he_uniform',
                      bias_initializer='he_uniform')
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

    model = Model(inputs=[source_input, target_input], outputs=logits)
    return model
def load_word2idx(path):
    word2idx = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            word2idx[temp[0]] = int(temp[1])
    return word2idx
def process(inpath, outpath):
    jieba.load_userdict("./user.dict")
    jieba.del_word('元用')
    word2idx = load_word2idx("./word2idx.dict")

    source_inputs = []
    target_inputs = []
    all_lineno = []
    with open(inpath, 'r', encoding='utf-8') as fin:
        for line in fin:
            lineno, sen1, sen2, score= line.strip().split('\t')
            idx1 = [word2idx.get(w, word2idx['<UNK>']) for w in jieba.cut(sen1) if w.strip()]
            idx2 = [word2idx.get(w, word2idx['<UNK>']) for w in jieba.cut(sen2) if w.strip()]
            all_lineno.append(lineno)
            def standard_length(idx):
                if len(idx) > seq_length:
                    idx = idx[:seq_length]
                else:
                    for i in range(len(idx), seq_length):
                        idx.append(word2idx['<UNK>'])
                return idx
            source_inputs.append(standard_length(idx1))
            target_inputs.append(standard_length(idx2))

    model = build_model(word2idx)
    model.load_weights('./atec.model')
    results = model.predict([np.array(source_inputs), np.array(target_inputs)])

    with open(outpath, 'w') as fout:
        for i, t in enumerate(results):
            target = np.argmax(t)
            if target== 0:
                fout.write(all_lineno[i] + '\t0\n')
            else:
                fout.write(all_lineno[i] + '\t1\n')


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
