import numpy as np
import jieba
from keras.layers import Dense, Input, Concatenate, Subtract, Multiply, Reshape, Lambda, Add, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, regularizers, AveragePooling1D, Masking, GRU
from keras.models import Model
import keras.backend as kb
from keras.utils import to_categorical
path = "atec_nlp_sim_train.csv"


seq_length = 30
class_num = 2
embedding_size = 50
filter_sizes = [1, 2, 3]
filter_num = 50
batch_size = 64
epochs_num = 10
drop_out_rate = 0.5
regularizer_rate = 0.004


def load_data(path):
    source_inputs = []
    target_inputs = []
    targets = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            temp = line.split('\t')
            source_inputs.append(temp[1].strip())
            target_inputs.append(temp[2].strip())
            targets.append(int(temp[3]))
    return source_inputs, target_inputs, targets

source_inputs, target_inputs, targets = load_data(path)


jieba.load_userdict("./user.dict")
jieba.del_word('元用')

word2idx = {}
word2idx['<UNK>'] = 0

def write_word2idx(path):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in word2idx.items():
            f.write(k + '\t' + str(v) + '\n')
    f.close()

def sentence2id(all_data):
    new_all_data = []
    for data in all_data:
        def cut_add(s):
            s.replace('*', 'NUM')
            words = [w for w in jieba.cut(s) if w.strip()]
            idx = []
            for w in words:
                if w not in word2idx:
                    word2idx[w] = len(word2idx)
                idx.append(word2idx[w])
            if len(idx) > seq_length:
                idx = idx[:seq_length]
            else:
                for i in range(len(words), seq_length):
                    idx.append(word2idx['<UNK>'])
            return np.array(idx)
        new_all_data.append(cut_add(data))
    return np.array(new_all_data)

source_inputs = sentence2id(source_inputs)
target_inputs = sentence2id(target_inputs)
targets = to_categorical(targets, num_classes=class_num)




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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for epoch in range(epochs_num):
    model.fit([source_inputs, target_inputs], targets, batch_size=batch_size,
              epochs=epochs_num, validation_split=0.8, class_weight={0: 2, 1: 10})
    model.save('./all_models/atec_epoch_' + str(epoch) + '.model')
write_word2idx('word2idx.dict')
